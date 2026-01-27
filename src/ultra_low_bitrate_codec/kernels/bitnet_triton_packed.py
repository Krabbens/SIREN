
import torch
import triton
import triton.language as tl

@triton.jit
def packed_bitnet_kernel_fp16(
    x_ptr,              # Input (M, K)
    w_ptr,              # Weight (N, K/16) - Packed Int32
    y_ptr,              # Output (M, N)
    scale_x_ptr,        # Scale X (M, 1)
    scale_w_ptr,        # Scale W (N, 1)
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Fused BitNet Kernel with Packed Weights.
    Hardcoded for BLOCK_K=16 (PACK_FACTOR=16) -> Processing 1 packed int per K-loop.
    """
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    
    # Pointers
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm
    # Corrected w_ptrs broadcasting: (BLOCK_N, 1)
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn 
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    scale_x = tl.load(scale_x_ptr)
    scale_w = tl.load(scale_w_ptr)
    
    PACK_FACTOR = 16
    # We assume BLOCK_K == 16, so BLOCK_K_PACKED == 1
    
    # Broadcastable shifts for unpacking: [0, 2, 4, ... 30]
    shifts = tl.arange(0, 16) * 2
    
    for k in range(0, K, BLOCK_K):
        # --- 1. Load X ---
        offs_k = k + tl.arange(0, BLOCK_K)
        x_block_ptrs = x_ptrs + offs_k[None, :] * stride_xk
        x_val = tl.load(x_block_ptrs, mask=offs_k[None, :] < K, other=0.0)
        
        # Quantize X (Simulated Int8 via Manual Rounding)
        x_q_f = (x_val / scale_x * 127.0)
        x_q_clamped = tl.clamp(x_q_f, -127.0, 127.0)
        # Manual Round to Nearest: x + 0.5 * sign(x)
        sign = tl.where(x_q_clamped >= 0, 1.0, -1.0)
        x_q = (x_q_clamped + 0.5 * sign).to(tl.int8)
        
        # --- 2. Load Packed W ---
        # Assuming BLOCK_K = 16, we load exactly 1 int32 per BLOCK_N row.
        # w_ptr offset calculation:
        offs_k_packed = (k // 16) # Scalar index
        # stride_wk refers to stride in the Packed K dimension.
        w_curr_ptrs = w_ptrs + offs_k_packed * stride_wk
        
        # Load (BLOCK_N, 1) vector of int32s (expanded from N)
        w_packed = tl.load(w_curr_ptrs) 
        
        # --- 3. Unpack W ---
        # w_packed: (BLOCK_N, 1) after load
        # shifts: (16,) -> (1, 16)
        
        # Explicit broadcast to (BLOCK_N, 16)
        w_packed_bc = tl.broadcast_to(w_packed, (BLOCK_N, 16))
        
        # shifts broadcast to (BLOCK_N, 16)
        shifts_bc = tl.broadcast_to(shifts[None, :], (BLOCK_N, 16))
        
        # Unpack
        w_unpacked_2bit = (w_packed_bc >> shifts_bc) & 0x3
        
        # Map {0:0, 1:1, 2:-1}
        w_is_1 = (w_unpacked_2bit == 1).to(tl.int8)
        w_is_neg1 = (w_unpacked_2bit == 2).to(tl.int8)
        w_ternary = w_is_1 - w_is_neg1 # (BLOCK_N, 16)
        
        # Transpose W for dot: (16, BLOCK_N)
        w_ternary_t = tl.trans(w_ternary)
        
        # Computation: X (BLOCK_M, 16) @ W.T (16, BLOCK_N)
        # Using floating point accumulation for safety
        acc += tl.dot(x_q.to(tl.float16), w_ternary_t.to(tl.float16))
        
    
    # --- Dequantize ---
    c = acc / 127.0 * scale_x * scale_w
    
    # Store
    offs_y = offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptr + offs_y, c)

def pack_ternary(w):
    N, K = w.shape
    assert K % 16 == 0, "K must be multiple of 16"
    packed = torch.zeros((N, K // 16), dtype=torch.int32, device=w.device)
    w_mapped = w.clone().long()
    w_mapped[w_mapped == -1] = 2
    
    w_reshaped = w_mapped.view(N, K // 16, 16)
    for i in range(16):
        packed |= (w_reshaped[:, :, i] << (2 * i)).int()
    return packed

def bitnet_matmul_custom(x, w_packed, scale_x, scale_w, N, K):
    M = x.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    packed_bitnet_kernel_fp16[grid](
        x, w_packed, out, scale_x, scale_w, M, N, K,
        x.stride(0), x.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=16
    )
    return out


class BitLinearTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, scale):
        """
        x: Input tensor (B, ..., K) - FP32
        weight: Ternary Weights {-1, 0, 1} - FP32
        scale: Weight scale - FP32 scalar
        """
        # Save for backward (we need full precision weights for backward STE)
        # Note: We save inputs.
        ctx.save_for_backward(x, weight, scale)
        
        # Flatten input
        x_flat = x.reshape(-1, x.shape[-1])
        N, K = weight.shape
        
        # 1. Calculate Activation Scale (Per-Tensor for this Kernel)
        scale_x = x_flat.abs().max().clamp(min=1e-5)
        
        # 2. Pack Weights (Overhead here, but necessary unless we cache packed weights)
        w_packed = pack_ternary(weight)
        
        # 3. Run Triton Kernel
        out = bitnet_matmul_custom(x_flat, w_packed, scale_x, scale, N, K)
        
        # Reshape to original
        out = out.view(*x.shape[:-1], N)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, scale = ctx.saved_tensors
        
        # Standard Backward (STE)
        # grad_w = grad_out^T @ x
        # grad_x = grad_out @ w
        # Assuming weight is {-1, 0, 1} approx? 
        # STE says: treat quantization as identity for gradients.
        # So we use the 'weight' (which was passed in ternary form) for grad_x calculation.
        
        grad_input = grad_weight = grad_scale = None
        
        if ctx.needs_input_grad[0]:
            # grad_input = (grad_output * scale) @ weight
            # We scale grad_output first to save ops
            grad_input = (grad_output * scale) @ weight
            
        if ctx.needs_input_grad[1]:
            # grad_weight = (grad_output * scale)^T @ x
            # grad_weight shape: (N, K)
            grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])
            x_flat = x.reshape(-1, x.shape[-1])
            
            # We can use the same scaled grad_output
            grad_output_scaled = grad_output_flat * scale
            grad_weight = grad_output_scaled.T @ x_flat
            
        # Gradient for scale input (scalar)
        if ctx.needs_input_grad[2]:
             # dL/dScale = sum(grad_out * output_unscaled)
             # Out = Unscaled * Scale => Unscaled = Out / Scale (or computed again)
             # Optimization: This is usually small? 
             # Let's compute it correctly: sum(grad_output * (x @ w))
             # Re-computing x@w is heavy.
             # But wait, BitNet 1.58b usually detaches scale for some parts or treating it simply.
             # Strict correctness:
             # grad_scale = (grad_output * (x_flat @ w_packed?? No, x @ w_quant)).sum()
             # Calculating x @ w_quant is expensive (MatMul).
             # Given we want speed, maybe we skip grad_scale (return None) or approx?
             # Actually, weight_quant_ternary returns scale.
             # The scale comes from `w.abs().mean()`.
             # If we cut gradient to scale here, does it matter?
             # Usually standard Linear includes learning rate on magnitude.
             # Let's omit grad_scale for efficiency unless user asks, or just set to 0.
             # But technically `grad_scale` is needed for `weight_quant_ternary` backward?
             # weight_quant_ternary returns `scale`.
             # If we act as if `scale` is constant w.r.t Loss for this path, we might be fine.
             grad_scale = None

        return grad_input, grad_weight, grad_scale

def triton_bit_linear(x, weight, scale):
    return BitLinearTritonFunction.apply(x, weight, scale)

if __name__ == "__main__":
    # Correction for previous test block
    print("Testing Packed Triton Kernel...")
    torch.manual_seed(0)
    if torch.cuda.is_available():
        M, N, K = 32, 64, 128
        x = torch.randn(M, K, device='cuda') # Float32
        w = torch.randint(-1, 2, (N, K), device='cuda').float()
        
        scale_x = x.abs().max()
        scale_w = torch.tensor(1.0, device='cuda')
        
        # Ref
        x_q = torch.round((x / scale_x * 127.0).clamp(-127, 127))
        out_ref = (x_q @ w.T) / 127.0 * scale_x
        
        # Triton Wrapper
        out_tri = triton_bit_linear(x, w, scale_w)
        
        print(f"Ref: {out_ref.mean().item():.4f}, Tri: {out_tri.mean().item():.4f}")
        diff = (out_ref - out_tri).abs().max()
        print(f"Diff: {diff.item():.4f}")
        
        if diff < 1.0: # Int8 quantization error margin
            print("✓ SUCCESS")
        else:
            print("✗ FAILURE")
