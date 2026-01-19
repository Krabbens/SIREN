
import struct

class RangeCoder:
    def __init__(self, precision=32):
        self.top = 1 << 24
        self.bottom = 1 << 16
        self.precision = precision
        self.max_range = 1 << precision
        
    def encode(self, symbols, cdfs, filename):
        """
        symbols: (N,) int
        cdfs: (N, M+1) float, [0...1]
        """
        low = 0
        range_ = 0xFFFFFFFF
        output = bytearray()
        
        # Buffer for output
        # We output bytes
        
        def out_byte(b):
            output.append(b)

        for i, sym in enumerate(symbols):
            sym = int(sym)
            cdf = cdfs[i]
            
            # Quantize CDF to max_range
            # Make sure it's strictly increasing and ends at max_range
            # cdf_int[k] = floor(cdf[k] * max_range)
            # but force 0 and max_range
            
            count_total = self.max_range
            
            # Find low_count and high_count for symbol
            # This is slow, but we pre-calculate or specific logic
            # cdf is float array
            
            # Convert float cdf to int
            # low = cdf[sym]
            # high = cdf[sym+1]
            
            sym_low = int(cdf[sym] * self.max_range)
            sym_high = int(cdf[sym+1] * self.max_range)
            
            # Ensure valid range
            if sym_low == sym_high:
                # Give at least 1 slot if possible, or error
                # In float, this means prob is very small.
                sym_high = sym_low + 1
            if sym_low >= sym_high:
                 sym_high = sym_low + 1
            
            # Update Range
            new_range = range_ // count_total
            low = low + new_range * sym_low
            range_ = new_range * (sym_high - sym_low)
            
            # Renormalize
            while True:
                if (low ^ (low + range_)) < (1 << 24):
                    # Top bits match
                    out_byte(low >> 24)
                    low = (low << 8) & 0xFFFFFFFF
                    range_ = (range_ << 8) & 0xFFFFFFFF
                elif range_ < (1 << 16):
                    # Range too small, no convergence yet?
                    # This happens if range becomes < bottom?
                    # Standard renormalization:
                    # If range < bottom, we must expand?
                    # The condition above (top match) handles most.
                    # This part handles "straddle" if implemented?
                    # For simplicity, implementing standard carry-less rangecoder 
                    # requires careful handling.
                    # Let's use a simpler implementation pattern:
                    # https://github.com/luciddreams/rangecoder/blob/master/rangecoder.py
                    # Or just:
                    out_byte(low >> 24) # Force output? No this is wrong if not converged.
                    # Risk of bug here writing from scratch.
                    pass # Trusting standard loop logic below
                    break
                else:
                    break
                    
        # Flush
        for _ in range(4):
            out_byte(low >> 24)
            low = (low << 8) & 0xFFFFFFFF
            
        with open(filename, 'wb') as f:
            f.write(output)


class RangeEncoder:
    def __init__(self, filename, precision=16):
        self.low = 0
        self.range = 0xFFFFFFFF
        self.file = open(filename, 'wb')
        self.precision = precision
        self.total_freq = 1 << precision
        self.follow_bytes = 0 
        
    def encode(self, sym_low_freq, sym_freq):
        # sym_low_freq: cumulative freq of symbols before
        # sym_freq: freq of this symbol
        # All normalized to total_freq
        
        new_range = self.range // self.total_freq
        self.low += sym_low_freq * new_range
        self.range = sym_freq * new_range
        
        while True:
            if (self.low ^ (self.low + self.range)) < (1 << 24):
                # Top byte matches - output it
                self._bit_plus_follow(self.low >> 24)
                self.low = (self.low << 8) & 0xFFFFFFFF
                self.range = (self.range << 8) & 0xFFFFFFFF
            elif self.range < (1 << 16):
                # Range too small, need to renormalize
                # This is the "straddle" case for carry handling
                # low < 0.75 and low+range > 0.25 ?
                # Standard impl:
                # low = low but range expanded? 
                # This is tricky without carry state.
                # Actually, standard Martin implementation uses 'follow_bytes' to handle carry.
                pass
                break
            else:
                break
                
    def _bit_plus_follow(self, b):
        # b is byte
        self.file.write(bytes([b]))
        while self.follow_bytes > 0:
            temp = b + 1
            # Handle carry? In simple byte output, b is 8 bits.
            # If we carry, we need to go back?
            # Range coding is hard to write from scratch without errors.
            pass

# To avoid debugging a Range Coder from scratch, I will use a library if possible.
# `import torchac` - I installed it.
# Can I use `torchac.encode_float_cdf` for the WHOLE file in compress,
# and for decompress, implement a dummy?
# NO, user wants to verify it works.

# Backtracking: `test_torchac.py` did NOT finish?
# If `torchac` works, we can try to use it chunk-by-chunk?
# Or just accept it produces a bytestream.
# If I can't stream decode, I can't implement autoregressive decompression with `torchac` easily.

# Is there ANY Python arithmetic coding library?
# `uv pip install context-based-adaptive-binary-arithmetic-coding`? No.
# `uv pip install arithmetic_coding`
# Let's try to install a python package for this.
