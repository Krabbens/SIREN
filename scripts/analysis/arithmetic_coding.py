
import struct

class ArithmeticEncoder:
    def __init__(self, precision=32):
        self.precision = precision
        self.full = 1 << precision
        self.half = 1 << (precision - 1)
        self.quarter = 1 << (precision - 2)
        self.mask = self.full - 1
        
        self.low = 0
        self.high = self.mask
        self.underflow_cnt = 0
        self.out = bytearray()
        
    def encode(self, cum_freq, freq, total_freq):
        # Scale Range
        range_Val = (self.high - self.low) + 1
        
        # Update High/Low
        # low = low + (range * cum_freq) // total_freq
        # high = low + (range * freq) // total_freq - 1
        
        # Optimized to avoid overflow
        self.high = self.low + (range_Val * (cum_freq + freq)) // total_freq - 1
        self.low = self.low + (range_Val * cum_freq) // total_freq
        
        # Renormalize
        while True:
            if self.high < self.half:
                # Top bit 0
                self._output_bit(0)
            elif self.low >= self.half:
                # Top bit 1
                self._output_bit(1)
                self.low -= self.half
                self.high -= self.half
            elif self.low >= self.quarter and self.high < (3 * self.quarter):
                # Underflow case
                self.underflow_cnt += 1
                self.low -= self.quarter
                self.high -= self.quarter
            else:
                break
                
            self.low = (self.low << 1) & self.mask
            self.high = ((self.high << 1) & self.mask) | 1
            
    def _output_bit(self, bit):
        self._add_bit(bit)
        while self.underflow_cnt > 0:
            self._add_bit(1 - bit)
            self.underflow_cnt -= 1
            
    def _add_bit(self, bit):
        # Accumulate bits into bytes
        # We need a buffer for current byte?
        # Actually simplest is just list of bits or handled by bitstream wrapper.
        # Impl below assumes simple bit writer.
        # But for speed let's just use a string of '0'/'1' and pack later? 
        # Or better: keep a byte buffer.
        
        if not hasattr(self, 'current_byte'):
            self.current_byte = 0
            self.bit_count = 0
            
        self.current_byte = (self.current_byte << 1) | bit
        self.bit_count += 1
        
        if self.bit_count == 8:
            self.out.append(self.current_byte)
            self.bit_count = 0
            self.current_byte = 0
            
    def finish(self):
        # Output 1 to distinguish end
        self.underflow_cnt += 1
        self._output_bit(0) # Flush
        
        # Flush remaining byte
        if self.bit_count > 0:
            self.current_byte = self.current_byte << (8 - self.bit_count)
            self.out.append(self.current_byte)
            
        return bytes(self.out)

class ArithmeticDecoder:
    def __init__(self, bitstream, precision=32):
        self.precision = precision
        self.full = 1 << precision
        self.half = 1 << (precision - 1)
        self.quarter = 1 << (precision - 2)
        self.mask = self.full - 1
        
        self.low = 0
        self.high = self.mask
        self.value = 0
        
        # Initialize Reader
        self.bitstream = bitstream
        self.byte_idx = 0
        self.bit_idx = 0
        self.bit_count = 0 # Bits consumed
        
        # Read initial 'precision' bits into value
        for _ in range(precision):
            self.value = (self.value << 1) & self.mask
            self.value |= self._read_bit()
            
    def _read_bit(self):
        if self.byte_idx >= len(self.bitstream):
            return 0 # Integrity check needed? Padding usually 0
            
        byte = self.bitstream[self.byte_idx]
        bit = (byte >> (7 - self.bit_idx)) & 1
        
        self.bit_idx += 1
        if self.bit_idx == 8:
            self.byte_idx += 1
            self.bit_idx = 0
            
        return bit
        
    def decode(self, cum_freqs, freqs, total_freq):
        # cum_freqs list corresponding to symbols?
        # Standard: find symbol such that 
        # low + range*low_count/total <= value < low + range*high_count/total
        
        range_Val = (self.high - self.low) + 1
        
        # Target scaled value
        # value = low + (range * scaled_value) // total
        # scaled_value = ((value - low + 1) * total - 1) // range
        
        # We need to find symbol `s` such that cum_freq[s] <= scaled < cum_freq[s+1]
        
        count = ((self.value - self.low + 1) * total_freq - 1) // range_Val
        
        # Search for symbol
        # cum_freqs is cumulative. cum_freqs[s] is freq sum BEFORE s
        # so we find s where cum_freqs[s] <= count < cum_freqs[s] + freqs[s]
        
        # Assuming cum_freqs is array like [0, f0, f0+f1, ...]
        # We can binary search or linear search if V is small.
        # Since V is small (~6500 for semantic, but we split into FSQ levels? 
        # No, EntropyModel operates on FSQ indices? 
        # RFSQ is residuals. We have multiple levels.
        # EntropyModel predicts indices for EACH level?
        # NO. EntropyModel is just a Transformer.
        # It takes ONE token index per step?
        # The `EntropyModel` code showed `sem_tokens` (vocab size).
        # `fsq_levels: [3,3,3,3,3,3,3,3]` (prod=6561).
        # The input to `EntropyModel` IS the single index from FSQ (0..6560).
        # So vocab size is ~6561. Linear search is slow.
        # But python loop for 30s audio (400 frames) is fine.
        
        symbol = 0
        # Simple linear search (optimize later if needed)
        # Actually cum_freq provided by user?
        # Let's pass cum_freqs array (len V+1).
        for s in range(len(cum_freqs) - 1):
            if cum_freqs[s+1] > count:
                symbol = s
                break
        
        sym_cum = cum_freqs[symbol]
        sym_freq = freqs[symbol]
        
        # Update State
        self.high = self.low + (range_Val * (sym_cum + sym_freq)) // total_freq - 1
        self.low = self.low + (range_Val * sym_cum) // total_freq
        
        # Renormalize
        while True:
            if self.high < self.half:
                pass # match 0
            elif self.low >= self.half:
                self.value -= self.half
                self.low -= self.half
                self.high -= self.half
            elif self.low >= self.quarter and self.high < (3 * self.quarter):
                self.value -= self.quarter
                self.low -= self.quarter
                self.high -= self.quarter
            else:
                break
                
            self.low = (self.low << 1) & self.mask
            self.high = ((self.high << 1) & self.mask) | 1
            self.value = ((self.value << 1) & self.mask) | self._read_bit()
            
        return symbol
