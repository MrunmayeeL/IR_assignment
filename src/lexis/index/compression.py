"""Compression algorithms for inverted index postings and integers.

Supports:
- Uncompressed (standard 4-byte / 32-bit integers)
- Gap encoding (delta differences)
- Variable-Byte (VB) encoding
- Elias Gamma encoding
- Dictionary string compression (concatenated term buffer + offsets)
"""

import struct
from typing import List, Tuple, Dict


class GapEncoder:
    """Encodes a strictly increasing list of integer IDs as positive gaps (deltas)."""

    @staticmethod
    def encode(numbers: List[int]) -> List[int]:
        if not numbers:
            return []
        gaps = [numbers[0]]
        for i in range(1, len(numbers)):
            gaps.append(numbers[i] - numbers[i - 1])
        return gaps

    @staticmethod
    def decode(gaps: List[int]) -> List[int]:
        if not gaps:
            return []
        numbers = [gaps[0]]
        for i in range(1, len(gaps)):
            numbers.append(numbers[-1] + gaps[i])
        return numbers


class VariableByteEncoder:
    """Variable-byte encoder and decoder for lists of positive integers."""

    @staticmethod
    def encode_number(number: int) -> bytes:
        if number < 0:
            raise ValueError(f"Variable-byte requires non-negative integers, got {number}")
        bytes_list = []
        while True:
            bytes_list.insert(0, number % 128)
            if number < 128:
                break
            number //= 128
        bytes_list[-1] += 128  # Set high bit on terminal byte
        return bytes(bytes_list)

    @classmethod
    def encode(cls, numbers: List[int]) -> bytes:
        """Encode a list of positive integers into bytes."""
        byte_stream = bytearray()
        for n in numbers:
            byte_stream.extend(cls.encode_number(n))
        return bytes(byte_stream)

    @staticmethod
    def decode(bytestream: bytes) -> List[int]:
        """Decode bytes back into a list of integers."""
        numbers: List[int] = []
        n = 0
        for byte in bytestream:
            if byte < 128:
                n = 128 * n + byte
            else:
                n = 128 * n + (byte - 128)
                numbers.append(n)
                n = 0
        return numbers


class GammaEncoder:
    """Elias Gamma encoder and decoder for positive integers (n >= 1).
    
    A positive integer n is represented as:
    1. Unary code for 1 + floor(log2(n)): (length - 1) zeros followed by a 1.
    2. Followed by the (length - 1) low-order bits of the binary representation of n.
    """

    @staticmethod
    def encode_number(n: int) -> str:
        if n < 1:
            raise ValueError(f"Gamma encoding requires positive integers >= 1, got {n}")
        if n == 1:
            return "1"
        b = bin(n)[2:]  # e.g., bin(9) -> '1001'
        offset = b[1:]  # e.g., '001'
        unary = "0" * len(offset) + "1"
        return unary + offset

    @classmethod
    def encode_to_bits(cls, numbers: List[int]) -> str:
        bit_str = []
        for num in numbers:
            bit_str.append(cls.encode_number(num))
        return "".join(bit_str)

    @classmethod
    def encode(cls, numbers: List[int]) -> bytes:
        """Encode a list of integers >= 0 into bytes (shifted by +1 for Gamma positive constraint)."""
        shifted = [n + 1 for n in numbers]
        bit_str = cls.encode_to_bits(shifted)
        # Pad with zeros to complete the last byte
        padding = (8 - len(bit_str) % 8) % 8
        bit_str += "0" * padding

        byte_arr = bytearray()
        for i in range(0, len(bit_str), 8):
            byte_arr.append(int(bit_str[i : i + 8], 2))
        return bytes(byte_arr)

    @staticmethod
    def decode_from_bits(bit_str: str, count: int) -> List[int]:
        numbers: List[int] = []
        idx = 0
        total_len = len(bit_str)

        while idx < total_len and len(numbers) < count:
            # Read unary part: count zeros until 1
            zeros = 0
            while idx < total_len and bit_str[idx] == "0":
                zeros += 1
                idx += 1
            if idx >= total_len:
                break
            # Skip the '1'
            idx += 1
            if zeros == 0:
                numbers.append(1)
            else:
                if idx + zeros > total_len:
                    break
                offset_str = bit_str[idx : idx + zeros]
                idx += zeros
                val = (1 << zeros) + int(offset_str, 2)
                numbers.append(val)
        return numbers

    @classmethod
    def decode(cls, bytestream: bytes, count: int) -> List[int]:
        bit_str = "".join(f"{b:08b}" for b in bytestream)
        decoded = cls.decode_from_bits(bit_str, count)
        return [n - 1 for n in decoded]


class UncompressedEncoder:
    """Uncompressed 32-bit unsigned integer encoding (4 bytes per int)."""

    @staticmethod
    def encode(numbers: List[int]) -> bytes:
        return struct.pack(f"<{len(numbers)}I", *numbers)

    @staticmethod
    def decode(bytestream: bytes) -> List[int]:
        if not bytestream:
            return []
        count = len(bytestream) // 4
        return list(struct.unpack(f"<{count}I", bytestream))


class DictionaryCompressor:
    """Compresses dictionary terms into a single string buffer with pointers/offsets."""

    @staticmethod
    def compress(terms: List[str]) -> Tuple[bytes, List[int]]:
        """Concatenate terms into a UTF-8 byte buffer and return (buffer, offsets)."""
        buffer = bytearray()
        offsets = []
        for term in terms:
            offsets.append(len(buffer))
            buffer.extend(term.encode("utf-8"))
        return bytes(buffer), offsets

    @staticmethod
    def decompress(buffer: bytes, offsets: List[int], lengths: List[int]) -> List[str]:
        terms = []
        for off, length in zip(offsets, lengths):
            terms.append(buffer[off : off + length].decode("utf-8"))
        return terms
