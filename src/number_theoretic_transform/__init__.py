"""Number Theoretic Transform over a fixed prime modulus.

This package exposes a small, dependency-free toolkit for performing exact
integer convolution via NTT. The transform operates modulo a single fixed
prime p = 998244353, which is chosen because it supports a power-of-two
transform length up to n = 2**23 via a large primitive root of unity.

Use this when you need fast convolution of integer sequences whose result is
known to fit within the modulus (see README for the bound). Outside that
regime you should reach for a different approach.
"""

from .core import (
    P,
    primitive_root_of_unity,
    ntt,
    intt,
    convolve,
    ConvolutionOverflow,
)

__all__ = [
    "P",
    "primitive_root_of_unity",
    "ntt",
    "intt",
    "convolve",
    "ConvolutionOverflow",
]
