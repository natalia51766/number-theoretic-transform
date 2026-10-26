# number_theoretic_transform

Exact integer convolution over the finite field `GF(998244353)`, using a
radix-2 Number Theoretic Transform. Standard library only.

```python
from number_theoretic_transform import convolve

# convolve returns exact signed integers when max_abs_value bounds the inputs.
print(convolve([1, 2, 3], [4, 5, 6], max_abs_value=10))
# -> [4, 13, 28, 27, 18]
```

Also exported: `P`, `primitive_root_of_unity(n)`, `ntt(a)`, `intt(a)`, and
`ConvolutionOverflow`.

## Why this exists

FFT-based convolution of integer sequences is exact when the underlying field
has a prime modulus `p` larger than twice the largest possible output
coefficient. This library fixes one such prime, `p = 998244353`, whose
`p - 1` is divisible by `2**23`, so any power-of-two transform length up to
`2**23` is supported with a primitive root of unity.

The trade-off: there is exactly one prime, and `p` is about `10**9`, so any
convolution whose true integer coefficients could reach or exceed `p/2` will
collide modulo `p`. There is no fallback to a second prime or CRT lifting.
If you need bigger coefficients, use a bigger prime somewhere else.

## The awkward edge

`convolve(a, b, max_abs_value=k)` raises `ConvolutionOverflow` when the
worst-case coefficient `k * k * min(len(a), len(b))` reaches `p / 2`. Call
`convolve(a, b)` without the bound if you want raw residues in `[0, p)` and are
prepared to interpret them yourself.

`intt` requires its input length to already be a power of two; it does not
zero-pad. `ntt` always pads to the next power of two.

## Performance

The window keeps a bounded buffer, so `push` is constant time and memory does not
grow with the length of the stream. `peak` and `trough` are linear in the window
size, which is the trade that keeps `push` cheap.

