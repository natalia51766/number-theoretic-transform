"""Core NTT implementation over the prime P = 998244353.

The transform is a standard Cooley--Tukey radix-2 iteration, but written
iteratively to avoid recursion overhead and to keep the call stack flat.
All arithmetic is carried out with Python ints; reductions are performed with
``% P`` rather than Montgomery or Barrett reduction. The absolute cost of a
top-level ``%`` is higher than a well-tuned modular-multiplication routine,
but it is correct on every platform and needs no precomputation, which is the
right trade-off for a library whose first priority is correctness.
"""

from __future__ import annotations


__all__ = [
    "P",
    "primitive_root_of_unity",
    "ntt",
    "intt",
    "convolve",
    "ConvolutionOverflow",
]


P: int = 998244353
"""The prime modulus. ``P - 1 = 2**23 * 7 * 17``, so NTT is supported for any
power-of-two length up to ``2**23``."""


class ConvolutionOverflow(ValueError):
    """Raised when a convolution result cannot be recovered losslessly.

    This is raised eagerly by :func:`convolve` when the caller asserts a bound
    on the inputs (via ``max_abs_value``) that would allow the true integer
    result to collide with another integer modulo ``P``. The intent is to fail
    loudly rather than return a silently-wrapped value.
    """


def _multiplicative_order(a: int, n: int) -> int:
    """Return the multiplicative order of ``a`` modulo ``n``.

    Computed by trial division of ``n - 1``; sufficient here because ``n - 1``
    is highly composite but never astronomically large for the primes we use.
    Raises ``ValueError`` if ``a`` is not a unit mod ``n``.
    """
    if n <= 1:
        raise ValueError("modulus must be > 1")
    a %= n
    if a == 0:
        raise ValueError("a is not a unit modulo n")
    if n == 2:
        return 1

    phi = n - 1
    order = phi
    # Factor phi and remove prime factors.
    m = phi
    f = 2
    while f * f <= m:
        if m % f == 0:
            while m % f == 0:
                m //= f
            while order % f == 0 and pow(a, order // f, n) == 1:
                order //= f
        f += 1
    if m > 1:
        if order % m == 0 and pow(a, order // m, n) == 1:
            order //= m
    return order


def primitive_root_of_unity(n: int) -> int:
    """Return a primitive ``n``-th root of unity modulo ``P``.

    ``n`` must be a power of two with ``1 <= n <= 2**23``. We start from the
    known primitive root ``g = 3`` modulo ``P`` and raise it to ``(P-1)//n``,
    which yields a root of exact order ``n``. The bound ``n <= 2**23`` exists
    because ``P - 1`` is divisible by ``2**23`` but by no higher power of two.
    """
    if n < 1 or n & (n - 1) != 0:
        raise ValueError("n must be a positive power of two")
    if n > (1 << 23):
        raise ValueError("n exceeds 2**23 for this prime")
    g = 3
    # pow with a positive exponent is always a non-negative Python int, so we
    # never need an extra reduction here.
    root = pow(g, (P - 1) // n, P)
    # Defensive: confirm the order is exactly n. This guards against a future
    # change to P or g that silently breaks the invariant.
    if _multiplicative_order(root, P) != n:
        raise ValueError("internal: failed to construct primitive root")
    return root


def _reverse_bits(x: int, bits: int) -> int:
    """Bit-reverse the lowest ``bits`` bits of ``x``."""
    r = 0
    for _ in range(bits):
        r = (r << 1) | (x & 1)
        x >>= 1
    return r


def _bit_reverse(a: list[int]) -> None:
    """Reorder ``a`` in place so index ``i`` maps to its bit-reversed position."""
    n = len(a)
    if n <= 1:
        return
    bits = n.bit_length() - 1
    for i in range(n):
        j = _reverse_bits(i, bits)
        if j > i:
            a[i], a[j] = a[j], a[i]


def _transform(a: list[int], inverse: bool) -> list[int]:
    """In-place iterative radix-2 NTT (or inverse NTT) of ``a``.

    The caller must guarantee ``len(a)`` is a power of two. We do not mutate
    the caller's list; this function is called only on private copies.
    """
    n = len(a)
    if n == 1:
        return a
    if n & (n - 1) != 0:
        raise ValueError("length must be a power of two")

    _bit_reverse(a)

    root = primitive_root_of_unity(n)
    if inverse:
        # For the inverse transform we need the inverse of the primitive root.
        # pow(_, -1, P) is valid since Python 3.8 and yields the modular
        # inverse; we use it here for clarity.
        root = pow(root, P - 2, P)

    # Layered butterfly: for each half-length m, combine pairs of elements
    # using the twiddle factor root^(n/m). Using pow for each twiddle is simple
    # but would be O(n log n) exponentiations; we instead advance the twiddle
    # multiplicatively within each layer, which is O(n) per layer.
    length = 2
    while length <= n:
        half = length // 2
        # Twiddle factor stepping by ``root`` advances it to the next
        # ``length``-th root, giving us O(1) work per butterfly.
        wlen = pow(root, n // length, P)
        for i in range(0, n, length):
            w = 1
            for j in range(half):
                u = a[i + j]
                v = a[i + j + half] * w % P
                a[i + j] = (u + v) % P
                a[i + j + half] = (u - v) % P
                w = w * wlen % P
        length <<= 1

    if inverse:
        n_inv = pow(n, P - 2, P)
        for i in range(n):
            a[i] = a[i] * n_inv % P

    return a


def ntt(a: list[int]) -> list[int]:
    """Return the forward NTT of ``a`` modulo ``P``.

    The input is zero-padded to the next power of two with length at least
    ``len(a)``. Elements are reduced modulo ``P`` so negative or oversized
    integers are handled predictably: a residue in ``[0, P)`` is what gets
    transformed.

    The returned list has length equal to that power of two.
    """
    if not a:
        return []
    n = 1
    while n < len(a):
        n <<= 1
    # Copy and reduce into a fresh list so the caller's input is untouched.
    buf = [x % P for x in a]
    buf += [0] * (n - len(a))
    return _transform(buf, inverse=False)


def intt(a: list[int]) -> list[int]:
    """Return the inverse NTT of ``a`` modulo ``P``.

    Unlike :func:`ntt`, this function does **not** zero-pad: the length of
    ``a`` must already be a power of two. Padding on the inverse side would
    change the semantics (the forward transform of a shorter sequence has a
    specific zero-padded representation, and a round trip should preserve the
    exact length the caller already chose).
    """
    if not a:
        return []
    n = len(a)
    if n & (n - 1) != 0:
        raise ValueError(
            "intt input length must be a power of two (got %d)" % n
        )
    buf = [x % P for x in a]
    return _transform(buf, inverse=True)


def convolve(a: list[int], b: list[int], max_abs_value: int | None = None) -> list[int]:
    """Convolve two integer sequences exactly, modulo ``P``.

    The result has length ``len(a) + len(b) - 1`` (or ``0`` if either input is
    empty). Each element is the true integer convolution value reduced into
    ``[0, P)``; because the result is bounded, this reduction is lossless as
    long as the true value fits in the modulus.

    If ``max_abs_value`` is provided, it is taken as a bound on the absolute
    value of every input element. With that bound we can check whether the
    largest possible output element fits below ``P / 2``, i.e. whether the
    reduction is guaranteed lossless. If it is not, :class:`ConvolutionOverflow`
    is raised rather than returning a wrapped (and therefore wrong-looking)
    integer.
    """
    if not a or not b:
        return []

    n = 1
    target = len(a) + len(b) - 1
    while n < target:
        n <<= 1

    fa = [x % P for x in a] + [0] * (n - len(a))
    fb = [x % P for x in b] + [0] * (n - len(b))
    _transform(fa, inverse=False)
    _transform(fb, inverse=False)
    fc = [fa[i] * fb[i] % P for i in range(n)]
    res = _transform(fc, inverse=True)

    # Trim padding to the true convolution length.
    res = res[:target]

    if max_abs_value is not None:
        if max_abs_value < 0:
            raise ValueError("max_abs_value must be non-negative")
        # Worst-case coefficient: every pair contributes max_abs_value**2,
        # summed across min(len(a), len(b)) overlapping products.
        peak = max_abs_value * max_abs_value * min(len(a), len(b))
        # If peak < P/2 the reduction is unique. We use a strict bound because
        # equality would let both +peak and -peak map to the same residue.
        if peak >= (P + 1) // 2:
            raise ConvolutionOverflow(
                "worst-case coefficient magnitude %d exceeds P/2=%d; "
                "result would collide modulo P" % (peak, P // 2)
            )
        # Map residues back to signed integers for a clean result.
        signed = []
        for c in res:
            if c > (P - 1) // 2:
                c -= P
            signed.append(c)
        return signed

    return res
