import unittest

from number_theoretic_transform import (
    P,
    primitive_root_of_unity,
    ntt,
    intt,
    convolve,
    ConvolutionOverflow,
)


class TestPrimitiveRoot(unittest.TestCase):
    def test_order_is_exactly_n(self):
        for e in range(0, 24):
            n = 1 << e
            w = primitive_root_of_unity(n)
            self.assertEqual(pow(w, n, P), 1)
            # No smaller positive power should be 1.
            if n > 1:
                self.assertNotEqual(pow(w, n // 2, P), 1)

    def test_rejects_non_power_of_two(self):
        with self.assertRaises(ValueError):
            primitive_root_of_unity(3)
        with self.assertRaises(ValueError):
            primitive_root_of_unity(0)

    def test_rejects_too_large(self):
        with self.assertRaises(ValueError):
            primitive_root_of_unity(1 << 24)


class TestNTTRoundTrip(unittest.TestCase):
    def test_round_trip_various_lengths(self):
        cases = [
            [1],
            [1, 2],
            [1, 2, 3, 4],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [5, 7, 11, 13, 17, 19, 23, 29],
        ]
        for a in cases:
            with self.subTest(a=a):
                self.assertEqual(intt(ntt(a)), a)

    def test_ntt_length_is_power_of_two(self):
        out = ntt([1, 2, 3])
        self.assertEqual(len(out), 4)
        out = ntt([1, 2, 3, 4, 5])
        self.assertEqual(len(out), 8)

    def test_intt_requires_power_of_two_length(self):
        with self.assertRaises(ValueError):
            intt([1, 2, 3])

    def test_empty(self):
        self.assertEqual(ntt([]), [])
        self.assertEqual(intt([]), [])

    def test_input_not_mutated(self):
        a = [1, 2, 3, 4]
        _ = ntt(a)
        self.assertEqual(a, [1, 2, 3, 4])

    def test_reduces_inputs_modulo_p(self):
        # P itself should reduce to 0.
        out = ntt([P, P + 1, -1, 2])
        self.assertEqual(intt(out), [0, 1, -1 % P, 2])


class TestConvolution(unittest.TestCase):
    def test_basic_convolution(self):
        a = [1, 2, 3]
        b = [4, 5, 6]
        # Expected: [4, 13, 28, 27, 18]
        out = convolve(a, b, max_abs_value=10)
        self.assertEqual(out, [4, 13, 28, 27, 18])

    def test_convolution_matches_definition(self):
        a = [2, -1, 0, 3]
        b = [1, 4, -2]
        expected = [
            a[0] * b[0],
            a[0] * b[1] + a[1] * b[0],
            a[0] * b[2] + a[1] * b[1] + a[2] * b[0],
            a[1] * b[2] + a[2] * b[1] + a[3] * b[0],
            a[2] * b[2] + a[3] * b[1],
            a[3] * b[2],
        ]
        out = convolve(a, b, max_abs_value=10)
        self.assertEqual(out, expected)

    def test_empty(self):
        self.assertEqual(convolve([], [1, 2]), [])
        self.assertEqual(convolve([1, 2], []), [])
        self.assertEqual(convolve([], []), [])

    def test_single_element(self):
        self.assertEqual(convolve([5], [7], max_abs_value=10), [35])

    def test_length_is_sum_minus_one(self):
        out = convolve([1, 2, 3], [4, 5])
        self.assertEqual(len(out), 4)

    def test_overflow_detected(self):
        # max_abs_value chosen so the worst-case product exceeds P/2.
        with self.assertRaises(ConvolutionOverflow):
            convolve([1] * 1000, [1] * 1000, max_abs_value=P)

    def test_overflow_negative_bound_rejected(self):
        with self.assertRaises(ValueError):
            convolve([1, 2], [3, 4], max_abs_value=-1)

    def test_no_bound_returns_residues(self):
        # Without a bound, results are raw residues in [0, P).
        out = convolve([1, 2], [3, 4])
        self.assertEqual(out, [3, 10, 8])
        for c in out:
            self.assertTrue(0 <= c < P)

    def test_large_but_safe(self):
        # A larger convolution that still fits under P/2.
        a = [1000] * 100
        b = [1000] * 100
        out = convolve(a, b, max_abs_value=1000)
        expected = [1000 * 1000 * (i + 1) for i in range(100)] + [
            1000 * 1000 * (99 - i) for i in range(99)
        ]
        self.assertEqual(out, expected)


if __name__ == "__main__":
    unittest.main()
