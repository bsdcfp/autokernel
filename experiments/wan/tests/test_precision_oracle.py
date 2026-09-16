import math
import unittest

from precision_followup import nearest_bf16


class PrecisionOracleTests(unittest.TestCase):
    def test_midpoint_neighbors_do_not_round_through_float32(self):
        lower, upper = -0.0947265625, -0.09423828125
        midpoint = (lower + upper) / 2
        self.assertEqual(nearest_bf16(math.nextafter(midpoint, -math.inf)), lower)
        self.assertEqual(nearest_bf16(math.nextafter(midpoint, math.inf)), upper)
        self.assertEqual(nearest_bf16(midpoint), lower)  # even BF16 significand
        self.assertEqual(nearest_bf16(-0.09448242182143599), upper)

    def test_zero_subnormal_and_exact_values(self):
        self.assertEqual(nearest_bf16(0), 0)
        self.assertEqual(nearest_bf16(2.0**-133), 2.0**-133)
        self.assertEqual(nearest_bf16(2.0**-134), 0)
        self.assertEqual(nearest_bf16(1.5), 1.5)
        with self.assertRaises(ValueError):
            nearest_bf16(float('nan'))
