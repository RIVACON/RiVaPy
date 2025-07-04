import unittest
from rivapy.tools.interpolate import Interpolator
from rivapy.tools.enums import InterpolationType, ExtrapolationType

#, delta=1e-5 ?
class TestInterpolator(unittest.TestCase):

    def setUp(self):
        """Test data, simple linear case. Extend to more robust if requested.
        """
        self.x = [0, 1, 2, 3]
        self.y = [0, 10, 20, 30]

    def test_linear_interpolation_scalar(self):
        """Test single target x argument
        """
        interpolator = Interpolator(InterpolationType.LINEAR, ExtrapolationType.LINEAR)
        result = interpolator.interp(self.x, self.y, 1.5, "LINEAR")
        self.assertAlmostEqual(result, 15.0)

    def test_linear_interpolation_list(self):
        """Test multi target x arguments
        """
        interpolator = Interpolator(InterpolationType.LINEAR, ExtrapolationType.LINEAR)
        result = interpolator.interp(self.x, self.y, [0.5, 1.5, 2.5], "LINEAR")
        expected = [5.0, 15.0, 25.0]
        self.assertEqual(len(result), len(expected))
        for r, e in zip(result, expected):
            self.assertAlmostEqual(r, e)

    def test_linear_extrapolation_left_right(self):
        """Test both edges of extrapolation.
        """
        interpolator = Interpolator(InterpolationType.LINEAR, ExtrapolationType.LINEAR)
        result = interpolator.interp(self.x, self.y, -1, "LINEAR")
        expected = -10.0  # slope = 10/unit, so -1 → -10 from given self.x and self.y
        self.assertAlmostEqual(result, expected)

        result = interpolator.interp(self.x, self.y, 4, "LINEAR")
        expected = 40
        self.assertAlmostEqual(result, expected)

    def test_constant_extrapolation(self):
        """Test the case of CONSTANT extrapolation mode selected.
        """
        interpolator = Interpolator(InterpolationType.LINEAR, ExtrapolationType.CONSTANT)
        result_left = interpolator.interp(self.x, self.y, -10, "CONSTANT")
        result_right = interpolator.interp(self.x, self.y, 10, "CONSTANT")
        self.assertEqual(result_left, self.y[0])
        self.assertEqual(result_right, self.y[-1])

    def test_no_extrapolation_raises(self):
        """Test for the case that extrapolation was set to NONE but target values
        are outside data range
        """
        interpolator = Interpolator(InterpolationType.LINEAR, ExtrapolationType.NONE)
        with self.assertRaises(ValueError):
            interpolator.interp(self.x, self.y, -1, "NONE")
        with self.assertRaises(ValueError):
            interpolator.interp(self.x, self.y, 4, "NONE")


    def test_mismatched_length_raises(self):
        """Test the case of incorrect data input length mismatch.
        """
        interpolator = Interpolator(InterpolationType.LINEAR, ExtrapolationType.LINEAR)
        with self.assertRaises(ValueError):
            interpolator.interp([0, 1, 2], [10, 20], 1.5, "LINEAR")

if __name__ == '__main__':
    unittest.main()