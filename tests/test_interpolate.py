import unittest
import numpy as np
from rivapy.tools.interpolate import Interpolator
from rivapy.tools.enums import InterpolationType, ExtrapolationType


# , delta=1e-5 ?
class TestInterpolator(unittest.TestCase):

    def setUp(self):
        """Test data, simple linear case. Extend to more robust if requested."""
        self.x = [0, 1, 2, 3]
        self.y = [0, 10, 20, 30]

    def test_linear_interpolation_scalar(self):
        """Test single target x argument"""
        interpolator = Interpolator(InterpolationType.LINEAR, ExtrapolationType.LINEAR)
        result = interpolator.interp(self.x, self.y, 1.5, "LINEAR")
        self.assertAlmostEqual(result, 15.0)

    def test_linear_interpolation_list(self):
        """Test multi target x arguments"""
        interpolator = Interpolator(InterpolationType.LINEAR, ExtrapolationType.LINEAR)
        result = interpolator.interp(self.x, self.y, [0.5, 1.5, 2.5], "LINEAR")
        expected = [5.0, 15.0, 25.0]
        self.assertEqual(len(result), len(expected))
        for r, e in zip(result, expected):
            self.assertAlmostEqual(r, e)

    def test_linear_extrapolation_left_right(self):
        """Test both edges of extrapolation."""
        interpolator = Interpolator(InterpolationType.LINEAR, ExtrapolationType.LINEAR)
        result = interpolator.interp(self.x, self.y, -1, "LINEAR")
        expected = -10.0  # slope = 10/unit, so -1 → -10 from given self.x and self.y
        self.assertAlmostEqual(result, expected)

        result = interpolator.interp(self.x, self.y, 4, "LINEAR")
        expected = 40
        self.assertAlmostEqual(result, expected)

    def test_constant_extrapolation(self):
        """Test the case of CONSTANT extrapolation mode selected."""
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

    def test_no_extrapolationType_raises(self):
        """Test for the case that extrapolation was set to NONE and the extrapolation
        argument given is ExtrapolationType and not a str
        """
        interpolator = Interpolator(InterpolationType.LINEAR, ExtrapolationType.NONE)
        with self.assertRaises(ValueError):
            interpolator.interp(self.x, self.y, -1, ExtrapolationType.NONE)
        with self.assertRaises(ValueError):
            interpolator.interp(self.x, self.y, 4, ExtrapolationType.NONE)

    def test_mismatched_length_raises(self):
        """Test the case of incorrect data input length mismatch."""
        interpolator = Interpolator(InterpolationType.LINEAR, ExtrapolationType.LINEAR)
        with self.assertRaises(ValueError):
            interpolator.interp([0, 1, 2], [10, 20], 1.5, "LINEAR")


class TestLinearLogInterpolator(unittest.TestCase):

    def setUp(self):
        """Set up sample positive data suitable for log-linear interpolation."""
        self.x = [0.0, 1.0, 2.0, 3.0]
        # discount factors that decline exponentially — log-linear interpolation should reproduce this exactly
        self.df = [1.0, np.exp(-0.02), np.exp(-0.04), np.exp(-0.06)]  # constant rate 2% per year

    def test_log_linear_exact_exponential(self):
        """
        For exponentially decaying data, log-linear interpolation should be exact.
        DF(x) = exp(-0.02 * x)
        """
        interpolator = Interpolator(InterpolationType.LINEAR_LOG, ExtrapolationType.LINEAR)
        for x in [0.5, 1.5, 2.5]:
            df_interp = interpolator.interp(self.x, self.df, x, "LINEAR_LOG")
            expected = np.exp(-0.02 * x)
            self.assertAlmostEqual(df_interp, expected, delta=1e-12)

    def test_log_linear_vector_input(self):
        """Test list input of target x values produces correct list output."""
        interpolator = Interpolator(InterpolationType.LINEAR_LOG, ExtrapolationType.LINEAR)
        x_targets = [0.5, 1.0, 2.5]
        result = interpolator.interp(self.x, self.df, x_targets, "LINEAR_LOG")
        expected = [np.exp(-0.02 * x) for x in x_targets]
        self.assertEqual(len(result), len(expected))
        for r, e in zip(result, expected):
            self.assertAlmostEqual(r, e, delta=1e-12)

    def test_log_linear_extrapolation_linear_mode(self):
        """Check extrapolation using LINEAR mode reproduces reasonable continuation."""
        interpolator = Interpolator(InterpolationType.LINEAR_LOG, ExtrapolationType.LINEAR)
        x_extrap = 4.0
        df_interp = interpolator.interp(self.x, self.df, x_extrap, "LINEAR_LOG")
        expected = np.exp(-0.02 * 4.0)
        self.assertAlmostEqual(df_interp, expected, delta=1e-12)

    def test_log_linear_requires_positive_y(self):
        """Log-linear interpolation must raise an error if any y <= 0."""
        interpolator = Interpolator(InterpolationType.LINEAR_LOG, ExtrapolationType.LINEAR)
        bad_y = [1.0, 0.5, 0.0, -0.5]
        with self.assertRaises(ValueError):
            interpolator.interp(self.x, bad_y, 1.0, "LINEAR_LOG")

    def test_log_linear_constant_extrapolation(self):
        """Verify constant extrapolation on left/right boundaries."""
        interpolator = Interpolator(InterpolationType.LINEAR_LOG, ExtrapolationType.CONSTANT)
        result_left = interpolator.interp(self.x, self.df, -1.0, "CONSTANT")
        result_right = interpolator.interp(self.x, self.df, 4.0, "CONSTANT")
        self.assertAlmostEqual(result_left, self.df[0], delta=1e-12)
        self.assertAlmostEqual(result_right, self.df[-1], delta=1e-12)

class TestHaganInterpolator(unittest.TestCase):

    def setUp(self):
        # Simple synthetic data — exponential discount curve
        # DF(x) = exp(-r*x), with r = 0.05 constant forward rate
        self.x = [0.0, 1.0, 2.0, 3.0, 5.0]
        self.r = 0.05
        self.df = [np.exp(-self.r * t) for t in self.x]

        # Forward rates between grid points should all be ~0.05
        self.fwd = (np.log(self.df[:-1]) - np.log(self.df[1:])) / (np.diff(self.x))


    def test_hagan_polynomials_shapes(self):
        """Ensure polynomial coefficient arrays have consistent lengths.
        """
        x_vals, a0, a1, a2 = Interpolator._hagan_polynomials(self.x, self.fwd)
        self.assertEqual(len(a0), len(a1))
        self.assertEqual(len(a1), len(a2))
        self.assertTrue(len(x_vals) >= 2)


    def test_hagan_constant_forward_rate(self):
        """Flat forward rate curve -> all interpolated points must equal 0.05.
        """
        for x in np.linspace(0.1, 4.9, 9):
            f_interp = Interpolator.hagan(self.x, self.fwd, x, "CONSTANT")
            self.assertAlmostEqual(f_interp, self.r, delta=1e-10)


    def test_hagan_integrate_matches_analytical(self):
        """
        For constant forward rate = 0.05, integral from 0→x should be 0.05 * x.
        """
        for x in [0.5, 1.0, 2.5, 4.0]:
            integral = Interpolator.hagan_integrate(self.x, self.fwd, x)
            expected = self.r * x
            self.assertAlmostEqual(integral, expected, delta=1e-10)


    def test_hagan_df_inside_grid(self):
        """Inside grid, DF(x) = exp(-0.05 * x).
        """
        for x in [0.5, 1.0, 2.0, 3.5]:
            df_interp = Interpolator.hagan_df(self.x, self.df, x, "CONSTANT_DF")
            expected = np.exp(-self.r * x)
            self.assertAlmostEqual(df_interp, expected, delta=1e-10)


    def test_hagan_df_extrapolation_constant_df(self):
        """Extrapolated DF should follow exp(-r_avg * x) rule.
        """
        # Compute "average rate" up to last point
        y = Interpolator.hagan_integrate(self.x, self.fwd, self.x[-1])
        r_avg = y / self.x[-1]

        for x in [6.0, 7.5, 10.0]:
            df_extrap = Interpolator.hagan_df(self.x, self.df, x, "CONSTANT_DF")
            expected = np.exp(-r_avg * x)
            self.assertAlmostEqual(df_extrap, expected, delta=1e-10)

    def test_hagan_df_extrapolation_below_first(self):
        """Check extrapolation below grid for CONSTANT_DF."""
        y = Interpolator.hagan_integrate(self.x, self.fwd, self.x[1])
        r_avg = y / self.x[1]

        for x in [-0.5, -1.0]:
            df_extrap = Interpolator.hagan_df(self.x, self.df, x, "CONSTANT_DF")
            expected = np.exp(-r_avg * x)
            self.assertAlmostEqual(df_extrap, expected, delta=1e-10)


    def test_hagan_df_derivative_relation(self):
        """
        DF'(x) = -f(x) * DF(x). Test numerically for constant forward rate.
        """
        for x in [0.5, 1.0, 2.5, 4.0]:
            df_interp = Interpolator.hagan_df(self.x, self.df, x, "CONSTANT_DF")
            f_interp = Interpolator.hagan(self.x, self.fwd, x, "CONSTANT_DF")
            df_prime = Interpolator.hagan_df_derivative(self.x, self.df, x, "CONSTANT_DF")

            expected = -f_interp * df_interp
            self.assertAlmostEqual(df_prime, expected, delta=1e-10)

    # --------------------------
    # Error cases
    def test_hagan_df_raises_for_short_input(self):
        """Require at least 2 discount factors."""
        with self.assertRaises(ValueError):
            Interpolator.hagan_df([1.0], [0.99], 0.5, "CONSTANT_DF")

    def test_hagan_extrapolation_raises_for_none(self):
        """Extrapolation NONE should raise."""
        with self.assertRaises(ValueError):
            Interpolator.hagan_df(self.x, self.df, 10.0, "NONE")

    def test_hagan_forward_out_of_bounds_raises(self):
        """Extrapolation type not allowed for forward interpolation."""
        with self.assertRaises(ValueError):
            Interpolator.hagan(self.x, self.fwd, -1.0, "NONE")



if __name__ == "__main__":
    unittest.main()
