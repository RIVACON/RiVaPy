# 2025.07.02 Hans Nguyen
#
#

# -----------------------------------
# #IMPORT Modules
from typing import List as _List, Union as _Union, Callable
import datetime as dt
import abc
import pandas as pd
import numpy as np
from bisect import bisect_left

# from scipy.optimize import curve_fit as curve_fit
# from scipy.interpolate import interp1d
from bisect import bisect_left
from rivapy.tools.enums import DayCounterType, InterpolationType, ExtrapolationType
from rivapy.tools.datetools import DayCounter


# TODO list of interpolaters to implement
# -----------------------------------
# FUNCTION def

#    class InterpolationType(_MyEnum):
#        CONSTANT = "CONSTANT"
#        LINEAR = "LINEAR"
#        LINEAR_LOG = "LINEARLOG"
#        CONSTRAINED_SPLINE = "CONSTRAINED_SPLINE"
#        HAGAN = "HAGAN"
#        HAGAN_DF = "HAGAN_DF"


class Interpolator:

    def __init__(self, interpolation_type: _Union[str, InterpolationType], extrapolation_type: _Union[str, ExtrapolationType]):
        """Constructor for Interpolatr class object.

        Args:
            interpolation_type (_Union[str, InterpolationType]): The interpolation method to be used
            extrapolation_type (_Union[str, ExtrapolationType]): the extrapolation method to be used
        """
        self._interpolation_type = InterpolationType.to_string(interpolation_type)
        self._extrapolation_type = ExtrapolationType.to_string(
            extrapolation_type
        )  # TODO is this redundant as we feed extrapolation method into interp  as argument?
        self._interp = Interpolator.get(self._interpolation_type)

    def interp(
        self, x_list: list, y_list: list, target_x: _Union[float, _List[float]], extrapolation: _Union[str, ExtrapolationType]
    ) -> _Union[float, _List[float]]:
        """Wrapper method to execute desired interpolation method. If given a list of targets will return a list.

        Args:
            x_list (_List[float]): x-values
            y_list (_List[float]): y-values
            target_x (float,_List[float]): x-value for which a desired y-value is to be determined

        Returns:
            _Union[float, _List[float]]: return interpolation valuues
        """

        extrapolation_type = ExtrapolationType.to_string(extrapolation)

        if isinstance(target_x, list):
            return [self._interp(x_list, y_list, target_x_, extrapolation_type) for target_x_ in target_x]
        else:
            return self._interp(x_list, y_list, target_x, extrapolation_type)

    @staticmethod
    def get(interpolator: _Union[str, InterpolationType]) -> Callable[[list, list, _Union[float, _List[float]], str], float]:
        """Mapping function to determine which interpolator to use

        Args:
            interpolator (_Union[str, InterpolationType]): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            Callable[[list, list, _Union[float, _List[float]], str], float]: _description_
        """
        interp = InterpolationType.to_string(interpolator)
        # extrap = ExtrapolationType.to_string(extrapolator)
        # the assumption at the moment is that for a given interpolation type, the extrapolation type must be the same or CONSTANT
        # this is a design choice for the moment

        mapping = {
            InterpolationType.LINEAR.value: Interpolator.linear,
            InterpolationType.LINEAR_LOG.value: Interpolator.linear_log,
            InterpolationType.CONSTANT.value: Interpolator.constant,
            InterpolationType.HAGAN.value: Interpolator.hagan,
            InterpolationType.HAGAN_DF.value: Interpolator.hagan_df,
        }

        if interp in mapping:
            return mapping[interp]
        else:
            raise NotImplementedError(f"{interp} not yet implemented.")

    @staticmethod
    def linear(x_list: list, y_list: list, x: float, extrapolation: str) -> float:
        """Simple linear interpolation. TDDO : simply use scipy? No...match design structure

        Args:
            x_list (_type_): values that are assumed to be sorted
            y_list (_type_): corresponding y values
            x (_type_): target x-value
            extrapolate (str): extrapolation method chosen for when x is outside x_list.

        Returns:
            float: interpolated value
        """
        # print("interpolation values")
        # print(x_list)  # DEBUG TEST TODO REMOVE
        # print(y_list)
        if not x_list or not y_list or len(x_list) != len(y_list):
            raise ValueError("x_list and y_list must be non-empty and of the same length.")

        if x <= x_list[0] or x_list[-1] <= x:

            if extrapolation == "NONE":
                raise ValueError("Extrapolation chosen as NONE but target 'x' lies outside of range")
            elif extrapolation == "CONSTANT":
                if x <= x_list[0]:
                    return y_list[0]
                elif x_list[-1] <= x:
                    return y_list[-1]
            elif extrapolation == "LINEAR":
                if x <= x_list[0]:
                    x0, x1 = x_list[0], x_list[1]
                    y0, y1 = y_list[0], y_list[1]
                elif x_list[-1] <= x:
                    x0, x1 = x_list[-2], x_list[-1]
                    y0, y1 = y_list[-2], y_list[-1]

        else:
            i = bisect_left(x_list, x)  # insert target x next to 2 closest points
            x0, x1 = x_list[i - 1], x_list[i]
            y0, y1 = y_list[i - 1], y_list[i]

        # Linear interpolation formula
        slope = (y1 - y0) / (x1 - x0)
        y = y0 + slope * (x - x0)

        return y

    @staticmethod
    def constant(x_list: list, y_list: list, x: float, extrapolation: str) -> float:
        """PLACEHOLDER #TODO implement"""

        return -9999.999

    @staticmethod
    def linear_log(x_list: list, y_list: list, x: float, extrapolation: str) -> float:
        # x_val = np.array(x_list)
        y_val = np.array(y_list)

        if np.any(y_val <= 0):
            raise ValueError("All y-values must be positive for log-linear interpolation.")

        log_y_val = np.log(y_val).tolist()

        # handle the extrapolation properly TODO
        if extrapolation == "LINEAR_LOG":
            extr = "LINEAR"
        else:
            extr = extrapolation
        log_y_interp = Interpolator.linear(x_list, log_y_val, x, extr)
        y_interp = np.exp(log_y_interp)
        return y_interp



    # Helper: compute Hagan piecewise polynomials
    @staticmethod
    def _hagan_polynomials(x_list: _List[float], y_list: _List[float]):
        """
        Compute the piecewise quadratic coefficients for Hagan interpolation.

        Args:
            x_list: grid of cell boundaries (length n+1)
            y_list: cell averages (length n)
        
        Returns:
            x_vals: left boundaries of polynomial segments
            a0, a1, a2: quadratic coefficients for each segment
        """
        x_cells = np.array(x_list, dtype=float) # cell boundaries
        u = np.array(y_list, dtype=float) # the cell averages, one per cell
        n = len(u) # numberr of cells

        if len(x_cells) != n + 1:
            raise ValueError("x_list must have length len(y_list)+1") # i.e. for every 2 subsequent x points, there is one y point representing the cell average

        dx = np.diff(x_cells) # the width of the cells used to compute the slope of scale polynomials

        # Compute breakpoints (un) based on neighboring cell averages in a weighted way. 
        # breakppoints = approximated value at cell boundaries
        un = np.zeros(n + 1)
        for i in range(1, n):
            un[i] = (dx[i]*u[i-1] + dx[i-1]*u[i]) / (dx[i] + dx[i-1])
        un[0] = 2*u[0] - un[1] #first point extrapolated to close the system
        un[n] = 2*u[n-1] - un[n-1] #last popint extrapolated to close the system

        # Store polynomial segments
        x_vals = [x_cells[0]] # Left boundaries of each polynomial segment. Segments will go from x_vals[i] -> x_vals[i+1].
        a0, a1, a2 = [], [], [] # Coefficients of quadratic polynomials on segment i
        EPS = 1e-15 # epsilon, tiny value to avoid numerical issues (e.g., division by zero).

        for i in range(n): # iterate through each segment
            # define shape of polynomial
            #  = how far does the polynomial need to go from the cell average to match the boundary values
            g0 = un[i] - u[i] # difference between the left breakpoint of cell i and the cell average.
            g1 = un[i+1] - u[i] # difference between the right breakpoint of cell i and the cell average.
            dx_i = dx[i] # width of cell i

            # NOTE: from what i can tell case i matches with case iii if g1 = -0.5*g0
            # NOTE case i matches case ii if g1 = -2*g0
            # the order of the cases defines the priority on the mathcing border cases

            # CASE i: simple quadratic  incl. g0=g1=0
            if ((g0 >= 0 and -2*g0 <= g1 <= -0.5*g0) or
                (g0 <= 0 and -0.5*g0 <= g1 <= -2*g0)):

                a0.append(un[i])                                  # value at left boundary
                a1.append((-4*un[i] - 2*un[i+1] + 6*u[i])/dx_i)   # slope
                a2.append((3*un[i] + 3*un[i+1] - 6*u[i])/dx_i**2) # curvature
                x_vals.append(x_cells[i+1])
                continue

            # CASE ii: constant + quadratic
            if ((g0 <= 0 and -2*g0 <= g1) or 
                (g0 >= 0 and g1 <= -2*g0)):

                eta = dx_i * ((2*g0 + g1) / (g1 - g0)) # point inside the cell where slope changes
                a0.append(un[i]); a1.append(0.0); a2.append(0.0) # first segment is constant, a1=a2=0, 
                x_vals.append(x_cells[i] + eta)
                if abs(dx_i - eta) > EPS: # second segment is quadratic,provided greater than EPS
                    a0.append(un[i]); a1.append(0.0)
                    a2.append((un[i+1]-un[i]) / (dx_i - eta)**2)
                    x_vals.append(x_cells[i+1])
                continue

            # CASE iii: negative/positive slope adjustments
            if ((g0 >= 0 and -0.5*g0 <= g1 <= 0) or 
                (g0 <= 0 and 0 <= g1 <= -0.5*g0)):

                eta = dx_i * (3*g1)/(g1 - g0) # eta= location inside cell for quadratic segment
                if abs(eta) > EPS:
                    q = (un[i]-un[i+1])/eta**2   #q = curvature needed to go from left boundary to the slope at eta
                    a0.append(un[i]); a1.append(-2*q*eta); a2.append(q)
                    x_vals.append(x_cells[i] + eta)
                a0.append(un[i+1]); a1.append(0.0); a2.append(0.0)
                x_vals.append(x_cells[i+1])
                continue

            # CASE iv: monotone convex
            if (g0 > 0 and g1 > 0) or (g0 < 0 and g1 < 0):
                
                r = g1 / (g1 + g0)
                eta = dx_i * r
                A = -g0 * r
                if abs(eta) > EPS:
                    q = (g0 - A)/eta**2
                    a0.append(g0+u[i]); a1.append(-2*(g0-A)/eta); a2.append(q)
                    x_vals.append(x_cells[i] + eta)
                if abs(dx_i - eta) > EPS:
                    a0.append(A+u[i]); a1.append(0.0)
                    a2.append((g1-A)/(dx_i-eta)**2)
                    x_vals.append(x_cells[i+1])
                continue

        return np.array(x_vals), np.array(a0), np.array(a1), np.array(a2)


    # Hagan interpolation for forward rate
    @staticmethod
    def hagan(x_list: _List[float], y_list: _List[float], x: float, extrapolation: str) -> float:
        """
        Interpolate the forward rate f(x) using Hagan's method.
        Returns the **exact piecewise quadratic value** at x.

        Args:
            x_list (List[float]): grid of cell boundaries (length n+1)
            y_list (List[float]): cell averages (length n)
            x (float): target point to interpolate
            extrapolation (str): determine extrapolation method to use if x is outside x_list

        Returns:

            float: interpolated forward rate at x
        """
        x_vals, a0, a1, a2 = Interpolator._hagan_polynomials(x_list, y_list)

        # Handle extrapolation
        if x <= x_vals[0]:
            if extrapolation.upper() in ("CONSTANT", "CONSTANT_DF"):
                return a0[0]
            else:
                raise ValueError("x below grid and extrapolation not allowed")
        if x >= x_vals[-1]:
            if extrapolation.upper() in ("CONSTANT", "CONSTANT_DF"):
                # last segment quadratic evaluation
                s = x_vals[-1] - x_vals[-2]
                return a0[-1] + (a1[-1] + a2[-1]*s)*s
            else:
                raise ValueError("x above grid and extrapolation not allowed")

        # find correct segment
        idx = np.searchsorted(x_vals, x) - 1
        s = x - x_vals[idx] # distance from left boundary of the segment
        return a0[idx] + (a1[idx] + a2[idx]*s)*s


    # Hagan integration (cumulative integral of forward rate)
    @staticmethod
    def hagan_integrate(x_list: _List[float], y_list: _List[float], x: float) -> float:
        """
        Integrate the Hagan forward rate piecewise polynomial from 0 to x.
        Returns the **exact integral**, used for computing DF(x) = exp(-integral)
        """
        x_vals, a0, a1, a2 = Interpolator._hagan_polynomials(x_list, y_list)
        integral = 0.0

        # Handle extrapolation implicitly: integrate only inside segments
        for i in range(len(a0)): # integrate oveer all segments up to x
            x_left = x_vals[i]
            x_right = x_vals[i+1] if i+1 < len(x_vals) else x_vals[-1]
            dx_seg = min(max(x - x_left, 0.0), x_right - x_left) # how much of this segment to integrate over. so if it is pst target x, dont integerate 
            if dx_seg > 0:
                integral += (a0[i]*dx_seg + 0.5*a1[i]*dx_seg**2 + (1./3.)*a2[i]*dx_seg**3)
            if x <= x_right:
                break

        return integral


    # Hagan discount factor DF(x)
    @staticmethod
    def hagan_df(x_list: _List[float], y_list: _List[float], x: float, extrapolation: str) -> float:
        """
        Compute the discount factor DF(x) = exp(-INTEGRAL( f(s) ds) )
        using Hagan's piecewise quadratic interpolation of the forward rate.

        Args:
            x_list: time grid (e.g. [0.5, 1.0, 2.0, ...])
            y_list: known discount factors DF(t_i) at those grid points
            x: target time where we want DF(x)
            extrapolation: defines behavior outside grid ("CONSTANT_DF" or error)

        Returns:
            float: interpolated discount factor at time x
        """

        # Convert input lists to numpy arrays for numerical ops
        x_cells = np.array(x_list, dtype=float)
        df = np.array(y_list, dtype=float)

        if len(df) < 2:
            raise ValueError("At least 2 discount factors required")

        # STEP 1: Compute "forward rates" between grid points
        # f_i = (log DF_i - log DF_{i+1}) / (x_{i+1} - x_i)
        #   This gives the *instantaneous forward rate* assumed constant
        #   between each pair of discount factors.
        #
        #   Recall: DF(x) = exp(-integral f(s) ds)
        #   So  ln DF_i - ln DF_{i+1} = integral_{x_i}^{x_{i+1}} f(s) ds
        #
        #   Approximating f(s) as constant over each small interval
        #   gives this finite-difference form.
        fwd = (np.log(df[:-1]) - np.log(df[1:])) / (x_cells[1:] - x_cells[:-1])


        # STEP 2: Handle extrapolation (x outside the grid)
        # Case A: x < first grid point
        if x < x_cells[0]:
            if extrapolation.upper() == "CONSTANT_DF":
                # Integrate forward rates up to the first point
                # This gives the cumulative area integral f(s) ds = y
                y = Interpolator.hagan_integrate(x_cells, fwd, x_cells[0])

                # Compute *average rate* up to the first knot:
                # r = (1/x_o) * integral(f(s) ds)
                r = y / x_cells[0]

                # Now assume that beyond this point, the discount factor
                # decays at that "constant average rate":
                # DF(x) = exp(-x * r)
                return np.exp(-x * r)
            else:
                raise ValueError("x below grid and extrapolation not allowed")

        # Case B: x > last grid point
        elif x > x_cells[-1]:
            if extrapolation.upper() == "CONSTANT_DF":
                # Integrate up to the last available grid point
                y = Interpolator.hagan_integrate(x_cells, fwd, x_cells[-1])

                # Compute the average continuously-compounded rate up to that last point
                r = y / x_cells[-1]

                # Apply constant extrapolation beyond that:
                return np.exp(-x * r)
            else:
                raise ValueError("x above grid and extrapolation not allowed")


        # STEP 3: Inside the grid — exact Hagan interpolation
        # Compute the "exact integral" of f(s) ds using the Hagan
        # piecewise quadratic polynomial representation of f(s)
        integral = Interpolator.hagan_integrate(x_cells, fwd, x)

        # Finally compute DF(x) = exp(-∫₀ˣ f(s) ds)
        return np.exp(-integral)


    # Hagan discount factor derivative DF'(x)
    @staticmethod
    def hagan_df_derivative(x_list: _List[float], df_list: _List[float], x: float, extrapolation: str) -> float:
        """
        Compute derivative of discount factor: DF'(x) = -f(x) * DF(x)
        Exact calculation, matching C++ InterpolationHagan1D_DF::computeDerivative
        """
        x_cells = np.array(x_list, dtype=float)
        df = np.array(df_list, dtype=float)
        if len(df) < 2:
            raise ValueError("At least 2 discount factors required")

        # Forward rates
        fwd = (np.log(df[:-1]) - np.log(df[1:])) / (x_cells[1:] - x_cells[:-1])

        # Extrapolation handling
        if x < x_cells[0]:
            if extrapolation.upper() == "CONSTANT_DF":
                y = Interpolator.hagan_integrate(x_cells, fwd, x_cells[0])
                r = y / x_cells[0]
                return -r * np.exp(-x * r)
            else:
                raise ValueError("x below grid and extrapolation not allowed")
        elif x > x_cells[-1]:
            if extrapolation.upper() == "CONSTANT_DF":
                y = Interpolator.hagan_integrate(x_cells, fwd, x_cells[-1])
                r = y / x_cells[-1]
                return -r * np.exp(-x * r)
            else:
                raise ValueError("x above grid and extrapolation not allowed")

        # Inside grid: exact DF and f(x)
        DF_val = Interpolator.hagan_df(x_list, df_list, x, extrapolation)
        r_val = Interpolator.hagan(x_list, fwd, x, extrapolation)
        return -r_val * DF_val

# if __name__ == "__main__":


# -----------------------------------
# Unit tests
# can be found in rivapy/tests/ folder
# please use the python unittest framework.
