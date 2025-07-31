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

# from scipy.optimize import curve_fit as curve_fit
# from scipy.interpolate import interp1d
from bisect import bisect_left
from rivapy.tools.enums import DayCounterType, InterpolationType, ExtrapolationType
from rivapy.tools.datetools import DayCounter

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
        """_summary_

        Args:
            interpolation_type (_Union[str, InterpolationType]): _description_
            extrapolation_type (_Union[str, ExtrapolationType]): _description_
        """
        self._interpolation_type = InterpolationType.to_string(interpolation_type)
        self._extrapolation_type = ExtrapolationType.to_string(
            extrapolation_type
        )  # TODO is this redundant as we feed extrapolation method into interp  as argument
        self._interp = Interpolator.get(self._interpolation_type)

    def interp(
        self, x_list: list, y_list: list, target_x: _Union[float, _List[float]], extrapolation: _Union[str, ExtrapolationType]
    ) -> _Union[float, _List[float]]:
        """_summary_

        Args:
            x_list (list): _description_
            y_list (list): _description_
            target_x (float): _description_

        Returns:
            _Union[float, _List[float]]: _description_
        """

        extrapolation_type = ExtrapolationType.to_string(extrapolation)

        if isinstance(target_x, list):
            return [self._interp(x_list, y_list, target_x_, extrapolation_type) for target_x_ in target_x]
        else:
            return self._interp(x_list, y_list, target_x, extrapolation_type)

    @staticmethod
    def get(interpolator: _Union[str, InterpolationType]) -> Callable[[list, list, _Union[float, _List[float]], str], float]:

        interp = InterpolationType.to_string(interpolator)
        # extrap = ExtrapolationType.to_string(extrapolator)
        # the assumption at the moment is that for a given interpolation type, the extrapolation type must be the same or CONSTANT
        # this is a design choice for the moment

        mapping = {
            InterpolationType.LINEAR.value: Interpolator.linear,
            InterpolationType.LINEAR_LOG.value: Interpolator.linear_log,
            InterpolationType.CONSTANT.value: Interpolator.constant,
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


# if __name__ == "__main__":


# -----------------------------------
# Unit tests
# can be found in rivapy/tests/ folder
# please use the python unittest framework.
