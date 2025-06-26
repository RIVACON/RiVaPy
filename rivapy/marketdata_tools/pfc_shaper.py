import abc
import holidays
import numpy as np
import pandas as pd
import datetime as dt
import rivapy.tools.interfaces as interfaces
import rivapy.tools._validators as validator
from rivapy.tools.scheduler import SimpleSchedule
from typing import List, Dict, Literal, Optional


class PFCShaper(interfaces.FactoryObject):
    """PFCShaper interface. Each shaping model for energy price forward curves must inherit from this base class.

    Args:
        spot_prices (pd.DataFrame): Data used to calibrate the shaping model.
        holiday_calendar (holidays.HolidayBase): Calendar object to obtain country specific holidays.
        normalization_config (Optional[Dict[Literal["D", "W", "ME"], Optional[int]]], optional): A dictionary configurating the shape normalization periods.
            Here ``D`` defines the number of days at the beginning of the shape over which the individual mean is normalized to one.
            ``W`` defines the number of weeks at the beginning of the shape over which the individual mean is normalized to one.
            ``ME`` defines the number of months at the beginning of the shape over which the individual mean is normalized to one. The remaining shape is then normalized over the individual years.Defaults to None.
    """

    def __init__(
        self,
        spot_prices: pd.DataFrame,
        holiday_calendar: holidays.HolidayBase,
        normalization_config: Optional[Dict[Literal["D", "W", "ME"], Optional[int]]] = None,
    ):
        super().__init__()
        validator._check_pandas_index_for_datetime(spot_prices)
        self.spot_prices = spot_prices
        self.holiday_calendar = holiday_calendar
        self.normalization_config = normalization_config

        # normalization order containing also the resampling string pattern for pandas resample method
        self.__normalization_order = [("D", "%Y-%m-%d"), ("W", "%G-%V"), ("ME", "%Y-%m")]

    @abc.abstractmethod
    def calibrate(self) -> np.ndarray:
        """Calibration of the shaping model

        Returns:
            np.ndarray: Numpy array containing the fit.
        """
        pass

    @abc.abstractmethod
    def apply(self, apply_schedule: SimpleSchedule):
        """Applies the model on a schedule in order to generate a shape for future dates.

        Args:
            apply_schedule (SimpleSchedule): Schedule object in order to generate a shape for future dates.
        """
        pass

    @abc.abstractmethod
    def _set_regression_parameters(self, params: np.ndarray):
        self._regression_parameters = params

    def normalize_shape(self, shape: pd.DataFrame) -> pd.DataFrame:
        """Normalizes the shape based on ``normalization_config``.\n
        ``D`` defines the number of days at the beginning of the shape over which the individual mean is normalized to one.\n
        ``W`` defines the number of weeks at the beginning of the shape over which the individual mean is normalized to one.\n
        ``ME`` defines the number of months at the beginning of the shape over which the individual mean is normalized to one.
        The remaining shape is then normalized over the individual years.\n

        Example:
        ``D`` is 2, ``W`` is 2 and ``ME`` is 1. The shape starts at 03.03.2025 (monday).
        Since ``D`` is 2, the shape is normalized for 03.03.2025 and 04.03.2025 individually.\n
        The weeks are normalized from 05.03.2025 to 09.03.2025 and from 10.03.2025 to 16.03.2025.\n
        The month is then normalized from 17.03.2025 to 31.03.2025.
        The remaining shape (starting from 01.04.2025) is normalized on a yearly level.

        Args:
            shape (pd.DataFrame): Shape which should be normalized

        Returns:
            pd.DataFrame: Normalized shape
        """

        datetime_list: List[dt.datetime] = list(shape.index.copy())

        # yearly normalization
        def _normalize_year(shape: pd.DataFrame, datetime_list: List[dt.datetime]) -> pd.DataFrame:
            base_y = shape.resample("YE").mean()
            _shape = shape.rename(index=lambda x: x.strftime("%Y")).divide(base_y.rename(index=lambda x: x.strftime("%Y")), axis="index")

            shape_df = _shape.reset_index(drop=True)
            shape_df.index = datetime_list
            return shape_df

        if self.normalization_config is None:
            shape_df = _normalize_year(shape=shape, datetime_list=datetime_list)
            return shape_df
        else:
            # the normalization through the normalization_config is done in different parts
            normalized_datetimes = []
            normalized_shapes = []

            # iterate over the correct normalization order
            for resample_freq, resample_format in self.__normalization_order:
                if self.normalization_config.get(resample_freq, None) is None:
                    continue
                else:
                    # if the whole shape is already normalized by the previous normalization processes, the loop is stopped
                    if len(normalized_datetimes) == len(shape):
                        return pd.concat(normalized_shapes, axis=0).sort_index(ascending=True)

                    # get the part of the shape which was not part of any previous normalizations
                    temp_shape = shape.loc[~shape.index.isin(normalized_datetimes), :]

                    # normalize shape by the cofigured amount of days, weeks or months
                    resampled_shape = temp_shape.resample(resample_freq).mean()
                    resampled_shape = resampled_shape.iloc[: self.normalization_config[resample_freq], :]

                    partially_normalized_shape = temp_shape.rename(index=lambda x: x.strftime(resample_format)).divide(
                        resampled_shape.rename(index=lambda x: x.strftime(resample_format)), axis="index"
                    )

                    # Due to the operations done in the previous lines, the partially_normalized_shape does not contain the exact datetime but rather
                    # a datetime corresponding to the resampled frequency. Hence, the correct datetimes are added to the DataFrame and set as an index.
                    # This allows to concatenate the partially normalized shapes more easily at a later stage
                    partially_normalized_shape["datetimes"] = list(temp_shape.index)
                    partially_normalized_shape = partially_normalized_shape.reset_index(drop=True).set_index("datetimes").dropna()
                    normalized_datetimes += list(partially_normalized_shape.index)
                    normalized_shapes.append(partially_normalized_shape)

            if len(normalized_datetimes) == len(shape):
                return pd.concat(normalized_shapes, axis=0).sort_index(ascending=True)

            # the remaining shape is normalized on a yearly basis
            leftover_shape = shape.loc[~shape.index.isin(normalized_datetimes), :]
            leftover_datetime = list(leftover_shape.index)
            yearly_normalized_shape = _normalize_year(shape=leftover_shape, datetime_list=leftover_datetime)

            return pd.concat(normalized_shapes + [yearly_normalized_shape], axis=0).sort_index(ascending=True)

    def _to_dict(self):
        return {"spot_prices": self.spot_prices, "holiday_calendar": self.holiday_calendar, "normalization_config": self.normalization_config}


class CategoricalRegression(PFCShaper):
    """Linear regression model using categorical predictor variables to construct a PFC shape.

    .. math::

        S(t) = S_0 + \sum^{23}_{i=1}\\beta^h_i\cdot\mathbb{I}_{h(t)=i} + \\beta^d\cdot\mathbb{I}_{d(t)=1}  + \\beta^H\cdot\mathbb{I}_{H(t)=1} + \sum^{12}_{i=2}\\beta^m_i\cdot\mathbb{I}_{m(t)=i}

    where:\n
    :math:`S_0`: Spot price level\n
    :math:`\mathbb{I}_x = \\begin{cases} 1, & \\text{if the } x \\text{ expression renders true} \\\\ 0, & \\text{if the } x \\text{ expression renders false} \\end{cases}` \n
    :math:`h(t)`: Hour of t\n
    :math:`d(t) = \\begin{cases} 1, & \\text{if t is a weekday} \\\\ 0, & \\text{if t is a day on a weekend} \\end{cases}` \n
    :math:`H(t) = \\begin{cases} 1, & \\text{if t public holidy} \\\\ 0, & \\text{if t is not a public holiday} \\end{cases}` \n
    :math:`m(t)`: Month of t\n

    Args:
        spot_prices (pd.DataFrame): Data used to calibrate the shaping model.
        holiday_calendar (holidays.HolidayBase): Calendar object to obtain country specific holidays.
        normalization_config (Optional[Dict[Literal["D", "W", "ME"], Optional[int]]], optional): A dictionary configurating the shape normalization periods.
            Here ``D`` defines the number of days at the beginning of the shape over which the individual mean is normalized to one.
            ``W`` defines the number of weeks at the beginning of the shape over which the individual mean is normalized to one.
            ``ME`` defines the number of months at the beginning of the shape over which the individual mean is normalized to one. The remaining shape is then normalized over the individual years.Defaults to None.
    """

    def __init__(
        self,
        spot_prices: pd.DataFrame,
        holiday_calendar: holidays.HolidayBase,
        normalization_config: Optional[Dict[Literal["D", "W", "M"], Optional[int]]] = None,
    ):
        super().__init__(spot_prices=spot_prices, holiday_calendar=holiday_calendar, normalization_config=normalization_config)

    def _transform(self, datetimes_list: List[dt.datetime]) -> np.ndarray:
        """Transforms a list of datetimes in a numpy array which can then be used for the linear regression.

        Args:
            datetimes_list (List[dt.datetime]): List of datetimes

        Returns:
            np.ndarray: Numpy array containing the transformed datetimes list
        """
        _datetime_series = pd.Series(datetimes_list)

        weekday = _datetime_series.dt.weekday.isin([0, 1, 2, 3, 4]).astype(int).to_numpy().reshape(-1, 1)
        holiday = _datetime_series.isin(pd.to_datetime(list(self.holiday_calendar.keys()))).astype(int).to_numpy().reshape(-1, 1)

        predictors = [weekday, holiday]

        if len(_datetime_series.dt.hour.unique()) > 1:
            hours = (
                pd.get_dummies(_datetime_series.dt.hour, prefix="hour", drop_first=True)
                .astype(int)
                .to_numpy()
                .reshape(-1, len(_datetime_series.dt.hour.unique()) - 1)
            )
            predictors.append(hours)

        month = pd.get_dummies(_datetime_series.dt.month, prefix="month", drop_first=True).astype(int).to_numpy().reshape(-1, 11)

        offset = np.ones(shape=(len(_datetime_series), 1))
        return np.concatenate([offset, weekday, holiday, hours, month], axis=1)

    def _set_regression_parameters(self, params: np.ndarray):
        super()._set_regression_parameters(params=params)

    def calibrate(self) -> np.ndarray:
        data_array = self._transform(datetimes_list=self.spot_prices.index)
        self._set_regression_parameters(
            np.linalg.inv(data_array.T @ data_array) @ data_array.T @ self.spot_prices.iloc[:, 0].to_numpy().reshape(-1, 1)
        )
        return data_array @ self._regression_parameters

    def apply(self, apply_schedule: SimpleSchedule) -> pd.DataFrame:
        apply_schedule_datetime_list = apply_schedule.get_schedule()
        data_array = self._transform(datetimes_list=apply_schedule_datetime_list)
        shape = data_array @ self._regression_parameters

        shape_df = pd.DataFrame({"shape": shape.squeeze()}, index=apply_schedule_datetime_list)
        shape_df = self.normalize_shape(shape=shape_df)
        return shape_df

    def _to_dict(self):
        return super()._to_dict()
