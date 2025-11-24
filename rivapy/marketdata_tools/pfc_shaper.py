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
    def calibrate(self):
        """Calibration of the shaping model"""
        pass

    @abc.abstractmethod
    def apply(self, apply_schedule: List[dt.datetime]):
        """Applies the model on a schedule in order to generate a shape for future dates.

        Args:
            apply_schedule (List[dt.datetime]): List of datetimes in order to generate a shape for future dates.
        """
        pass

    def _preprocess(self, spot: pd.DataFrame, remove_outlier: bool, lower_quantile: float, upper_quantile: float) -> pd.DataFrame:
        """
        Preprocess spot price data by ensuring hourly continuity, interpolating missing values,
        and optionally removing outliers based on normalized yearly values.

        This method performs the following steps:
          1. Aggregates duplicate timestamps by taking the mean of their values.
          2. Reindexes the time series to ensure a continuous hourly frequency and
             linearly interpolates missing values.
          3. If `remove_outlier=True`, normalizes the time series on a yearly basis and
             removes data points outside the specified quantile range.

        Args:
            spot (pd.DataFrame): Raw spot price data indexed by datetime. The first column
                is assumed to contain the price values.
            remove_outlier (bool): Whether to remove outliers after normalization.
            lower_quantile (float): Lower quantile threshold (e.g., 0.01) used for outlier removal.
            upper_quantile (float): Upper quantile threshold (e.g., 0.99) used for outlier removal.

        Returns:
            pd.DataFrame: A cleaned and time-continuous spot price time series, with optional
            outliers removed.
        """
        # remove duplicate hours by replacing these with their mean
        spot = spot.groupby(level=0).mean()

        # include missing hours
        full_idx = pd.date_range(start=spot.index.min(), end=spot.index.max(), freq="h")
        spot = spot.reindex(full_idx)
        spot.index.name = "date"
        spot.iloc[:, 0] = spot.iloc[:, 0].interpolate(method="linear")

        if remove_outlier:

            yearly_normalized = self._normalize_year(df=spot)

            q_lower = np.quantile(yearly_normalized.iloc[:, 0].to_numpy(), lower_quantile)
            q_upper = np.quantile(yearly_normalized.iloc[:, 0].to_numpy(), upper_quantile)

            remove_ids = yearly_normalized.loc[(yearly_normalized.iloc[:, 0] < q_lower) | (yearly_normalized.iloc[:, 0] > q_upper), :].index
            spot = spot.loc[~spot.index.isin(remove_ids), :]
        return spot

    def _normalize_year(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize time series values by their yearly mean.

        This method computes the mean value for each calendar year and divides all data
        points within that year by the corresponding annual mean. The result is a
        year-normalized time series where each year has an average value of 1.

        Args:
            df (pd.DataFrame): A DataFrame indexed by datetime, containing one or more
                numeric columns to be normalized.

        Returns:
            pd.DataFrame: A DataFrame where the values of each year have been normalized
            relative to their annual mean.
        """
        yearly_mean = df.resample("YE").transform("mean")

        normalized = df / yearly_mean
        return normalized

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

        if self.normalization_config is None:
            shape_df = self._normalize_year(df=shape)
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
            yearly_normalized_shape = self._normalize_year(df=leftover_shape)

            return pd.concat(normalized_shapes + [yearly_normalized_shape], axis=0).sort_index(ascending=True)

    def _to_dict(self):
        return {"spot_prices": self.spot_prices, "holiday_calendar": self.holiday_calendar, "normalization_config": self.normalization_config}


class SimpleCategoricalRegression(PFCShaper):
    r"""Linear regression model using categorical predictor variables to construct a PFC shape.

    .. math::

        s(t) = s_0 + \sum^{23}_{i=1}\beta^h_i\cdot\mathbb{I}_{h(t)=i} + \beta^d\cdot\mathbb{I}_{d(t)=1}  + \beta^H\cdot\mathbb{I}_{H(t)=1} + \sum^{12}_{i=2}\beta^m_i\cdot\mathbb{I}_{m(t)=i}

    where:

    :math:`s_0`: Shape level level

    :math:`\mathbb{I}_x = \begin{cases} 1, & \text{if the } x \text{ expression renders true} \\ 0, & \text{if the } x \text{ expression renders false} \end{cases}`

    :math:`h(t)`: Hour of t

    :math:`d(t) = \begin{cases} 1, & \text{if t is a weekday} \\ 0, & \text{if t is a day on a weekend} \end{cases}`

    :math:`H(t) = \begin{cases} 1, & \text{if t public holidy} \\ 0, & \text{if t is not a public holiday} \end{cases}`

    :math:`m(t)`: Month of t

    Args:
        spot_prices (pd.DataFrame): Data used to calibrate the shaping model.
        holiday_calendar (holidays.HolidayBase): Calendar object to obtain country specific holidays.
        normalization_config (Optional[Dict[Literal["D", "W", "ME"], Optional[int]]], optional): A dictionary configurating the shape normalization periods.
            Here ``D`` defines the number of days at the beginning of the shape over which the individual mean is normalized to one.
            ``W`` defines the number of weeks at the beginning of the shape over which the individual mean is normalized to one.
            ``ME`` defines the number of months at the beginning of the shape over which the individual mean is normalized to one. The remaining shape is then normalized over the individual years.Defaults to None.
        remove_outlier (bool): Wether to remove outliers for the seasonality shape regression. Defaults to False.
        lower_quantile (float): Lower quantile for outlier detection. Defauls to 0.005.
        upper_quantile (float): Upper quantile for outlier detection. Defaults to 0.995.
    """

    def __init__(
        self,
        spot_prices: pd.DataFrame,
        holiday_calendar: holidays.HolidayBase,
        normalization_config: Optional[Dict[Literal["D", "W", "M"], Optional[int]]] = None,
        remove_outlier: bool = False,
        lower_quantile: float = 0.005,
        upper_quantile: float = 0.995,
    ):
        super().__init__(spot_prices=spot_prices, holiday_calendar=holiday_calendar, normalization_config=normalization_config)
        self.remove_outlier = remove_outlier
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

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
        return np.concatenate([offset, weekday, holiday, month, hours], axis=1)

    def calibrate(self):
        spot = self.spot_prices.copy()
        spot = self._preprocess(spot=spot, remove_outlier=self.remove_outlier, lower_quantile=self.lower_quantile, upper_quantile=self.upper_quantile)

        spot_normalized = self._normalize_year(spot)
        data_array = self._transform(datetimes_list=self.spot_prices.index)
        self._regression_parameters = np.linalg.inv(data_array.T @ data_array) @ data_array.T @ spot_normalized.iloc[:, 0].to_numpy().reshape(-1, 1)

        # fit = data_array @ self._regression_parameters

        # df = pd.DataFrame(fit.squeeze(), index=spot.index, columns=["shape"])
        # return self._normalize_year(df)

    def apply(self, apply_schedule: List[dt.datetime]) -> pd.DataFrame:
        data_array = self._transform(datetimes_list=apply_schedule)
        shape = data_array @ self._regression_parameters

        shape_df = pd.DataFrame({"shape": shape.squeeze()}, index=apply_schedule)
        shape_df = self.normalize_shape(shape=shape_df)
        return shape_df

    def _to_dict(self):
        return super()._to_dict()


class CategoricalRegression(PFCShaper):
    r"""Linear regression model using categorical predictor variables to construct a PFC shape.
    We follow the methodology in:
    
    https://cem-a.org/wp-content/uploads/2019/10/A-Structureal-Model-for-Electricity-Forward-Prices.pdf
    
    https://ieeexplore.ieee.org/document/6607349
    
    https://www.researchgate.net/publication/229051446_Robust_Calculation_and_Parameter_Estimation_of_the_Hourly_Price_Forward_Curve

    We create a regression model for bot the seasonality shape and the intra day shape. For the regression model of the seasonality shape, 
    the days are split into weekday, Saturdays and Sundays. Public holidays are considered as Sundays while bridge days are expected to behave like Saturdays.
    Afterwards, weekdays are split into clusters representing the month they are in, while Saturdays and Sundays are assigned to clusters reaching over three months.
    For the regression model of the intra day shape we keep the seasonality clusters but add a hourly cluster for each individual hour such that the
    total number of intra day clusters becomes #Season Clusters * 24.
    
    .. math::
        \begin{aligned}
            y_\text{season}(d) &= \frac{\frac{1}{24}\sum_{i=1}^{24}h_i^d}{\frac{1}{N_y}\sum_{i=1}^{N_y} h_i^y} \\
            \hat{y}_\text{season}(d) & =\beta^{0}_\text{season} + \sum_{c \in C^\text{season}}\beta^c\cdot\mathbb{I}_{\text{Cluster}(d)=c} \\
            y_\text{id}(h,d) &= \frac{h_i^d}{\frac{1}{24}\sum_{i=1}^{24}h_i^d} \\
            \hat{y}_\text{id}(h,d) & =\beta^{0}_\text{id} + \sum_{c \in C^\text{id}}\beta^c\cdot\mathbb{I}_{\text{Cluster}(h)=c} \\
            s(h,d) &= \hat{y}_\text{id}(h,d)\cdot\hat{y}_\text{season}(d)
        \end{aligned}
    
    where:
    
    :math:`h_i^d`: i-th hour of d-th day
    
    :math:`h_i^y`: i-th hour of the year y
    
    :math:`N_y`: number of days in year y
    
    :math:`C^\text{season}`: set of all clusters for the seasonality shape
    
    :math:`C^\text{id}`: set of all clusters for the intra day shape
    
    :math:`\text{Cluster}(X)`: returns the cluster of X

    :math:`\mathbb{I}_x = \begin{cases}
    1, & \text{if the } x \text{ expression renders true}\\
    0, & \text{if the } x \text{ expression renders false}
    \end{cases}`

    Args:
        spot_prices (pd.DataFrame): Data used to calibrate the shaping model.
        holiday_calendar (holidays.HolidayBase): Calendar object to obtain country specific holidays.
        normalization_config (Optional[Dict[Literal["D", "W", "ME"], Optional[int]]], optional): A dictionary configurating the shape normalization periods.
            Here ``D`` defines the number of days at the beginning of the shape over which the individual mean is normalized to one.
            ``W`` defines the number of weeks at the beginning of the shape over which the individual mean is normalized to one.
            ``ME`` defines the number of months at the beginning of the shape over which the individual mean is normalized to one. The remaining shape is then normalized over the individual years.Defaults to None.
        remove_outlier_season (bool): Wether to remove outliers for the seasonality shape regression. Defaults to False.
        remove_outlier_id (bool): Wether to remove outliers for the intra day shape regression. Defaults to False.
        lower_quantile_season (float): Lower quantile for outlier detection. Defauls to 0.005.
        upper_quantile_season (float): Upper quantile for outlier detection. Defaults to 0.995.
        lower_quantile_id (float): Lower quantile for outlier detection. Defauls to 0.005.
        upper_quantile_id (float): Upper quantile for outlier detection. Defaults to 0.995.
    """

    def __init__(
        self,
        spot_prices: pd.DataFrame,
        holiday_calendar: holidays.HolidayBase,
        normalization_config: Optional[Dict[Literal["D", "W", "M"], Optional[int]]] = None,
        remove_outlier_season: bool = False,
        remove_outlier_id: bool = False,
        lower_quantile_season: float = 0.005,
        upper_quantile_season: float = 0.995,
        lower_quantile_id: float = 0.005,
        upper_quantile_id: float = 0.995,
    ):
        super().__init__(spot_prices=spot_prices, holiday_calendar=holiday_calendar, normalization_config=normalization_config)
        self.remove_outlier_season = remove_outlier_season
        self.remove_outlier_id = remove_outlier_id
        self.lower_quantile_season = lower_quantile_season
        self.upper_quantile_season = upper_quantile_season
        self.lower_quantile_id = lower_quantile_id
        self.upper_quantile_id = upper_quantile_id

    def _create_cluster_df(self, day_list: List[dt.datetime], use_hours: bool = False):
        """Create a DataFrame containing the clusters for the regression models.

        Args:
            day_list (List[dt.datetime]): List of datetimes for which a clustering should be performed
            use_hours (bool, optional): Wether to extend the clustering to include hours. Defaults to False.

        Returns:
            None
        """
        holidays_list = pd.to_datetime(list(self.holiday_calendar.keys()))
        cluster_df = pd.DataFrame(index=day_list)

        cluster_df["year"] = cluster_df.index.year
        cluster_df["month"] = cluster_df.index.month
        cluster_df["day"] = cluster_df.index.day
        cluster_df["weekday"] = cluster_df.index.weekday

        if use_hours:
            cluster_df["hour"] = cluster_df.index.hour

        # get holidays and bridge days
        temp_cluster_df = cluster_df[["year", "month", "day", "weekday"]].drop_duplicates().sort_index()
        temp_cluster_df["holiday"] = 0
        temp_cluster_df["bridge"] = 0
        temp_cluster_df.loc[temp_cluster_df.index.isin(holidays_list), "holiday"] = 1

        is_monday_bridge = (temp_cluster_df.index + pd.Timedelta(days=1)).isin(holidays_list) & (temp_cluster_df.index.weekday == 0)
        is_friday_bridge = (temp_cluster_df.index - pd.Timedelta(days=1)).isin(holidays_list) & (temp_cluster_df.index.weekday == 4)
        temp_cluster_df.loc[is_friday_bridge | is_monday_bridge, "bridge"] = 1

        cluster_df = pd.merge(cluster_df, temp_cluster_df, on=["year", "month", "day", "weekday"])
        # cluster_df.set_index(day_list, inplace=True)
        cluster_df.index = day_list

        cluster_df["day_indicator"] = 0
        cluster_df.loc[cluster_df.index.weekday < 5, "day_indicator"] = 1
        cluster_df.loc[cluster_df.index.weekday == 5, "day_indicator"] = 2
        cluster_df.loc[cluster_df.index.weekday == 6, "day_indicator"] = 3

        cluster_df.loc[cluster_df["holiday"] == 1, "day_indicator"] = 3
        cluster_df.loc[cluster_df["bridge"] == 1, "day_indicator"] = 2

        cluster_df.loc[(cluster_df.index.month == 12) & (cluster_df.index.day.isin([24, 31])) & (cluster_df.index.weekday < 5), "day_indicator"] = 2

        cluster_df.loc[:, "cluster"] = 0
        cluster_df.loc[cluster_df["day_indicator"] == 1, "cluster"] = cluster_df.loc[cluster_df["day_indicator"] == 1, "month"]

        weekend_cluster_month = [[1, 2, 12], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        # weekend_cluster_month = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12]]
        count = cluster_df["month"].max()

        for day_indicator in [2, 3]:
            for month_lst in weekend_cluster_month:
                if not len(cluster_df.loc[(cluster_df["day_indicator"] == day_indicator) & (cluster_df["month"].isin(month_lst)), "cluster"]) == 0:
                    count += 1
                    cluster_df.loc[(cluster_df["day_indicator"] == day_indicator) & (cluster_df["month"].isin(month_lst)), "cluster"] = count

        if use_hours:
            self.__add_hours_cluster(
                df=cluster_df,
                clusters_clmn="cluster",
                hours_clmn="hour",
                unique_clusters=cluster_df["cluster"].unique(),
                unique_hours=cluster_df["hour"].unique(),
            )

        return cluster_df

    def __add_hours_cluster(self, df: pd.DataFrame, clusters_clmn: str, hours_clmn: str, unique_clusters: List[int], unique_hours: List[int]):
        """Add hourly clustering in the `cluster_hours` column of the provided DataFrame

        Args:
            df (pd.DataFrame): DataFrame containing the infos needed for an hourly clustering.
            clusters_clmn (str): Column containing the seasonality clusters
            hours_clmn (str): Columns containing the hours
            unique_clusters (List[int]): List of all clusters
            unique_hours (List[int]): List of all hourly clusters
        """
        df["cluster_hours"] = 0
        count = 1
        for cluster in unique_clusters:
            for hour in unique_hours:
                df.loc[(df[clusters_clmn] == cluster) & (df[hours_clmn] == hour), "cluster_hours"] = count
                count += 1

    def _create_one_hot_matrix(self, rows: int, clusters: pd.Series, max_clusters: int, adjust_clusters: bool, offset_col: bool):
        """Create a matrix for a one hot encoding of a clusters pandas Series.

        Args:
            rows (int): Number of data points
            clusters (pd.Series): Series containing the cluster for each data point
            max_clusters (int): Total number of individual clusters
            adjust_clusters (bool): Wether to adjust cluster by subtracting each cluster by one.
            offset_col (bool): Wether to use the last column as an intercept for the regression model.

        Returns:
            _type_: _description_
        """
        one_hot = np.zeros(shape=(rows, max_clusters))
        if adjust_clusters:
            cluster_series = clusters - 1
        else:
            cluster_series = clusters

        one_hot[np.arange(rows), cluster_series] = 1

        if offset_col:
            one_hot[:, -1] = 1
        else:
            one_hot = one_hot[:, :-1]

        return one_hot

    def _preprocess(self, spot: pd.DataFrame) -> pd.DataFrame:
        # remove duplicate hours by replacing these with their mean
        spot = spot.groupby(level=0).mean()

        # include missing hours
        full_idx = pd.date_range(start=spot.index.min(), end=spot.index.max(), freq="h")
        spot = spot.reindex(full_idx)
        spot.index.name = "date"
        spot.iloc[:, 0] = spot.iloc[:, 0].interpolate(method="linear")
        return spot

    @staticmethod
    def _remove_outliers(df: pd.DataFrame, value_clmn: str, grouping_clmn: str, lower_quantile: float, upper_quantile: float):
        """
        Remove outliers from a DataFrame based on quantile thresholds within groups.

        This function applies a quantile-based filter to the values in `value_clmn` for each
        unique category defined in `grouping_clmn`. For each group, values below the
        `lower_quantile` or above the `upper_quantile` are considered outliers and removed.
        The filtered rows are then returned as a cleaned DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame containing the data.
            value_clmn (str): Name of the column containing the numerical values to evaluate for outliers.
            grouping_clmn (str): Name of the column used to group the data before applying the quantile filter.
            lower_quantile (float): Lower quantile threshold (e.g., 0.05). Values below this quantile are removed.
            upper_quantile (float): Upper quantile threshold (e.g., 0.95). Values above this quantile are removed.

        Returns:
            pd.DataFrame: A DataFrame containing only the data points within the specified quantile bounds for each group.
        """

        def remove_outliers(series, lower_quantile=lower_quantile, upper_quantile=upper_quantile):
            lower_bound = series.quantile(lower_quantile)
            upper_bound = series.quantile(upper_quantile)
            return series[(series >= lower_bound) & (series <= upper_bound)]

        keep_ids = df.groupby(grouping_clmn, group_keys=False)[value_clmn].apply(remove_outliers).index
        df_clean = df.loc[df.index.isin(keep_ids), :]
        return df_clean

    def calibrate(
        self,
    ):
        spot = self.spot_prices.copy()
        spot = self._preprocess(spot=spot)

        cluster_df = self._create_cluster_df(spot.index, use_hours=True)

        season_shape = self._normalize_year(spot)
        season_shape = season_shape.resample("D").mean().dropna()

        cluster_df_daily = cluster_df[["year", "month", "day", "weekday", "day_indicator", "cluster"]].drop_duplicates().sort_index()
        calib_season_df = pd.merge(season_shape, cluster_df_daily, left_index=True, right_index=True)

        if self.remove_outlier_season:
            value_clmn = calib_season_df.columns[0]
            calib_season_df = self._remove_outliers(
                df=calib_season_df,
                value_clmn=value_clmn,
                grouping_clmn="cluster",
                lower_quantile=self.lower_quantile_season,
                upper_quantile=self.upper_quantile_season,
            )

        self.__max_cluster = calib_season_df["cluster"].max()

        season_one_hot = self._create_one_hot_matrix(
            rows=len(calib_season_df),
            clusters=calib_season_df["cluster"],
            max_clusters=self.__max_cluster,
            adjust_clusters=True,  # since clusters do not start at 0
            offset_col=True,  # since we would ignore the last column because it is obsolete due to our categorical variables,
            # we actually set it all to 1 to account for the offset in our regression model
        )

        self._season_regression_params = (
            np.linalg.inv(season_one_hot.T @ season_one_hot) @ season_one_hot.T @ calib_season_df.iloc[:, 0].to_numpy().reshape(-1, 1)
        )

        id_shape = spot / spot.resample("D").transform("mean").dropna()

        calib_id_df = pd.merge(id_shape, cluster_df, left_index=True, right_index=True)

        if self.remove_outlier_id:
            value_clmn = calib_id_df.columns[0]
            calib_id_df = self._remove_outliers(
                df=calib_id_df,
                grouping_clmn="cluster_hours",
                value_clmn=value_clmn,
                lower_quantile=self.lower_quantile_id,
                upper_quantile=self.upper_quantile_id,
            )

        self.__max_hour_clusters = calib_id_df["cluster_hours"].max()

        id_one_hot = self._create_one_hot_matrix(
            rows=len(calib_id_df),
            clusters=calib_id_df["cluster_hours"],
            max_clusters=self.__max_hour_clusters,
            adjust_clusters=True,
            offset_col=True,
        )

        self._id_regression_params = np.linalg.inv(id_one_hot.T @ id_one_hot) @ id_one_hot.T @ calib_id_df.iloc[:, 0].to_numpy().reshape(-1, 1)

    def apply(self, apply_schedule: List[dt.datetime]) -> pd.DataFrame:
        cluster_df = self._create_cluster_df(apply_schedule, use_hours=True)

        season_one_hot = self._create_one_hot_matrix(
            rows=len(cluster_df),
            clusters=cluster_df["cluster"],
            max_clusters=self.__max_cluster,
            adjust_clusters=True,
            offset_col=True,
        )

        id_one_hot = self._create_one_hot_matrix(
            rows=len(cluster_df),
            clusters=cluster_df["cluster_hours"],
            max_clusters=self.__max_hour_clusters,
            adjust_clusters=True,
            offset_col=True,
        )

        season_fit = season_one_hot @ self._season_regression_params
        id_fit = id_one_hot @ self._id_regression_params

        cluster_df["shape"] = (season_fit * id_fit).squeeze()
        cluster_df["shape"] = self._normalize_year(df=cluster_df.loc[:, "shape"])
        shape_df = pd.DataFrame(cluster_df.loc[:, "shape"])
        shape_df = self.normalize_shape(shape=shape_df)
        return shape_df

    def _to_dict(self):
        return super()._to_dict()


class CategoricalFourierShaper(PFCShaper):
    r"""



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
        k_fourier: int = 2,
        remove_outlier_season: bool = True,
        remove_outlier_id: bool = True,
        lower_quantile_season: float = 0.005,
        upper_quantile_season: float = 0.995,
        lower_quantile_id: float = 0.005,
        upper_quantile_id: float = 0.995,
    ):
        super().__init__(spot_prices=spot_prices, holiday_calendar=holiday_calendar, normalization_config=normalization_config)
        self.k_fourier = k_fourier

        self.remove_outlier_season = remove_outlier_season
        self.remove_outlier_id = remove_outlier_id
        self.lower_quantile_season = lower_quantile_season
        self.upper_quantile_season = upper_quantile_season
        self.lower_quantile_id = lower_quantile_id
        self.upper_quantile_id = upper_quantile_id

    def _create_cluster_df(self, day_list: List[dt.datetime], use_hours: bool = False):
        holidays_list = pd.to_datetime(list(self.holiday_calendar.keys()))
        cluster_df = pd.DataFrame(index=day_list)

        cluster_df["year"] = cluster_df.index.year
        cluster_df["month"] = cluster_df.index.month
        cluster_df["day"] = cluster_df.index.day
        cluster_df["weekday"] = cluster_df.index.weekday

        if use_hours:
            cluster_df["hour"] = cluster_df.index.hour

        # get holidays and bridge days
        temp_cluster_df = cluster_df[["year", "month", "day", "weekday"]].drop_duplicates().sort_index()
        temp_cluster_df["holiday"] = 0
        temp_cluster_df["bridge"] = 0
        temp_cluster_df.loc[temp_cluster_df.index.isin(holidays_list), "holiday"] = 1

        is_monday_bridge = (temp_cluster_df.index + pd.Timedelta(days=1)).isin(holidays_list) & (temp_cluster_df.index.weekday == 0)
        is_friday_bridge = (temp_cluster_df.index - pd.Timedelta(days=1)).isin(holidays_list) & (temp_cluster_df.index.weekday == 4)
        temp_cluster_df.loc[is_friday_bridge | is_monday_bridge, "bridge"] = 1

        cluster_df = pd.merge(cluster_df, temp_cluster_df, on=["year", "month", "day", "weekday"])
        # cluster_df.set_index(day_list, inplace=True)
        cluster_df.index = day_list

        cluster_df["day_indicator"] = 0
        cluster_df.loc[cluster_df.index.weekday < 5, "day_indicator"] = 1
        cluster_df.loc[cluster_df.index.weekday == 5, "day_indicator"] = 2
        cluster_df.loc[cluster_df.index.weekday == 6, "day_indicator"] = 3

        cluster_df.loc[cluster_df["holiday"] == 1, "day_indicator"] = 3
        cluster_df.loc[cluster_df["bridge"] == 1, "day_indicator"] = 2

        cluster_df.loc[(cluster_df.index.month == 12) & (cluster_df.index.day.isin([24, 31])) & (cluster_df.index.weekday < 5), "day_indicator"] = 2

        cluster_df.loc[:, "cluster"] = 0
        cluster_df.loc[cluster_df["day_indicator"] == 1, "cluster"] = cluster_df.loc[cluster_df["day_indicator"] == 1, "month"]

        weekend_cluster_month = [[1, 2, 12], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        count = cluster_df["month"].max()

        for day_indicator in [2, 3]:
            for month_lst in weekend_cluster_month:
                if not len(cluster_df.loc[(cluster_df["day_indicator"] == day_indicator) & (cluster_df["month"].isin(month_lst)), "cluster"]) == 0:
                    count += 1
                    cluster_df.loc[(cluster_df["day_indicator"] == day_indicator) & (cluster_df["month"].isin(month_lst)), "cluster"] = count

        if use_hours:
            self.__add_hours_cluster(
                df=cluster_df,
                clusters_clmn="cluster",
                hours_clmn="hour",
                unique_clusters=cluster_df["cluster"].unique(),
                unique_hours=cluster_df["hour"].unique(),
            )

        return cluster_df

    def __add_hours_cluster(self, df: pd.DataFrame, clusters_clmn: str, hours_clmn: str, unique_clusters: List[int], unique_hours: List[int]):
        df["cluster_hours"] = 0
        count = 1
        for cluster in unique_clusters:
            for hour in unique_hours:
                df.loc[(df[clusters_clmn] == cluster) & (df[hours_clmn] == hour), "cluster_hours"] = count
                count += 1

    def _create_one_hot_matrix(self, rows: int, clusters: pd.Series, max_clusters: int, adjust_clusters: bool, offset_col: bool):
        one_hot = np.zeros(shape=(rows, max_clusters))
        if adjust_clusters:
            cluster_series = clusters - 1
        else:
            cluster_series = clusters

        one_hot[np.arange(rows), cluster_series] = 1

        if offset_col:
            one_hot[:, -1] = 1
        else:
            one_hot = one_hot[:, :-1]

        return one_hot

    def times_to_zero_one(self, h: pd.DatetimeIndex):
        """
        Convert any time point into the corresponding fraction w.r.t. the
        beginning of the year it belongs to.
        1st of January are always zeros, while 31th of December can be
        1 / 365 or 1/366 according to the year.
        The idea is to map any year to [0,1), on which the seasonality
        curve, periodic on [0,1], is fitted.

        """
        if not isinstance(h, pd.DatetimeIndex):
            raise TypeError("index must be of type pd.DatetimeIndex")
        if len(h) < 1:
            raise ValueError("index must contain at least one value!")
        # Build a DataFrame where each point is the start of the year
        # w.r.t. each date in h
        start_of_years = pd.to_datetime(h.year.astype(str) + "-01-01 00:00:00").tz_localize(h.tz)
        # Build a DataFrame where each point is the start of the year
        # w.r.t. each date in h
        end_of_years = pd.to_datetime((h.year + 1).astype(str) + "-01-01 00:00:00").tz_localize(h.tz)
        # Compute then the fractions, using pandas vectorization
        result = np.array((h - start_of_years) / (end_of_years - start_of_years))
        # Internal sanity check: all points must lie in [0, 1)
        assert all(result < 1.0) and all(result >= 0.0)
        return result

    def _preprocess(self, spot: pd.DataFrame) -> pd.DataFrame:
        # remove duplicate hours by replacing these with their mean
        spot = spot.groupby(level=0).mean()

        # include missing hours
        full_idx = pd.date_range(start=spot.index.min(), end=spot.index.max(), freq="h")
        spot = spot.reindex(full_idx)
        spot.index.name = "date"
        spot.iloc[:, 0] = spot.iloc[:, 0].interpolate(method="linear")
        return spot

    @staticmethod
    def _remove_outliers(df: pd.DataFrame, value_clmn: str, grouping_clmn: str, lower_quantile: float, upper_quantile: float):
        def remove_outliers(series, lower_quantile=lower_quantile, upper_quantile=upper_quantile):
            lower_bound = series.quantile(lower_quantile)
            upper_bound = series.quantile(upper_quantile)
            return series[(series >= lower_bound) & (series <= upper_bound)]

        keep_ids = df.groupby(grouping_clmn, group_keys=False)[value_clmn].apply(remove_outliers).index
        df_clean = df.loc[df.index.isin(keep_ids), :]
        return df_clean

    def calibrate(
        self,
    ):
        spot = self.spot_prices.copy()
        spot = self._preprocess(spot=spot)

        cluster_df = self._create_cluster_df(spot.index, use_hours=True)

        season_shape = self._normalize_year(spot)
        season_shape = season_shape.resample("D").mean().dropna()

        cluster_df_daily = cluster_df[["year", "month", "day", "weekday", "day_indicator", "cluster"]].drop_duplicates().sort_index()
        calib_season_df = pd.merge(season_shape, cluster_df_daily, left_index=True, right_index=True)

        if self.remove_outlier_season:
            value_clmn = calib_season_df.columns[0]
            calib_season_df = self._remove_outliers(
                df=calib_season_df,
                value_clmn=value_clmn,
                grouping_clmn="cluster",
                lower_quantile=self.lower_quantile_season,
                upper_quantile=self.upper_quantile_season,
            )

        self.__max_cluster = calib_season_df["cluster"].max()

        season_one_hot = self._create_one_hot_matrix(
            rows=len(calib_season_df),
            clusters=calib_season_df["cluster"],
            max_clusters=self.__max_cluster,
            adjust_clusters=True,  # since clusters do not start at 0
            offset_col=True,  # since we would ignore the last column because it is obsolete due to our categorical variables,
            # we actually set it all to 1 to account for the offset in our regression model
        )

        self._season_regression_params = (
            np.linalg.inv(season_one_hot.T @ season_one_hot) @ season_one_hot.T @ calib_season_df.iloc[:, 0].to_numpy().reshape(-1, 1)
        )

        id_shape = self._normalize_year(spot)

        calib_id_df = pd.merge(id_shape, cluster_df, left_index=True, right_index=True)
        calib_id_df["t"] = self.times_to_zero_one(calib_id_df.index)

        if self.remove_outlier_id:
            value_clmn = calib_id_df.columns[0]
            calib_id_df = self._remove_outliers(
                df=calib_id_df,
                grouping_clmn="hour",
                value_clmn=value_clmn,
                lower_quantile=self.lower_quantile_id,
                upper_quantile=self.upper_quantile_id,
            )

        self._id_params = {}
        for h in calib_id_df["hour"].unique():
            x = calib_id_df.loc[calib_id_df["hour"] == h, "t"].to_numpy().reshape(-1, 1)
            y = calib_id_df.loc[calib_id_df["hour"] == h, :].iloc[:, 0].to_numpy().reshape(-1, 1)
            m = self.k_fourier * 2 + 1
            x_fit = np.zeros((x.shape[0], m))
            fourier_parts = []
            for k in np.arange(start=1, stop=self.k_fourier + 1):
                fourier_parts.append(np.sin(2 * k * np.pi * x))
                fourier_parts.append(np.cos(2 * k * np.pi * x))

            x_fit[:, :-1] = np.concatenate(fourier_parts, axis=1)
            x_fit[:, -1] = 1.0
            self._id_params[h] = np.linalg.pinv(x_fit) @ y

    def apply(self, apply_schedule: List[dt.datetime]) -> pd.DataFrame:
        cluster_df = self._create_cluster_df(apply_schedule, use_hours=True)

        cluster_df["t"] = self.times_to_zero_one(cluster_df.index)

        season_one_hot = self._create_one_hot_matrix(
            rows=len(cluster_df),
            clusters=cluster_df["cluster"],
            max_clusters=self.__max_cluster,
            adjust_clusters=True,
            offset_col=True,
        )

        season_fit = season_one_hot @ self._season_regression_params
        self._season_fit = season_fit

        cluster_df["id_fit"] = 0.0
        for h in cluster_df["hour"].unique():
            x = cluster_df.loc[cluster_df["hour"] == h, "t"].to_numpy().reshape(-1, 1)
            x_fit = np.zeros((x.shape[0], 5))
            m = self.k_fourier * 2 + 1
            x_fit = np.zeros((x.shape[0], m))
            fourier_parts = []
            for k in np.arange(start=1, stop=self.k_fourier + 1):
                fourier_parts.append(np.sin(2 * k * np.pi * x))
                fourier_parts.append(np.cos(2 * k * np.pi * x))
            x_fit[:, :-1] = np.concatenate(fourier_parts, axis=1)
            x_fit[:, -1] = 1.0
            cluster_df.loc[cluster_df["hour"] == h, "id_fit"] = (x_fit @ self._id_params[h]).squeeze()

        cluster_df["shape"] = season_fit.squeeze() * cluster_df["id_fit"].to_numpy()
        cluster_df["shape"] = self._normalize_year(df=cluster_df.loc[:, "shape"])
        return pd.DataFrame(cluster_df.loc[:, "shape"])

    def _to_dict(self):
        return super()._to_dict()
