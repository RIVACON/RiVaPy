import unittest
import holidays
import itertools
import pandas as pd
import numpy as np
import datetime as dt
from rivapy.tools.scheduler import SimpleSchedule, OffPeakSchedule, PeakSchedule, GasSchedule, BaseSchedule
from rivapy.marketdata_tools import PFCShifter
from rivapy.marketdata_tools.pfc_shaper import PFCShaper, CategoricalRegression
from rivapy.instruments.energy_futures_specifications import EnergyFutureSpecifications
from rivapy.sample_data.dummy_power_spot_price import spot_price_model
from rivapy.marketdata.curves import EnergyPriceForwardCurve
from typing import Dict
from collections import defaultdict


class TestPFCShifter(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestPFCShifter, self).__init__(*args, **kwargs)
        self.parameter_dict = {
            "spot_price_level": 100,
            "peak_price_level": 10,
            "solar_price_level": 8,
            "weekend_price_level": 10,
            "winter_price_level": 20,
            "epsilon_mean": 0,
            "epsilon_var": 5,
        }
        self.date_range = pd.date_range(start="1/1/2023", end="1/1/2025", freq="h", inclusive="left")
        self.spot_prices = list(map(lambda x: spot_price_model(x, **self.parameter_dict), self.date_range))
        self.df = pd.DataFrame(data=self.spot_prices, index=self.date_range, columns=["Spot"])
        base_y = self.df.resample("YE").mean()
        base_y.index = base_y.index.strftime("%Y")

        df_spot = self.df.copy()
        df_spot.index = df_spot.index.strftime("%Y")

        shape = df_spot.divide(base_y, axis="index")
        self.shape_df = pd.DataFrame(data=shape["Spot"].tolist(), index=self.date_range, columns=["shape"])

    def __get_contracts_dict(self, contracts_schedules: Dict[str, Dict[str, SimpleSchedule]]) -> Dict[str, Dict[str, EnergyFutureSpecifications]]:
        contracts = defaultdict(dict)
        for contract_type, contracts_dict in contracts_schedules.items():
            for contract_name, schedule in contracts_dict.items():
                tg = schedule.get_schedule()
                price = self.df.loc[tg, :].mean().iloc[0]
                contracts[contract_type][contract_name] = EnergyFutureSpecifications(schedule=schedule, price=price, name=contract_name)
        return dict(contracts)

    def test_pfc_shifter(self):
        contracts_schedules = {
            "off_peak": {
                "Cal23_OffPeak": OffPeakSchedule(dt.datetime(2023, 1, 1), dt.datetime(2024, 1, 1)),
                "Q2/23_OffPeak": OffPeakSchedule(dt.datetime(2023, 4, 1), dt.datetime(2023, 7, 1)),
            },
            "peak": {
                "Cal23_Peak": PeakSchedule(dt.datetime(2023, 1, 1), dt.datetime(2024, 1, 1)),
                "Cal24_Peak": PeakSchedule(dt.datetime(2024, 1, 1), dt.datetime(2025, 1, 1)),
                "Q1/23_Peak": PeakSchedule(dt.datetime(2023, 1, 1), dt.datetime(2023, 4, 1)),
                "Q2/23_Peak": PeakSchedule(dt.datetime(2023, 4, 1), dt.datetime(2023, 7, 1)),
                "Q3/23_Peak": PeakSchedule(dt.datetime(2023, 7, 1), dt.datetime(2023, 10, 1)),
                "Q4/23_Peak": PeakSchedule(dt.datetime(2023, 10, 1), dt.datetime(2024, 1, 1)),
                "Q2/24_Peak": PeakSchedule(dt.datetime(2024, 4, 1), dt.datetime(2024, 7, 1)),
            },
        }
        contracts = self.__get_contracts_dict(contracts_schedules=contracts_schedules)
        data_dict = {
            "off_peak": self.shape_df.loc[
                SimpleSchedule(
                    dt.datetime(2023, 1, 1),
                    dt.datetime(2024, 1, 1),
                    freq="h",
                    hours=[0, 1, 2, 3, 4, 5, 6, 7, 20, 21, 22, 23],
                    ignore_hours_for_weekdays=[5, 6],
                ).get_schedule()
            ],
            "peak": self.shape_df.loc[
                SimpleSchedule(
                    dt.datetime(2023, 1, 1),
                    dt.datetime(2025, 1, 1),
                    freq="h",
                    hours=[8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                    weekdays=[0, 1, 2, 3, 4],
                ).get_schedule()
            ],
        }
        result = []
        for contracts_type, data in data_dict.items():
            pfc_shifter = PFCShifter(shape=data, contracts=list(contracts[contracts_type].values()))
            result.append(pfc_shifter.compute())

        df_shifted = pd.concat(result, axis=0)
        df_shifted.sort_index(inplace=True, ascending=True)

        contracts_all = self.__get_contracts_dict(contracts_schedules=contracts_schedules)
        for contracts_type, contracts_dict in contracts_all.items():
            for contract_name, contract_spec in contracts_dict.items():
                shift_price = df_shifted.loc[contract_spec.get_schedule(), :].mean().iloc[0]
                self.assertAlmostEqual(contract_spec.price, shift_price, delta=10 ** (-10))

    def test_only_overlapping_contracts(self):
        contracts_schedules = {
            "GasBase": {
                "Cal23_GasBase": GasSchedule(dt.datetime(2023, 1, 1), dt.datetime(2024, 1, 1)),
                "Q1/23_GasBase": GasSchedule(dt.datetime(2023, 1, 1), dt.datetime(2023, 4, 1)),
                "Q2/23_GasBase": GasSchedule(dt.datetime(2023, 4, 1), dt.datetime(2023, 7, 1)),
                "Q3/23_GasBase": GasSchedule(dt.datetime(2023, 7, 1), dt.datetime(2023, 10, 1)),
                "Q4/23_GasBase": GasSchedule(dt.datetime(2023, 10, 1), dt.datetime(2024, 1, 1)),
            },
        }
        contracts = self.__get_contracts_dict(contracts_schedules=contracts_schedules)
        data_dict = {
            "GasBase": self.shape_df.loc[GasSchedule(dt.datetime(2023, 1, 1), dt.datetime(2024, 1, 1)).get_schedule()],
        }
        result = []
        for contracts_type, data in data_dict.items():
            pfc_shifter = PFCShifter(shape=data, contracts=list(contracts[contracts_type].values()))
            result.append(pfc_shifter.compute())

        df_shifted = pd.concat(result, axis=0)
        df_shifted.sort_index(inplace=True, ascending=True)

        contracts_all = self.__get_contracts_dict(contracts_schedules=contracts_schedules)
        for contracts_type, contracts_dict in contracts_all.items():
            for contract_name, contract_spec in contracts_dict.items():
                shift_price = df_shifted.loc[contract_spec.get_schedule(), :].mean().iloc[0]
                self.assertAlmostEqual(contract_spec.price, shift_price, delta=10 ** (-10))

    def test_non_coverage_of_shape(self):
        contracts_schedules = {
            "Peak": {
                "Cal23_GasBase": PeakSchedule(dt.datetime(2023, 1, 1), dt.datetime(2024, 1, 1)),
                "Q1/23_GasBase": PeakSchedule(dt.datetime(2023, 1, 1), dt.datetime(2023, 4, 1)),
                "Q2/23_GasBase": PeakSchedule(dt.datetime(2023, 4, 1), dt.datetime(2023, 7, 1)),
                "Q3/23_GasBase": PeakSchedule(dt.datetime(2023, 7, 1), dt.datetime(2023, 10, 1)),
                "Q4/23_GasBase": PeakSchedule(dt.datetime(2023, 10, 1), dt.datetime(2024, 1, 1)),
            },
        }
        contracts = self.__get_contracts_dict(contracts_schedules=contracts_schedules)
        data_dict = {
            "Peak": self.shape_df.loc[SimpleSchedule(dt.datetime(2023, 1, 1), dt.datetime(2024, 1, 1)).get_schedule()],
        }
        result = []
        with self.assertRaises(ValueError):
            for contracts_type, data in data_dict.items():
                pfc_shifter = PFCShifter(shape=data, contracts=list(contracts[contracts_type].values()))
                result.append(pfc_shifter())


class TestPFCShaper(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestPFCShaper, self).__init__(*args, **kwargs)

        self.example_spot_price_data = pd.read_excel(
            "./tests/data/hpfc_test.xlsx", parse_dates=["Date"], date_format="%d.%m.%Y %H:%M", index_col="Date"
        )

    def test_categoricalregression(self):
        parameter_dict = {
            "spot_price_level": 100,
            "peak_price_level": 10,
            "solar_price_level": 8,
            "weekend_price_level": 10,
            "winter_price_level": 20,
            "epsilon_mean": 0,
            "epsilon_var": 5,
        }
        date_range = pd.date_range(start="1/1/2024", end="1/1/2025", freq="h", inclusive="left")
        spot_prices = list(map(lambda x: spot_price_model(x, **parameter_dict), date_range))
        spot_prices = pd.DataFrame(data=spot_prices, index=date_range, columns=["Spot"])

        holiday_calendar = holidays.country_holidays("DE", years=[2024])

        apply_schedule = SimpleSchedule(start=dt.datetime(2025, 1, 1), end=dt.datetime(2026, 1, 1), freq="h")

        pfc_shaper = CategoricalRegression(spot_prices=spot_prices, holiday_calendar=holiday_calendar)
        pfc_fit = pfc_shaper.calibrate()

        self.assertLess(np.sum((spot_prices.values - pfc_fit)) ** 2, 10 ** (-5))

        pfc_rollout = pfc_shaper.apply(apply_schedule=apply_schedule)
        spot_prices_rollout = np.array(list(map(lambda x: spot_price_model(x, **parameter_dict), apply_schedule.get_schedule())))

        self.assertLess(np.sum((spot_prices_rollout / np.mean(spot_prices_rollout) - pfc_rollout.values)) ** 2, 10 ** (-5))

    def test_normalization_with_config(self):
        normalization_config = {"D": 2, "W": 2, "ME": 1}
        holiday_calendar = holidays.country_holidays("DE", years=[2024, 2025, 2026])

        pfc_shaper = CategoricalRegression(
            spot_prices=self.example_spot_price_data, holiday_calendar=holiday_calendar, normalization_config=normalization_config
        )

        apply_schedule = BaseSchedule(start=dt.datetime(2025, 12, 31), end=dt.datetime(2027, 1, 1))

        pfc_spot = pfc_shaper.calibrate()
        pfc_shape = pfc_shaper.apply(apply_schedule=apply_schedule)

        self.assertLess(np.abs(pfc_shape.loc[pfc_shape.index < dt.datetime(2026, 1, 1), :].mean().iloc[0] - 1.0), 10 ** (-5))

        self.assertLess(
            np.abs(pfc_shape.loc[(dt.datetime(2026, 1, 1) <= pfc_shape.index) & (pfc_shape.index < dt.datetime(2026, 1, 2)), :].mean().iloc[0] - 1.0),
            10 ** (-5),
        )

        self.assertLess(
            np.abs(pfc_shape.loc[(dt.datetime(2026, 1, 2) <= pfc_shape.index) & (pfc_shape.index < dt.datetime(2026, 1, 5)), :].mean().iloc[0] - 1.0),
            10 ** (-5),
        )

        self.assertLess(
            np.abs(
                pfc_shape.loc[(dt.datetime(2026, 1, 5) <= pfc_shape.index) & (pfc_shape.index < dt.datetime(2026, 1, 12)), :].mean().iloc[0] - 1.0
            ),
            10 ** (-5),
        )

        self.assertLess(
            np.abs(
                pfc_shape.loc[(dt.datetime(2026, 1, 12) <= pfc_shape.index) & (pfc_shape.index < dt.datetime(2026, 2, 1)), :].mean().iloc[0] - 1.0
            ),
            10 ** (-5),
        )

        self.assertLess(
            np.abs(pfc_shape.loc[(dt.datetime(2026, 2, 1) <= pfc_shape.index) & (pfc_shape.index < dt.datetime(2027, 1, 1)), :].mean().iloc[0] - 1.0),
            10 ** (-5),
        )

    def test_normalization_without_config(self):

        holiday_calendar = holidays.country_holidays("DE", years=[2024, 2025, 2026])

        pfc_shaper = CategoricalRegression(spot_prices=self.example_spot_price_data, holiday_calendar=holiday_calendar)

        apply_schedule = BaseSchedule(start=dt.datetime(2025, 12, 31), end=dt.datetime(2027, 1, 1))

        pfc_spot = pfc_shaper.calibrate()
        pfc_shape = pfc_shaper.apply(apply_schedule=apply_schedule)

        self.assertLess(np.abs(pfc_shape.loc[pfc_shape.index < dt.datetime(2026, 1, 1), :].mean().iloc[0] - 1.0), 10 ** (-5))

        self.assertLess(
            np.abs(pfc_shape.loc[(dt.datetime(2026, 1, 1) <= pfc_shape.index) & (pfc_shape.index < dt.datetime(2027, 1, 1)), :].mean().iloc[0] - 1.0),
            10 ** (-5),
        )


class TestEnergyPriceForwardCurve(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestEnergyPriceForwardCurve, self).__init__(*args, **kwargs)

        self.example_spot_price_data = pd.read_excel(
            "./tests/data/hpfc_test.xlsx", parse_dates=["Date"], date_format="%d.%m.%Y %H:%M", index_col="Date"
        )
        self.parameter_dict = {
            "spot_price_level": 100,
            "peak_price_level": 10,
            "solar_price_level": 8,
            "weekend_price_level": 10,
            "winter_price_level": 20,
            "epsilon_mean": 0,
            "epsilon_var": 5,
        }
        self.date_range = pd.date_range(start="1/1/2023", end="1/1/2025", freq="h", inclusive="left")
        self.spot_prices = list(map(lambda x: spot_price_model(x, **self.parameter_dict), self.date_range))
        self.df = pd.DataFrame(data=self.spot_prices, index=self.date_range, columns=["Spot"])
        base_y = self.df.resample("YE").mean()
        base_y.index = base_y.index.strftime("%Y")

        df_spot = self.df.copy()
        df_spot.index = df_spot.index.strftime("%Y")

        shape = df_spot.divide(base_y, axis="index")
        self.shape_df = pd.DataFrame(data=shape["Spot"].tolist(), index=self.date_range, columns=["shape"])

    def __get_contracts_list(self, contracts_schedules: Dict[str, Dict[str, SimpleSchedule]]) -> Dict[str, Dict[str, EnergyFutureSpecifications]]:
        contracts = defaultdict(dict)
        for contract_type, contracts_dict in contracts_schedules.items():
            for contract_name, schedule in contracts_dict.items():
                tg = schedule.get_schedule()
                price = self.df.loc[tg, :].mean().iloc[0]
                contracts[contract_type][contract_name] = EnergyFutureSpecifications(schedule=schedule, price=price, name=contract_name)
        return list(itertools.chain(*[list(constracts_spec.values()) for constracts_spec in contracts.values()]))

    def test_from_shape(self):
        contracts_schedules = {
            "base": {
                "Cal23_Base": BaseSchedule(dt.datetime(2023, 1, 1), dt.datetime(2024, 1, 1)),
                "Q2/23_Base": BaseSchedule(dt.datetime(2023, 4, 1), dt.datetime(2023, 7, 1)),
            },
            "peak": {
                "Cal23_Peak": PeakSchedule(dt.datetime(2023, 1, 1), dt.datetime(2024, 1, 1)),
                "Cal24_Peak": PeakSchedule(dt.datetime(2024, 1, 1), dt.datetime(2025, 1, 1)),
                "Q1/23_Peak": PeakSchedule(dt.datetime(2023, 1, 1), dt.datetime(2023, 4, 1)),
                "Q2/23_Peak": PeakSchedule(dt.datetime(2023, 4, 1), dt.datetime(2023, 7, 1)),
                "Q3/23_Peak": PeakSchedule(dt.datetime(2023, 7, 1), dt.datetime(2023, 10, 1)),
                "Q4/23_Peak": PeakSchedule(dt.datetime(2023, 10, 1), dt.datetime(2024, 1, 1)),
                "Q2/24_Peak": PeakSchedule(dt.datetime(2024, 4, 1), dt.datetime(2024, 7, 1)),
            },
        }
        contracts_list = self.__get_contracts_list(contracts_schedules=contracts_schedules)
        pfc_obj = EnergyPriceForwardCurve.from_existing_shape(id=None, refdate=None, pfc_shape=self.shape_df, future_contracts=contracts_list)

        pfc = pfc_obj.get_pfc()

        for contract in contracts_list:
            pfc_price = np.mean(pfc.loc[contract.get_schedule(), :])
            self.assertAlmostEqual(pfc_price, contract.get_price(), delta=10 ** (-10))

    def test_from_scratch(self):
        contracts_schedules = {
            "base": {
                "Cal23_Base": SimpleSchedule(dt.datetime(2023, 1, 1), dt.datetime(2024, 1, 1), freq="D"),
                "Q2/23_Base": BaseSchedule(dt.datetime(2023, 4, 1), dt.datetime(2023, 7, 1)),
            },
            "peak": {
                "Cal23_Peak": PeakSchedule(dt.datetime(2023, 1, 1), dt.datetime(2024, 1, 1)),
                "Cal24_Peak": PeakSchedule(dt.datetime(2024, 1, 1), dt.datetime(2025, 1, 1)),
                "Q1/23_Peak": PeakSchedule(dt.datetime(2023, 1, 1), dt.datetime(2023, 4, 1)),
                "Q2/23_Peak": PeakSchedule(dt.datetime(2023, 4, 1), dt.datetime(2023, 7, 1)),
                "Q3/23_Peak": PeakSchedule(dt.datetime(2023, 7, 1), dt.datetime(2023, 10, 1)),
                "Q4/23_Peak": PeakSchedule(dt.datetime(2023, 10, 1), dt.datetime(2024, 1, 1)),
                "Q2/24_Peak": PeakSchedule(dt.datetime(2024, 4, 1), dt.datetime(2024, 7, 1)),
            },
        }
        contracts_list = self.__get_contracts_list(contracts_schedules=contracts_schedules)
        normalization_config = {"D": 2, "W": 2, "ME": 1}
        holiday_calendar = holidays.country_holidays("DE", years=[2024, 2025, 2026])
        apply_schedule = BaseSchedule(start=dt.datetime(2025, 12, 31), end=dt.datetime(2027, 1, 1))

        pfc_shaper = CategoricalRegression(
            spot_prices=self.example_spot_price_data, holiday_calendar=holiday_calendar, normalization_config=normalization_config
        )
        # raise ValueError due to different frequencies in the contracts
        with self.assertRaises(ValueError):
            pfc_obj = EnergyPriceForwardCurve.from_scratch(
                id=None, refdate=None, apply_schedule=apply_schedule, pfc_shaper=pfc_shaper, future_contracts=contracts_list
            )

    def test_existing_pfc(self):
        # create shape without DateTimeIndex
        not_acceptable_shape = pd.DataFrame(data=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
        with self.assertRaises(TypeError):
            pfc_obj = EnergyPriceForwardCurve.from_existing_pfc(id=None, refdate=None, pfc=not_acceptable_shape)

        # create shape which is not a pd.DataFrame
        not_acceptable_shape = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        with self.assertRaises(TypeError):
            pfc_obj = EnergyPriceForwardCurve.from_existing_pfc(id=None, refdate=None, pfc=not_acceptable_shape)

        with self.assertRaises(ValueError):
            EnergyPriceForwardCurve(id=None, refdate=None)


if __name__ == "__main__":
    unittest.main()
