import unittest
import pandas as pd
import numpy as np
import datetime as dt
from rivapy.tools.scheduler import SimpleSchedule, OffPeakSchedule, PeakSchedule, GasSchedule
from rivapy.marketdata.pfc_shifter import PFCShifter
from rivapy.instruments.energy_futures_specifications import EnergyFutureSpecifications
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
        self.date_range = pd.date_range(start="1/1/2023", end="1/1/2025", freq="H", inclusive="left")
        self.spot_prices = list(map(lambda x: self.__spot_price_model(x, **self.parameter_dict), self.date_range))
        self.df = pd.DataFrame(data=self.spot_prices, index=self.date_range, columns=["Spot"])
        base_y = self.df.resample("Y").mean()
        base_y.index = base_y.index.strftime("%Y")

        df_spot = self.df.copy()
        df_spot.index = df_spot.index.strftime("%Y")

        shape = df_spot.divide(base_y, axis="index")
        self.shape_df = pd.DataFrame(data=shape["Spot"].tolist(), index=self.date_range, columns=["shape"])

    def __spot_price_model(
        self,
        timestamp: dt.datetime,
        spot_price_level: float,
        peak_price_level: float,
        solar_price_level: float,
        weekend_price_level: float,
        winter_price_level: float,
        epsilon_mean: float = 0,
        epsilon_var: float = 1,
    ):
        np.random.seed(42)
        spot_price = spot_price_level
        if (timestamp.hour >= 8 and timestamp.hour < 11) or (timestamp.hour >= 16 and timestamp.hour <= 20):
            spot_price += peak_price_level
        elif timestamp.hour >= 11 and timestamp.hour < 16:
            spot_price -= solar_price_level

        if timestamp.weekday() >= 5:
            spot_price -= weekend_price_level

        if timestamp.month in {1, 2, 3, 10, 11, 12}:
            spot_price += winter_price_level

        spot_price += np.random.normal(loc=epsilon_mean, scale=np.sqrt(epsilon_var))
        return spot_price

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
                    freq="1H",
                    hours=[0, 1, 2, 3, 4, 5, 6, 7, 20, 21, 22, 23],
                    ignore_hours_for_weekdays=[5, 6],
                ).get_schedule()
            ],
            "peak": self.shape_df.loc[
                SimpleSchedule(
                    dt.datetime(2023, 1, 1),
                    dt.datetime(2025, 1, 1),
                    freq="1H",
                    hours=[8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                    weekdays=[0, 1, 2, 3, 4],
                ).get_schedule()
            ],
        }
        result = []
        for contracts_type, data in data_dict.items():
            pfc_shifter = PFCShifter(shape=data, contracts=contracts[contracts_type])
            result.append(pfc_shifter())

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
            "peak": self.shape_df.loc[
                SimpleSchedule(
                    dt.datetime(2023, 1, 1),
                    dt.datetime(2025, 1, 1),
                    freq="1H",
                    hours=[8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                    weekdays=[0, 1, 2, 3, 4],
                ).get_schedule()
            ],
        }
        result = []
        for contracts_type, data in data_dict.items():
            pfc_shifter = PFCShifter(shape=data, contracts=contracts[contracts_type])
            result.append(pfc_shifter())

        df_shifted = pd.concat(result, axis=0)
        df_shifted.sort_index(inplace=True, ascending=True)

        contracts_all = self.__get_contracts_dict(contracts_schedules=contracts_schedules)
        for contracts_type, contracts_dict in contracts_all.items():
            for contract_name, contract_spec in contracts_dict.items():
                shift_price = df_shifted.loc[contract_spec.get_schedule(), :].mean().iloc[0]
                self.assertAlmostEqual(contract_spec.price, shift_price, delta=10 ** (-10))


if __name__ == "__main__":
    unittest.main()
