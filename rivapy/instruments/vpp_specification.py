from typing import Dict, Union
from rivapy.tools.interfaces import FactoryObject, BaseDatedCurve
import datetime as dt

class VPPSpecification(FactoryObject):
    class Restrictions(FactoryObject):
        def __init__(self, capa_min: Union[float,BaseDatedCurve], capa_max: float, ramping, min_runtime: int, min_standstill: int, 
                     max_starts_day: int, no_load_heatconsumption: float,
                     increase_heat_consumption: float, emission_factor_fuel: float):
            self.capa_min = capa_min
            self.capa_max = capa_max
            self.ramping = ramping
            self.min_runtime = min_runtime
            self.min_standstill = min_standstill
            self.max_starts_day = max_starts_day
            self.no_load_heatconsumption = no_load_heatconsumption
            self.increase_heat_consumption = increase_heat_consumption
            self.emission_factor_fuel = emission_factor_fuel

    class Costs(FactoryObject):
        def __init__(self, logistic_fuel: BaseDatedCurve, 
                     operation_maintenance: BaseDatedCurve, 
                     standstill_costs: BaseDatedCurve, 
                     other_costs:Dict[str,BaseDatedCurve]=None):
            self.logistic_fuel = logistic_fuel
            self.operation_maintenance = operation_maintenance
            self.standstill_costs = standstill_costs
            self.other_costs = other_costs

    def __init__(self,
                 delivery_granularity: dt.timedelta,
                 start_delivery: dt.datetime,
                end_delivery: dt.datetime,
                initial_volume: float,
                initial_offtime: int,
                restrictions: VPPSpecification.Restrictions,
                costs: VPPRestrictions.Costs,
                fuel_consumption: BaseDatedCurve,
                nomination_lead_time: int,
				id:str = None):
        self.delivery_granularity = delivery_granularity
        self.start_delivery = start_delivery
        self.end_delivery = end_delivery
        self.initial_volume = initial_volume
        self.initial_offtime = initial_offtime
        self.restrictions = restrictions
        self.costs = costs
        self.fuel_consumption = fuel_consumption
        self.nomination_lead_time = nomination_lead_time
        self.id = id
