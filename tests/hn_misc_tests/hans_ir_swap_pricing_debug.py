from rivapy.pricing.interest_rate_swap_pricing import InterestRateSwapPricer
from rivapy.instruments.ir_swap_specification import InterestRateSwapSpecification, IrFixedLegSpecification, IrFloatLegSpecification
from rivapy.instruments.notional_structure import NotionalStructure, ConstNotionalStructure, VariableNotionalStructure, ResettingNotionalStructure
from rivapy.pricing.pricing_data import (
    InterestRateSwapFloatLegPricingData_rivapy,
    InterestRateSwapLegPricingData_rivapy,
    InterestRateSwapPricingData_rivapy,
)
from rivapy.pricing.pricing_request import InterestRateSwapPricingRequest


import datetime as dt
from rivapy.marketdata.curves import DiscountCurve
from rivapy.tools.enums import InterpolationType, ExtrapolationType
import math


if __name__ == "__main__":

    # Discount curve - we use these discount factors to get the present values of both the fixed and floating leg as well as

    object_id = "TEST_DC"
    refdatedc = dt.datetime(2017, 1, 1)
    days_to_maturity = [180, 360, 540]
    dates = [refdatedc + dt.timedelta(days=d) for d in days_to_maturity]
    # discount factors from constant rate
    rates = [0.10, 0.105, 0.11]
    df = [math.exp(-r * d / 360) for r, d in zip(rates, days_to_maturity)]
    dc = DiscountCurve(
        id=object_id, refdate=refdatedc, dates=dates, df=df, interpolation=InterpolationType.LINEAR, extrapolation=ExtrapolationType.LINEAR
    )

    # Create the vectors defining the statdates, enddates, paydates and reset dates
    days_to_maturity = [0, 180, 360, 540]
    dates = [dt.datetime(2017, 1, 1) + dt.timedelta(d) for d in days_to_maturity]

    startdates = dates[:-1]
    enddates = dates[1:]
    paydates = enddates
    resetdates = startdates
    refdate = dates[0]

    # notionals = [100.0]

    # fixedleg = IrFixedLegSpecification(0.08, notionals, startdates, enddates, paydates, 'EUR', 'Act360')
    fixed_leg = IrFixedLegSpecification(
        fixed_rate=0.08,
        obj_id="dummy_fixed_leg",
        notional=100.0,
        start_dates=startdates,
        end_dates=enddates,
        pay_dates=paydates,
        currency="EUR",
        day_count_convention="Act360",
    )

    spread = 0.00

    ns = ConstNotionalStructure(100.0)
    # floatleg = IrFloatLegSpecification(notionals, resetdates, startdates, enddates, paydates, 'EUR', 'test_udl',
    #                                   'Act360', spread)

    float_leg = IrFloatLegSpecification(
        obj_id="dummy_float_leg",
        notional=ns,
        reset_dates=resetdates,
        start_dates=startdates,
        end_dates=enddates,
        rate_start_dates=startdates,
        rate_end_dates=enddates,
        pay_dates=paydates,
        currency="EUR",
        udl_id="test_udl_id",
        fixing_id="test_fixing_id",
        day_count_convention="Act360",
        spread=spread,
    )

    maturity_date = refdate + dt.timedelta(600)
    # ir_swap = InterestRateSwapSpecification('TEST_SWAP', 'DBK', 'COLLATERALIZED', 'EUR', paydates[-1], fixedleg, floatleg)
    ir_swap = InterestRateSwapSpecification(
        obj_id="dummy_swap",
        notional=ns,
        issue_date=refdate,
        maturity_date=maturity_date,
        pay_leg=fixed_leg,
        receive_leg=float_leg,
        currency="EUR",
        day_count_convention="Act360",
        issuer="DBK",
        securitization_level="COLLATERALIZED",
    )

    #############
    # pay_leg_pricing_data = InterestRateSwapLegPricingData(
    #     spec=ir_swap.getPayLeg(),
    #     discount_curve=dc,
    #     fx_rate=1.0,
    #     weight=-1.0
    # )
    pay_leg_pricing_data = InterestRateSwapLegPricingData_rivapy(
        spec=ir_swap.get_pay_leg(),
        discount_curve=dc,
        forward_curve=dc,  # TODO check if really needed
        fixing_map=None,  # not needed for now... double check pipeline of why we ned it in the Pricing Data ...
        fx_rate=1.0,
        weight=-1.0,
    )
    # rec_leg_pricing_data = InterestRateSwapFloatLegPricingData(
    #     spec=ir_swap.getReceiveLeg(),
    #     discount_curve=dc,
    #     fixing_curve=dc,
    #     fx_rate=1.0,
    #     weight=1.0,
    # )
    rec_leg_pricing_data = InterestRateSwapFloatLegPricingData_rivapy(
        spec=ir_swap.get_receive_leg(),
        discount_curve=dc,
        forward_curve=dc,  # TODO check if really needed
        fixing_map=None,  # not needed for now... double check pipeline of why we ned it in the Pricing Data ...
        fixing_grace_period=0,  # CHECK as above
        fixing_curve=dc,
        fx_rate=1.0,
        weight=1.0,
    )
    # ir_swap_pricing_data = InterestRateSwapPricingData(
    #     spec=ir_swap,
    #     val_date=refdate,
    #     ccy='EUR',
    #     leg_pricing_data=[pay_leg_pricing_data, rec_leg_pricing_data], # we did it differently, RETHINK
    #     pricing_request=[]
    # )

    pricing_data_all = {}
    pricing_data_all["discount_curve_pay_leg"] = dc
    pricing_data_all["discount_curve_receive_leg"] = dc
    pricing_data_all["fixing_curve_pay_leg"] = dc
    pricing_data_all["fixing_curve_receive_leg"] = dc
    pricing_data_all["fx_fwd_curve_pay_leg"] = dc
    pricing_data_all["fx_fwd_curve_receive_leg"] = dc
    pricing_data_all["pricing_param"] = {"fixing_grace_period": 0}
    pricing_data_all["fixing_map"] = None
    pricing_data_all["fx_pay_leg"] = 1.0
    pricing_data_all["fx_receive_leg"] = 1.0

    ir_swap_pr = InterestRateSwapPricingRequest()

    ir_swap_pricing_data = InterestRateSwapPricingData_rivapy(
        spec=ir_swap,
        val_date=refdate,
        pricing_request=ir_swap_pr,
        pricer="InterestRateSwapPricer",
        ccy="EUR",
        leg_pricing_data=pricing_data_all,  # previousl this heald the pricingData objects for each leg...
    )

    tic = dt.datetime.now()
    pr = ir_swap_pricing_data.price()
    print(f"runtime: {dt.datetime.now() - tic}")
    print(f"Price: {pr}")

    print("------------------------------------------")
    print("Debugging: each leg present value")

    fixed_PV = InterestRateSwapPricer.price_leg(
        refdate, dc, dc, None, fixed_leg, None, 0  # discount  # forward/fixing  # fx_fwd  # leg_spec  # fixing table  # fixing grace peropd
    )
    float_PV = InterestRateSwapPricer.price_leg(
        refdate, dc, dc, None, float_leg, None, 0  # discount  # forward/fixing  # fx_fwd  # leg_spec  # fixing table  # fixing grace peropd
    )

    print(f"float leg pv: {float_PV}")
    print(f"fixed leg pv: {fixed_PV}")
    print(f"fair swap rate: flot/fix: {float_PV/fixed_PV}")

    print("------------------------------------------")
    print("Debugging: each leg cashflow matrix")

    print("generating cashflow table for FIXED leg")
    cashflow_table_fix = InterestRateSwapPricer._populate_cashflows_fix(refdate, fixed_leg, dc, dc, None)

    print("generating cashflow table for FLOAT leg")
    cashflow_table_float = InterestRateSwapPricer._populate_cashflows_float(refdate, float_leg, dc, dc, None, None, 0)
