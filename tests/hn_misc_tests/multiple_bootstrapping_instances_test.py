# imports
import datetime as dt
from dateutil.relativedelta import relativedelta
import pandas as pd
import matplotlib.pyplot as plt
from rivapy.marketdata.bootstrapping import bootstrap_curve, get_quote
from rivapy.instruments.deposit_specifications import DepositSpecification
from rivapy.instruments.fra_specifications import ForwardRateAgreementSpecification
from rivapy.instruments.ir_swap_specification import (
    InterestRateSwapSpecification,
    IrFixedLegSpecification,
    IrFloatLegSpecification,
    IrOISLegSpecification,
)
from rivapy.instruments.components import ConstNotionalStructure
from rivapy.tools.enums import DayCounterType, InterpolationType, ExtrapolationType

import matplotlib.pyplot as plt
import math
from rivapy.tools.datetools import DayCounter
from rivapy.tools._validators import print_member_values
from rivapy.pricing.deposit_pricing import DepositPricer
from rivapy.pricing.interest_rate_swap_pricing import InterestRateSwapPricer
from rivapy.marketdata.curves import DiscountCurve


if __name__ == "__main__":

    ##########################################
    # setting up depoist
    # calculation date
    ref_date = dt.datetime(2019, 8, 31)  # refdate = dt.datetime(2017, 8, 31)

    # start date of the accrual period with spot lag equal to 2 days
    start_date = ref_date + dt.timedelta(days=2)

    # end date of the accrual period is 1 day after startdate
    end_date = start_date + dt.timedelta(days=1)

    # specification of the deposit
    ccy = "EUR"
    dcc = "Act365Fixed"  # "Act360"normally a bond is ACT360, we keep consistency with the swaps
    rate = 0.01
    notional = 100.0
    deposit = DepositSpecification(
        obj_id="OVERNIGHT_DEPOSIT",
        issuer="dummy_issuer",
        currency=ccy,
        fixing_date=ref_date,
        start_date=start_date,
        maturity_date=end_date,
        notional=notional,
        rate=rate,
        day_count_convention=dcc,
    )

    # check dates, deposit start date cannot be the same as  maturity date
    print(f"ref date: {ref_date}")
    print(f"start date: {start_date} with spot lag 2 days")
    print(f"end date: {end_date}")
    print(f"adjusted start date: {deposit.start_date}")
    print(f"adjusted end date: {deposit.maturity_date}")

    print_member_values(deposit)

    # Note that a fixed rate is given for the specification as it is required. However, for the creation of the bootrstrapped curve, the market quotes are used

    # Using same reference dance, we derive more deposit specifications to cover more maturities
    # consider using dt.relativedelta(months=...)

    end_date_deposits = [
        start_date + dt.timedelta(days=1),
        start_date + dt.timedelta(days=7),
        start_date + dt.timedelta(days=30),
        start_date + dt.timedelta(days=60),
        start_date + dt.timedelta(days=90),
        start_date + dt.timedelta(days=181),  # 180 - error due to BCC roll oveer mismatch between start and end date... look into! #TODO
        start_date + dt.timedelta(days=270),
    ]

    # Provide a list of market quotes from which to derive the discount curve
    quotes_deposits = [0.025, 0.028, 0.0283, 0.029, 0.0305, 0.0315, 0.0348]

    # List where the multiple deposit specfications are stored
    multiple_deposits = []

    for i in range(len(quotes_deposits)):
        print(i)
        temp_deposit = DepositSpecification(
            obj_id="DEPOSIT_" + str(i + 1),
            issuer="dummy_issuer",
            currency=ccy,
            fixing_date=ref_date,
            start_date=start_date,
            maturity_date=end_date_deposits[i],
            notional=notional,
            rate=quotes_deposits[i],
            day_count_convention=dcc,
        )
        multiple_deposits.append(temp_deposit)

    ################################################################################
    # SETTING UP SWAPS
    # ref_date is the same as was in deposits

    # 1Y maturity 3M swap, i.e. the floating leg is reset every 3M
    # start dates of the accrual periods corresponding to the tenor of the underlying index (3 months). The spot lag is set to 0.
    start_dates = [ref_date + relativedelta(months=3 * i) for i in range(4)]

    # reset dates are equal to start dates if spot lag is 0.
    reset_dates = start_dates

    # the end dates of the accral periods
    end_dates = [x + relativedelta(months=3) for x in start_dates]

    # the actual payment dates of the cashflows may differ from the end of the accrual period (e.g. OIS).
    # in the standard case these two sets of dates coincide
    pay_dates = end_dates

    print(end_dates)
    ns = ConstNotionalStructure(100.0)
    spread = 0.00

    # # definition of the floating leg
    float_leg = IrFloatLegSpecification(
        obj_id="dummy_float_leg",
        notional=ns,
        reset_dates=reset_dates,
        start_dates=start_dates,
        end_dates=end_dates,
        rate_start_dates=start_dates,
        rate_end_dates=end_dates,
        pay_dates=pay_dates,
        currency="EUR",
        udl_id="test_udl_id",
        fixing_id="test_fixing_id",
        day_count_convention="Act365Fixed",
        spread=spread,
    )

    # # definition of the fixed leg
    # Note that a fixed rate is given for the specification as it is required.
    # However, for the creation of the bootrstrapped curve, the market quotes are used as the target swap par rate
    fixed_leg = IrFixedLegSpecification(
        fixed_rate=0.01,
        obj_id="dummy_fixed_leg",
        notional=100.0,
        start_dates=start_dates,
        end_dates=end_dates,
        pay_dates=pay_dates,
        currency="EUR",
        day_count_convention="Act365Fixed",
    )

    # # definition of the IR swap
    ir_swap = InterestRateSwapSpecification(
        obj_id="3M_SWAP",
        notional=ns,
        issue_date=ref_date,
        maturity_date=pay_dates[-1],
        pay_leg=fixed_leg,
        receive_leg=float_leg,
        currency="EUR",
        day_count_convention="Act365Fixed",
        issuer="dummy_issuer",
        securitization_level="COLLATERALIZED",
    )

    ##########################################
    # 2Y maturity 3M swap
    start_dates2 = [ref_date + relativedelta(months=3 * i) for i in range(4 * 2)]
    reset_dates2 = start_dates2
    end_dates2 = [x + relativedelta(months=3) for x in start_dates2]
    pay_dates2 = end_dates2

    ns = ConstNotionalStructure(100.0)
    spread = 0.00

    # # definition of the floating leg
    float_leg2 = IrFloatLegSpecification(
        obj_id="dummy_float_leg2",
        notional=ns,
        reset_dates=reset_dates2,
        start_dates=start_dates2,
        end_dates=end_dates2,
        rate_start_dates=start_dates2,
        rate_end_dates=end_dates2,
        pay_dates=pay_dates2,
        currency="EUR",
        udl_id="test_udl_id",
        fixing_id="test_fixing_id",
        day_count_convention="Act365Fixed",
        spread=spread,
    )

    # # definition of the fixed leg
    fixed_leg2 = IrFixedLegSpecification(
        fixed_rate=0.01,
        obj_id="dummy_fixed_leg2",
        notional=100.0,
        start_dates=start_dates2,
        end_dates=end_dates2,
        pay_dates=pay_dates2,
        currency="EUR",
        day_count_convention="Act365Fixed",
    )

    # # definition of the IR swap
    ir_swap2 = InterestRateSwapSpecification(
        obj_id="3M_SWAP2",
        notional=ns,
        issue_date=ref_date,
        maturity_date=pay_dates2[-1],
        pay_leg=fixed_leg2,
        receive_leg=float_leg2,
        currency="EUR",
        day_count_convention="Act365Fixed",
        issuer="dummy_issuer",
        securitization_level="COLLATERALIZED",
    )

    ##########################################
    # 3Y maturity 3M swap

    start_dates3 = [ref_date + relativedelta(months=3 * i) for i in range(4 * 3)]
    reset_dates3 = start_dates3
    end_dates3 = [x + relativedelta(months=3) for x in start_dates3]
    pay_dates3 = end_dates3
    ns = ConstNotionalStructure(100.0)
    spread = 0.00

    # # definition of the floating leg
    float_leg3 = IrFloatLegSpecification(
        obj_id="dummy_float_leg3",
        notional=ns,
        reset_dates=reset_dates3,
        start_dates=start_dates3,
        end_dates=end_dates3,
        rate_start_dates=start_dates3,
        rate_end_dates=end_dates3,
        pay_dates=pay_dates3,
        currency="EUR",
        udl_id="test_udl_id",
        fixing_id="test_fixing_id",
        day_count_convention="Act365Fixed",
        spread=spread,
    )

    # # definition of the fixed leg
    fixed_leg3 = IrFixedLegSpecification(
        fixed_rate=0.01,
        obj_id="dummy_fixed_leg3",
        notional=100.0,
        start_dates=start_dates3,
        end_dates=end_dates3,
        pay_dates=pay_dates3,
        currency="EUR",
        day_count_convention="Act365Fixed",
    )

    # # definition of the IR swap
    ir_swap3 = InterestRateSwapSpecification(
        obj_id="3M_SWAP3",
        notional=ns,
        issue_date=ref_date,
        maturity_date=pay_dates3[-1],
        pay_leg=fixed_leg3,
        receive_leg=float_leg3,
        currency="EUR",
        day_count_convention="Act365Fixed",
        issuer="dummy_issuer",
        securitization_level="COLLATERALIZED",
    )

    ##########################################
    # organising the swaps and their taarget market quotes
    multiple_swaps = [ir_swap, ir_swap2, ir_swap3]
    quotes_swaps = [0.05, 0.06, 0.07]

    # combine the multiple instruments to be given to the bootstrapper
    instruments_both = multiple_deposits + multiple_swaps
    quotes_both = quotes_deposits + quotes_swaps

    boot_curve = bootstrap_curve(
        ref_date,
        "bootstrapped_DC",
        DayCounterType.Act365Fixed,
        instruments_both,
        quotes_both,
        interpolation_type=InterpolationType.LINEAR_LOG,
        extrapolation_type=ExtrapolationType.LINEAR_LOG,
    )

    ##########################################
    # PLotting results
    plt.figure(3)
    dates_final = boot_curve.get_dates()
    df_final = boot_curve.get_df()

    # calculated df from dates
    dates_new = [dates_final[0]]
    days = 10  # create a smoother curve by adding dates in between
    for i in range(1, len(dates_final)):
        while dates_new[-1] + dt.timedelta(days=days) < dates_final[i]:
            dates_new.append(dates_new[-1] + dt.timedelta(days=days))
        dates_new.append(dates_final[i])

    print(dates_new)
    values = [boot_curve.value(boot_curve.refdate, d) for d in dates_new]
    print(values)

    plt.plot(dates_final, df_final, marker="^", label="bootstrapped")
    plt.plot(dates_new, values, label="interpolated")

    plt.xlabel("year")
    plt.ylabel("DF")
    plt.legend()

    # continuous compounding
    plt.figure(4)
    dcc = DayCounter(boot_curve.daycounter)
    zero_final = []
    for i in range(1, len(df_final)):
        delta_t = dcc.yf(boot_curve.refdate, dates_final[i])  # float((dates_new[i] - estr.refdate).days) / 365.0
        zero_final.append(-math.log(df_final[i]) / delta_t)
    zero_final.insert(0, zero_final[0])  # adding first entry to account for division by zero for dt

    zero_interp = []
    for i in range(1, len(values)):
        delta_t = dcc.yf(boot_curve.refdate, dates_new[i])  # float((dates_new[i] - estr.refdate).days) / 365.0
        zero_interp.append(-math.log(values[i]) / delta_t)
    zero_interp.insert(0, zero_interp[0])

    plt.plot(dates_final, zero_final, marker="^", label="bootstrapped")
    # plt.scatter(end_date_deposits, quotes_deposits)
    plt.plot(dates_new, zero_interp, label="interpolated")

    plt.xlabel("year")
    plt.ylabel("zero rates")
    plt.legend()
    plt.show()

    ##########################################
    # As a sort of calibration, the bootstrapped curve needs to reproduce the market quotes exactly when repricing the instrumnts using this derived curve

    model_quotes = []
    pricing_params = {"fixing_grace_period": 0.0, "set_rate": True, "desired_rate": 1.0}
    curves_dict = {"discount_curve": boot_curve, "fixing_curve": boot_curve}

    for i in range(len(instruments_both)):

        model_quote = get_quote(ref_date, instruments_both[i], curves_dict)
        # print(model_quote)
        model_quotes.append(model_quote)

    # we compare the percent difference between the target market quotes and the rates given from the bootstrapped curve
    for i in range(len(quotes_both)):

        per_diff = (model_quotes[i] - quotes_both[i]) / quotes_both[i] * 100
        print(f"{model_quotes[i]} - {quotes_both[i]}")
        print(f"i:{i} date:{dates_final[i+1]}  percent diff: {per_diff}")

    ################################################################################################################
    # CREATE OIS discount curve
    # This is the SECOND instance in the same code of bootstrapping a curve

    # Specify a couple of OIS swaps
    # Note that here we represent them as a typical IR swap, except that the float leg uses an OIS specification instead of FLoatIRS specification.
    # THe fixed IRS specification is used the same

    # We will use OIS based on a different underlying tenors
    # will assume 1 year maturity for all ...

    ##########################################
    # calculation date
    ref_date = dt.datetime(2019, 8, 31)

    # 3M maturity 3M underlying tenor swap, i.e. the floating leg is reset every 3M
    # since it is OIS

    # start dates of the accrual periods corresponding to the tenor of the underlying index (3 months). The spot lag is set to 0.
    start_dates = [ref_date + relativedelta(months=3 * i) for i in range(1)]

    # the end dates of the accrual periods
    end_dates = [x + relativedelta(months=3) for x in start_dates]

    # the actual payment dates of the cashflows may differ from the end of the accrual period (e.g. OIS).
    # in the standard case these two sets of dates coincide
    # pay_dates = end_dates

    print(end_dates)
    ns = ConstNotionalStructure(100.0)
    spread = 0.00

    # the difference here is that rate_date arrays are excpected to be 2 dimensional, i.e. keep track of the daily resetting per accrual period

    res = IrOISLegSpecification.ois_scheduler_2D(start_dates, end_dates)

    daily_rate_start_dates = res[0]  # 2D list: coupon i -> list of daily starts
    daily_rate_end_dates = res[1]  # 2D list: coupon i -> list of daily ends
    daily_rate_reset_dates = res[2]  # 2D list: coupon i -> list of reset dates
    pay_dates = res[3]

    float_leg = IrOISLegSpecification(
        obj_id="dummy_float_leg",
        notional=ns,
        rate_reset_dates=daily_rate_reset_dates,
        start_dates=start_dates,
        end_dates=end_dates,
        rate_start_dates=daily_rate_start_dates,
        rate_end_dates=daily_rate_end_dates,
        pay_dates=pay_dates,
        currency="EUR",
        udl_id="test_udl_id",
        fixing_id="test_fixing_id",
        day_count_convention="Act365Fixed",
        rate_day_count_convention="Act365Fixed",
        spread=spread,
    )

    # # definition of the fixed leg
    # Note that a fixed rate is given for the specification as it is required.
    # However, for the creation of the bootrstrapped curve, the market quotes are used as the target swap par rate
    fixed_leg = IrFixedLegSpecification(
        fixed_rate=0.01,
        obj_id="dummy_fixed_leg",
        notional=100.0,
        start_dates=start_dates,
        end_dates=end_dates,
        pay_dates=pay_dates,
        currency="EUR",
        day_count_convention="Act365Fixed",
    )

    # # definition of the IR swap
    ois_swap_3M = InterestRateSwapSpecification(
        obj_id="3M_SWAP",
        notional=ns,
        issue_date=ref_date,
        maturity_date=pay_dates[-1],
        pay_leg=fixed_leg,
        receive_leg=float_leg,
        currency="EUR",
        day_count_convention="Act365Fixed",
        issuer="dummy_issuer",
        securitization_level="COLLATERALIZED",
    )

    ##########################################
    # 6M Maturity, 6M tenor

    start_dates = [ref_date + relativedelta(months=6 * i) for i in range(1)]

    # the end dates of the accrual periods
    end_dates = [x + relativedelta(months=6) for x in start_dates]

    # the actual payment dates of the cashflows may differ from the end of the accrual period (e.g. OIS).
    # in the standard case these two sets of dates coincide
    # pay_dates = end_dates

    print(end_dates)
    ns = ConstNotionalStructure(100.0)
    spread = 0.00

    # the difference here is that rate_date arrays are excpected to be 2 dimensional, i.e. keep track of the daily resetting per accrual period

    res = IrOISLegSpecification.ois_scheduler_2D(start_dates, end_dates)

    daily_rate_start_dates = res[0]  # 2D list: coupon i -> list of daily starts
    daily_rate_end_dates = res[1]  # 2D list: coupon i -> list of daily ends
    daily_rate_reset_dates = res[2]  # 2D list: coupon i -> list of reset dates
    pay_dates = res[3]

    float_leg = IrOISLegSpecification(
        obj_id="dummy_float_leg",
        notional=ns,
        rate_reset_dates=daily_rate_reset_dates,
        start_dates=start_dates,
        end_dates=end_dates,
        rate_start_dates=daily_rate_start_dates,
        rate_end_dates=daily_rate_end_dates,
        pay_dates=pay_dates,
        currency="EUR",
        udl_id="test_udl_id",
        fixing_id="test_fixing_id",
        day_count_convention="Act365Fixed",
        rate_day_count_convention="Act365Fixed",
        spread=spread,
    )

    # # definition of the fixed leg
    # Note that a fixed rate is given for the specification as it is required.
    # However, for the creation of the bootrstrapped curve, the market quotes are used as the target swap par rate
    fixed_leg = IrFixedLegSpecification(
        fixed_rate=0.01,
        obj_id="dummy_fixed_leg",
        notional=100.0,
        start_dates=start_dates,
        end_dates=end_dates,
        pay_dates=pay_dates,
        currency="EUR",
        day_count_convention="Act365Fixed",
    )

    # # definition of the IR swap
    ois_swap_6M = InterestRateSwapSpecification(
        obj_id="6M_SWAP",
        notional=ns,
        issue_date=ref_date,
        maturity_date=pay_dates[-1],
        pay_leg=fixed_leg,
        receive_leg=float_leg,
        currency="EUR",
        day_count_convention="Act365Fixed",
        issuer="dummy_issuer",
        securitization_level="COLLATERALIZED",
    )

    ##########################################
    # 9M Maturity, 9M tenor
    # we have to make sure teh mat date is correrct

    start_dates = [ref_date + relativedelta(months=9 * i) for i in range(1)]

    # the end dates of the accrual periods
    end_dates = [x + relativedelta(months=9) for x in start_dates]

    # maturity_date = start_dates[0] + relativedelta(months=12)

    # the actual payment dates of the cashflows may differ from the end of the accrual period (e.g. OIS).
    # in the standard case these two sets of dates coincide
    # pay_dates = end_dates

    print(end_dates)
    ns = ConstNotionalStructure(100.0)
    spread = 0.00

    # the difference here is that rate_date arrays are excpected to be 2 dimensional, i.e. keep track of the daily resetting per accrual period

    res = IrOISLegSpecification.ois_scheduler_2D(start_dates, end_dates)

    daily_rate_start_dates = res[0]  # 2D list: coupon i -> list of daily starts
    daily_rate_end_dates = res[1]  # 2D list: coupon i -> list of daily ends
    daily_rate_reset_dates = res[2]  # 2D list: coupon i -> list of reset dates
    pay_dates = res[3]

    float_leg = IrOISLegSpecification(
        obj_id="dummy_float_leg",
        notional=ns,
        rate_reset_dates=daily_rate_reset_dates,
        start_dates=start_dates,
        end_dates=end_dates,
        rate_start_dates=daily_rate_start_dates,
        rate_end_dates=daily_rate_end_dates,
        pay_dates=pay_dates,
        currency="EUR",
        udl_id="test_udl_id",
        fixing_id="test_fixing_id",
        day_count_convention="Act365Fixed",
        rate_day_count_convention="Act365Fixed",
        spread=spread,
    )

    # # definition of the fixed leg
    # Note that a fixed rate is given for the specification as it is required.
    # However, for the creation of the bootrstrapped curve, the market quotes are used as the target swap par rate
    fixed_leg = IrFixedLegSpecification(
        fixed_rate=0.01,
        obj_id="dummy_fixed_leg",
        notional=100.0,
        start_dates=start_dates,
        end_dates=end_dates,
        pay_dates=pay_dates,
        currency="EUR",
        day_count_convention="Act365Fixed",
    )

    # # definition of the IR swap
    ois_swap_9M = InterestRateSwapSpecification(
        obj_id="39_SWAP",
        notional=ns,
        issue_date=ref_date,
        maturity_date=pay_dates[-1],
        pay_leg=fixed_leg,
        receive_leg=float_leg,
        currency="EUR",
        day_count_convention="Act365Fixed",
        issuer="dummy_issuer",
        securitization_level="COLLATERALIZED",
    )

    ##########################################
    # 1M Maturity, 1 M tenor

    start_dates = [ref_date + relativedelta(months=1 * i) for i in range(1)]

    # the end dates of the accrual periods
    end_dates = [x + relativedelta(months=1) for x in start_dates]

    # the actual payment dates of the cashflows may differ from the end of the accrual period (e.g. OIS).
    # in the standard case these two sets of dates coincide
    # pay_dates = end_dates

    print(end_dates)
    ns = ConstNotionalStructure(100.0)
    spread = 0.00

    # the difference here is that rate_date arrays are excpected to be 2 dimensional, i.e. keep track of the daily resetting per accrual period

    res = IrOISLegSpecification.ois_scheduler_2D(start_dates, end_dates)

    daily_rate_start_dates = res[0]  # 2D list: coupon i -> list of daily starts
    daily_rate_end_dates = res[1]  # 2D list: coupon i -> list of daily ends
    daily_rate_reset_dates = res[2]  # 2D list: coupon i -> list of reset dates
    pay_dates = res[3]

    float_leg = IrOISLegSpecification(
        obj_id="dummy_float_leg",
        notional=ns,
        rate_reset_dates=daily_rate_reset_dates,
        start_dates=start_dates,
        end_dates=end_dates,
        rate_start_dates=daily_rate_start_dates,
        rate_end_dates=daily_rate_end_dates,
        pay_dates=pay_dates,
        currency="EUR",
        udl_id="test_udl_id",
        fixing_id="test_fixing_id",
        day_count_convention="Act365Fixed",
        rate_day_count_convention="Act365Fixed",
        spread=spread,
    )

    # # definition of the fixed leg
    # Note that a fixed rate is given for the specification as it is required.
    # However, for the creation of the bootrstrapped curve, the market quotes are used as the target swap par rate
    fixed_leg = IrFixedLegSpecification(
        fixed_rate=0.01,
        obj_id="dummy_fixed_leg",
        notional=100.0,
        start_dates=start_dates,
        end_dates=end_dates,
        pay_dates=pay_dates,
        currency="EUR",
        day_count_convention="Act365Fixed",
    )

    # # definition of the IR swap
    ois_swap_1M = InterestRateSwapSpecification(
        obj_id="1M_SWAP",
        notional=ns,
        issue_date=ref_date,
        maturity_date=pay_dates[-1],
        pay_leg=fixed_leg,
        receive_leg=float_leg,
        currency="EUR",
        day_count_convention="Act365Fixed",
        issuer="dummy_issuer",
        securitization_level="COLLATERALIZED",
    )

    ##########################################
    # Different deposit instrument for testing
    # start date of the accrual period with spot lag equal to 2 days
    start_date = ref_date + dt.timedelta(days=2)

    # end date of the accrual period is 1 day after startdate
    end_date = start_date + dt.timedelta(days=1)

    # specification of the deposit
    ccy = "EUR"
    dcc = "Act365Fixed"  # "Act360"normally a bond is ACT360, we keep consistency with the swaps
    rate = -0.00345
    notional = 100.0
    deposit_estr = DepositSpecification(
        obj_id="OVERNIGHT_DEPOSIT",
        issuer="dummy_issuer",
        currency=ccy,
        fixing_date=ref_date,
        start_date=start_date,
        maturity_date=end_date,
        notional=notional,
        rate=rate,
        day_count_convention=dcc,
    )

    #########################################
    # Bootstrapping for the OIS curve
    # instruments_ois = [deposit_estr, ois_swap_1M,ois_swap_3M,ois_swap_6M,ois_swap_9M]
    # quotes_ois = [-0.00345, -0.00358, -0.00358, -0.00358, -0.00357] #taken from the .csv as example

    # instruments_ois = [ois_swap_1M, ois_swap_3M, ois_swap_6M, ois_swap_9M] #worked seperately from the first instance
    # quotes_ois = [-0.00358, -0.00358, -0.00358, -0.00357]  # taken from the .csv as example

    instruments_ois = [deposit_estr, ois_swap_1M]
    quotes_ois = [-0.00345, -0.00358]  # taken from the .csv as example

    # instruments_ois = [deposit_estr, ir_swap]
    # quotes_ois = [-0.00345, 0.05] #taken from the .csv as example

    boot_curve_ois = bootstrap_curve(
        ref_date,
        "bootstrapped_ois_DC",
        DayCounterType.Act365Fixed,
        instruments_ois,
        quotes_ois,
        interpolation_type=InterpolationType.LINEAR_LOG,
        extrapolation_type=ExtrapolationType.LINEAR_LOG,
    )  # NO CURVES DICT WAS PASSED, why is it not EMPTY?
    # def bootstrap_curve(
    #     ref_date: _Union[date, datetime],
    #     curve_id: str,
    #     day_count_convention: _Union[DayCounterType, str],
    #     instruments: _List,
    #     quotes: _List,
    #     curves: dict = None,
    #     # discount_curve: DiscountCurve = None,
    #     # basis_curve: DiscountCurve = None,
    #     interpolation_type: InterpolationType = InterpolationType.LINEAR,
    #     extrapolation_type: ExtrapolationType = ExtrapolationType.LINEAR,
    #     tolerance: float = 1.0e-8,
    #     max_iterations: int = 10000,
    # ) -> DiscountCurve:
    ##########################################
    # PLotting results
    # PLotting Discount Curve

    plt.figure(5)
    dates_final = boot_curve_ois.get_dates()
    df_final = boot_curve_ois.get_df()

    # calculated df from dates
    dates_new = [dates_final[0]]
    days = 10  # create a smoother curve by adding dates in between
    for i in range(1, len(dates_final)):
        while dates_new[-1] + dt.timedelta(days=days) < dates_final[i]:
            dates_new.append(dates_new[-1] + dt.timedelta(days=days))
        dates_new.append(dates_final[i])

    print(dates_new)
    values = [boot_curve_ois.value(boot_curve_ois.refdate, d) for d in dates_new]
    print(values)

    plt.plot(dates_final, df_final, marker="^", label="bootstrapped")
    plt.plot(dates_new, values, label="interpolated")

    plt.xlabel("year")
    plt.ylabel("DF")
    plt.legend()

    # continuous compounding
    plt.figure(6)
    dcc = DayCounter(boot_curve_ois.daycounter)
    zero_final = []
    for i in range(1, len(df_final)):
        delta_t = dcc.yf(boot_curve_ois.refdate, dates_final[i])  # float((dates_new[i] - estr.refdate).days) / 365.0
        zero_final.append(-math.log(df_final[i]) / delta_t)
    zero_final.insert(0, zero_final[0])  # adding first entry to account for division by zero for dt

    zero_interp = []
    for i in range(1, len(values)):
        delta_t = dcc.yf(boot_curve_ois.refdate, dates_new[i])  # float((dates_new[i] - estr.refdate).days) / 365.0
        zero_interp.append(-math.log(values[i]) / delta_t)
    zero_interp.insert(0, zero_interp[0])

    plt.plot(dates_final, zero_final, marker="^", label="bootstrapped")
    # plt.scatter(end_date_deposits, quotes_deposits)
    plt.plot(dates_new, zero_interp, label="interpolated")

    plt.xlabel("year")
    plt.ylabel("zero rates")
    plt.legend()
    plt.show()
