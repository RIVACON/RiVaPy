# imports
import datetime as dt
from dateutil.relativedelta import relativedelta
import pandas as pd
import matplotlib.pyplot as plt
from rivapy.marketdata.bootstrapping import bootstrap_curve
from rivapy.instruments.deposit_specifications import DepositSpecification
from rivapy.instruments.fra_specifications import ForwardRateAgreementSpecification
from rivapy.instruments.ir_swap_specification import (
    InterestRateSwapSpecification,
    IrFixedLegSpecification,
    IrFloatLegSpecification,
    IrOISLegSpecification,
)
from rivapy.instruments.components import ConstNotionalStructure
from rivapy.tools.enums import DayCounterType

import matplotlib.pyplot as plt
import math
from rivapy.tools.datetools import DayCounter

# from rivapy.pricing._logger import logger
from rivapy.pricing.deposit_pricing import DepositPricer
from rivapy.pricing.interest_rate_swap_pricing import InterestRateSwapPricer
from rivapy.tools.enums import InterpolationType, ExtrapolationType


def temp_ois_scheduler(start_dates: list, end_dates: list):

    # CONSIDER USING A SCHEDULER FUNCTION ONCE IT IS FINISHED
    daily_rate_start_dates = []  # 2D list: coupon i -> list of daily starts
    daily_rate_end_dates = []  # 2D list: coupon i -> list of daily ends
    daily_rate_reset_dates = []  # 2D list: coupon i -> list of reset dates
    pay_dates = []  # 1D list: one pay date per coupon

    for i in range(len(start_dates)):

        # for this test we keep it simple and ignore conventions e.g. business day or so. i.e just take every day
        num_days = (end_dates[i] - start_dates[i]).days
        daily_schedule = [start_dates[i] + dt.timedelta(days=j) for j in range(num_days)]

        # Build start/end date pairs for accrual periods
        starts = daily_schedule[:-1]  # all except last
        ends = daily_schedule[1:]  # all except first

        daily_rate_start_dates.append(starts)
        daily_rate_end_dates.append(ends)

        # 4. Compute reset dates (fixing lag applied to each start)
        # resets = [add_business_days(start, fixingLag, rateHolidays)
        #           for start in starts]
        # assume simple case reset date is the same as start date
        resets = starts  # reset dates are equal to start dates if spot lag is 0.
        daily_rate_reset_dates.append(resets)

        # Compute payment date for the coupon
        # pay_date = add_business_days(end_dates[i], payLag, holidays)
        # assume simple case, pay date is end date
        pay_date = end_dates[i]
        pay_dates.append(pay_date)

    return [daily_rate_start_dates, daily_rate_end_dates, daily_rate_reset_dates, pay_dates]


if __name__ == "__main__":

    ##########################################
    # setting up depoist
    # calculation date
    ref_date = dt.datetime(2019, 8, 31)

    # start date of the accrual period with spot lag equal to 2 days
    start_date = ref_date + dt.timedelta(days=2)

    # end date of the accrual period is 1 day after startdate
    end_date = start_date + dt.timedelta(days=1)

    # specification of the deposit
    # deposit = pyvacon.finance.specification.DepositSpecification(
    #     'OVERNIGHT_DEPOSIT', 'dummy_issuer', 'NONE', 'EUR',
    #     refdate, startdate, enddate, 100, 0.01, 'Act365Fixed')
    ccy = "EUR"
    dcc = "Act360"
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

    end_date_deposits = [
        start_date + dt.timedelta(days=7),
        start_date + dt.timedelta(days=30),
        start_date + dt.timedelta(days=60),
        start_date + dt.timedelta(days=90),
        start_date + dt.timedelta(days=120),
        start_date + dt.timedelta(days=150),
        start_date + dt.timedelta(days=360),
    ]

    quotes_deposits = [0.0512, 0.0515, 0.0518, 0.052, 0.0525, 0.0528, 0.053]
    epsilon = 0.007
    other_deposits = []

    for i in range(len(quotes_deposits)):
        temp_deposit = DepositSpecification(
            obj_id="DEPOSIT_" + str(i + 1),
            issuer="dummy_issuer",
            currency=ccy,
            fixing_date=ref_date,
            start_date=start_date,
            maturity_date=end_date_deposits[i],
            notional=notional,
            # rate=rate + i * 0.007,
            rate=quotes_deposits[i] + epsilon,
            day_count_convention=dcc,
        )
        other_deposits.append(temp_deposit)

    ##########################################
    # setting up swap
    # 1Y maturity with quartlery payment
    # start dates of the accrual periods corresponding to the tenor of the underlying index (3 months). The spot lag is set to 0.
    start_dates = [ref_date + relativedelta(months=3 * i) for i in range(4)]

    # reset dates are equal to start dates if spot lag is 0.
    reset_dates = start_dates

    # the end dates of the accral periods
    end_dates = [x + relativedelta(months=3) for x in start_dates]

    # # 1 YEAR with 1 payment
    # # start dates of the accrual periods corresponding to the tenor of the underlying index (3 months). The spot lag is set to 0.
    # start_dates = [ref_date]

    # # reset dates are equal to start dates if spot lag is 0.
    # reset_dates = start_dates

    # # the end dates of the accral periods
    # end_dates = [x + relativedelta(months=12) for x in start_dates]

    # the actual payment dates of the cashflows may differ from the end of the accrual period (e.g. OIS).
    # in the standard case these two sets of dates coincide
    pay_dates = end_dates
    # notionals = [1.0 for i in range(len(start_dates))] #i.e constant ...
    print(end_dates)
    ns = ConstNotionalStructure(100.0)
    spread = 0.00

    # # definition of the floating leg
    # floatleg = pyvacon.finance.specification.IrFloatLegSpecification(
    #     notionals, reset_dates, start_dates, end_dates, pay_dates,'EUR', 'dummy_udl',
    #     'Act365Fixed', 0.0)
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
    # fixedleg = pyvacon.finance.specification.IrFixedLegSpecification(
    #     0.01, notionals, start_dates, end_dates, pay_dates,'EUR', 'Act365Fixed')
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
    # ir_swap = pyvacon.finance.specification.InterestRateSwapSpecification(
    #     '3M_SWAP', 'dummy_issuer', 'COLLATERALIZED', 'EUR', pay_dates[-1], fixedleg, floatleg)
    # maturity_date = ref_date + dt.timedelta(600) # arbitrarily long
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

    # instruments = [deposit, ir_swap]
    # quotes = [0.0025, 0.01]
    instruments = [ir_swap]
    quotes = [0.05]
    # instruments = other_deposits
    # quotes = quotes_deposits

    estr = bootstrap_curve(ref_date, "ESTR_DC", DayCounterType.Act365Fixed, instruments, quotes)

    plt.figure(1)
    estr.plot(discount_factors=True)
    plt.xlabel("year")
    plt.ylabel("DF")
    plt.legend()

    plt.figure(2)
    estr.plot(discount_factors=False)
    plt.xlabel("year")
    plt.ylabel("zero rate")
    plt.legend()

    plt.figure(3)
    dates_final = estr.get_dates()
    df_final = estr.get_df()

    # calculated df from dates
    dates_new = [dates_final[0]]
    days = 10  # create a smoother curve by adding dates in between
    for i in range(1, len(dates_final)):
        while dates_new[-1] + dt.timedelta(days=days) < dates_final[i]:
            dates_new.append(dates_new[-1] + dt.timedelta(days=days))
        dates_new.append(dates_final[i])

    print(dates_new)
    values = [estr.rivapy_value(estr.refdate, d) for d in dates_new]
    print(values)

    plt.plot(dates_final, df_final, marker="^", label="bootstrapped")

    plt.plot(dates_new, values, label="interpolated")

    plt.xlabel("year")
    plt.ylabel("DF")
    plt.legend()

    # continuous compounding
    plt.figure(4)
    dcc = DayCounter(estr.daycounter)
    zero_final = []
    for i in range(1, len(df_final)):
        delta_t = dcc.yf(estr.refdate, dates_final[i])  # float((dates_new[i] - estr.refdate).days) / 365.0
        zero_final.append(-math.log(df_final[i]) / delta_t)
    zero_final.insert(0, zero_final[0])  # adding first entry to account for division by zero for dt

    zero_interp = []
    for i in range(1, len(values)):
        delta_t = dcc.yf(estr.refdate, dates_new[i])  # float((dates_new[i] - estr.refdate).days) / 365.0
        zero_interp.append(-math.log(values[i]) / delta_t)
    zero_interp.insert(0, zero_interp[0])

    plt.plot(dates_final, zero_final, marker="^", label="bootstrapped")
    # plt.scatter(end_date_deposits, quotes_deposits)
    plt.plot(dates_new, zero_interp, label="interpolated")

    plt.xlabel("year")
    plt.ylabel("zero rates")
    plt.legend()
    plt.show()

    ##################################################################################

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

    # Specify a couple of OIS swaps
    # Note that here we represent them as a typical IR swap, except that the float leg uses an OIS specification instead of FLoatIRS specification.
    # THe fixed IRS specification is used the same

    # We will use OIS based on a 1 WEEk, 1M,3M, 6M, 9M underlying indexes
    # will assume 1 year maturity for all ...

    # MATURITY has to match underlying TENOR????, by that logic, it is just 1 accrual period?
    # well it needs to have increasing maturity ... so we will do that for now ...

    # calculation date # same as deposit
    # ref_date = dt.datetime(2019, 8, 31)

    ######################################################################
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

    res = temp_ois_scheduler(start_dates, end_dates)

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

    res = temp_ois_scheduler(start_dates, end_dates)

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

    res = temp_ois_scheduler(start_dates, end_dates)

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

    res = temp_ois_scheduler(start_dates, end_dates)

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

    #####################################################
    # combine the multiple instruments to be given to the bootstrapper
    instruments_ois = [ois_swap_1M, ois_swap_3M, ois_swap_6M, ois_swap_9M]
    quotes_ois = [-0.00358, -0.00358, -0.00358, -0.00357]  # taken from the .csv as example

    # instruments_ois = [ois_swap_3M, ois_swap_6M]
    # quotes_ois = [-0.00358, -0.00358] #taken from the .csv as example

    instruments_ois = [deposit_estr, ois_swap_1M, ois_swap_3M, ois_swap_6M, ois_swap_9M]
    quotes_ois = [-0.00345, -0.00358, -0.00358, -0.00358, -0.00357]  # taken from the .csv as example

    # instruments_ois = [deposit_estr, ois_swap_1M]
    # quotes_ois = [-0.00345, -0.00358]  # taken from the .csv as example

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
    )

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
    values = [boot_curve_ois.rivapy_value(boot_curve_ois.refdate, d) for d in dates_new]
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
