# imports
import datetime as dt
from dateutil.relativedelta import relativedelta
import pandas as pd
import matplotlib.pyplot as plt
from rivapy.marketdata.bootstrapping_2025 import bootstrap_curve
from rivapy.instruments.deposit_specifications import DepositSpecification
from rivapy.instruments.fra_specifications import ForwardRateAgreementSpecification
from rivapy.instruments.ir_swap_specification import InterestRateSwapSpecification, IrFixedLegSpecification, IrFloatLegSpecification
from rivapy.instruments.notional_structure import ConstNotionalStructure
from rivapy.tools.enums import DayCounterType

import matplotlib.pyplot as plt
import math
from rivapy.tools.datetools import DayCounter

# from rivapy.pricing._logger import logger
from rivapy.pricing.deposit_pricing import DepositPricer
from rivapy.pricing.interest_rate_swap_pricing import InterestRateSwapPricer

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
