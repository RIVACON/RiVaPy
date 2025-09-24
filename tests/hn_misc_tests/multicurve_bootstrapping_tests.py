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


##############################
# Multicurve test
# calculation date
ref_date = dt.datetime(2019, 8, 31)  # refdate = dt.datetime(2017, 8, 31)
# boot_curve_ois
# Discount CURVE:

# TODO FIX
# since multiple runs in a file seem to not work at the moment...

# daycounter: DayCounterType.Act365Fixed
# extrapolation: ExtrapolationType.LINEAR_LOG
# id: bootstrapped_ois_DC
# interpolation: InterpolationType.LINEAR_LOG
# refdate: 2019-08-31 00:00:00


YC_dates = [
    dt.datetime(2019, 8, 31, 0, 0),
    dt.datetime(2019, 9, 3, 0, 0),
    dt.datetime(2019, 9, 13, 0, 0),
    dt.datetime(2019, 9, 23, 0, 0),
    dt.datetime(2019, 9, 30, 0, 0),
    dt.datetime(2019, 10, 10, 0, 0),
    dt.datetime(2019, 10, 20, 0, 0),
    dt.datetime(2019, 10, 30, 0, 0),
    dt.datetime(2019, 11, 9, 0, 0),
    dt.datetime(2019, 11, 19, 0, 0),
    dt.datetime(2019, 11, 29, 0, 0),
    dt.datetime(2019, 11, 30, 0, 0),
    dt.datetime(2019, 12, 10, 0, 0),
    dt.datetime(2019, 12, 20, 0, 0),
    dt.datetime(2019, 12, 30, 0, 0),
    dt.datetime(2020, 1, 9, 0, 0),
    dt.datetime(2020, 1, 19, 0, 0),
    dt.datetime(2020, 1, 29, 0, 0),
    dt.datetime(2020, 2, 8, 0, 0),
    dt.datetime(2020, 2, 18, 0, 0),
    dt.datetime(2020, 2, 28, 0, 0),
    dt.datetime(2020, 2, 29, 0, 0),
    dt.datetime(2020, 3, 10, 0, 0),
    dt.datetime(2020, 3, 20, 0, 0),
    dt.datetime(2020, 3, 30, 0, 0),
    dt.datetime(2020, 4, 9, 0, 0),
    dt.datetime(2020, 4, 19, 0, 0),
    dt.datetime(2020, 4, 29, 0, 0),
    dt.datetime(2020, 5, 9, 0, 0),
    dt.datetime(2020, 5, 19, 0, 0),
    dt.datetime(2020, 5, 29, 0, 0),
    dt.datetime(2020, 5, 31, 0, 0),
]

YC_DF = [
    1.0,
    1.00002835670044,
    1.000126873012154,
    1.0002253990290562,
    1.000294373015987,
    1.000392540026814,
    1.000490716671567,
    1.0005889029511918,
    1.0006870988666334,
    1.0007853044188377,
    1.0008835196087507,
    1.0008933416578523,
    1.000991647830395,
    1.0010899636584156,
    1.0011882891428627,
    1.0012866242846843,
    1.001384969084829,
    1.001483323544246,
    1.0015816876638832,
    1.00168006144469,
    1.001778444887615,
    1.0017882837633605,
    1.0018859407435448,
    1.0019836072435904,
    1.0020812832644257,
    1.0021789688069789,
    1.0022766638721778,
    1.002374368460951,
    1.0024720825742266,
    1.0025698062129333,
    1.0026675393779998,
    1.002687087154258,
]

# create final discount curve
boot_curve_ois = DiscountCurve(
    id="bootstrapped_ois_DC",
    refdate=ref_date,
    dates=YC_dates,  # populate with correct dates
    df=YC_DF,  # populated with corresponding discount factors
    interpolation=InterpolationType.LINEAR_LOG,
    extrapolation=ExtrapolationType.LINEAR_LOG,
    daycounter=DayCounterType.Act365Fixed,
)

# Relevant Instruments data
#    Maturity Instrument Currency     Quote UnderlyingIndex UnderlyingTenor
# 34       3M    DEPOSIT      EUR -0.003290         EURIBOR              3M  modified following modified following
# 35       6M    DEPOSIT      EUR -0.003277         EURIBOR              3M
# 36       9M    DEPOSIT      EUR -0.003238         EURIBOR              3M
# 37       1Y        IRS      EUR -0.003204         EURIBOR              3M
# 41       2Y        IRS      EUR -0.002615         EURIBOR              3M

##########################################
# 3M deposit

# setting up depoist


# start date of the accrual period with spot lag equal to 2 days
start_date = ref_date + dt.timedelta(days=2)

# end date of the accrual period is 1 day after startdate
end_date = start_date + relativedelta(months=3)

# specification of the deposit
ccy = "EUR"
dcc = "Act365Fixed"  # "Act360"normally a bond is ACT360, we keep consistency with the swaps
rate = 0.01
notional = 100.0
dep_3M = DepositSpecification(
    obj_id="3M_DEPOSIT",
    issuer="dummy_issuer",
    currency=ccy,
    fixing_date=ref_date,
    start_date=start_date,
    maturity_date=end_date,
    notional=notional,
    rate=rate,
    day_count_convention=dcc,
)
##########################################
# 6M deposit
# start date of the accrual period with spot lag equal to 2 days
start_date = ref_date + dt.timedelta(days=2)

# end date of the accrual period is 1 day after startdate
end_date = start_date + relativedelta(months=6)

# specification of the deposit
ccy = "EUR"
dcc = "Act365Fixed"  # "Act360"normally a bond is ACT360, we keep consistency with the swaps
rate = 0.01
notional = 100.0
dep_6M = DepositSpecification(
    obj_id="6M_DEPOSIT",
    issuer="dummy_issuer",
    currency=ccy,
    fixing_date=ref_date,
    start_date=start_date,
    maturity_date=end_date,
    notional=notional,
    rate=rate,
    day_count_convention=dcc,
)
##########################################
# 9M deposit
# start date of the accrual period with spot lag equal to 2 days
start_date = ref_date + dt.timedelta(days=2)

# end date of the accrual period is 1 day after startdate
end_date = start_date + relativedelta(months=9)

# specification of the deposit
ccy = "EUR"
dcc = "Act365Fixed"  # "Act360"normally a bond is ACT360, we keep consistency with the swaps
rate = 0.01
notional = 100.0
dep_9M = DepositSpecification(
    obj_id="9M_DEPOSIT",
    issuer="dummy_issuer",
    currency=ccy,
    fixing_date=ref_date,
    start_date=start_date,
    maturity_date=end_date,
    notional=notional,
    rate=rate,
    day_count_convention=dcc,
)
##########################################
# 1Y IRS
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
float_leg_1Y = IrFloatLegSpecification(
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
fixed_leg_1Y = IrFixedLegSpecification(
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
irs_1Y = InterestRateSwapSpecification(
    obj_id="3M_SWAP_1Y",
    notional=ns,
    issue_date=ref_date,
    maturity_date=pay_dates[-1],
    pay_leg=fixed_leg_1Y,
    receive_leg=float_leg_1Y,
    currency="EUR",
    day_count_convention="Act365Fixed",
    issuer="dummy_issuer",
    securitization_level="COLLATERALIZED",
)

##########################################
# 2 Yr IRS
# 2Y maturity 3M swap, i.e. the floating leg is reset every 3M
# start dates of the accrual periods corresponding to the tenor of the underlying index (3 months). The spot lag is set to 0.
start_dates = [ref_date + relativedelta(months=3 * i) for i in range(4 * 2)]

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
float_leg_2Y = IrFloatLegSpecification(
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
fixed_leg_2Y = IrFixedLegSpecification(
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
irs_2Y = InterestRateSwapSpecification(
    obj_id="3M_SWAP_2Y",
    notional=ns,
    issue_date=ref_date,
    maturity_date=pay_dates[-1],
    pay_leg=fixed_leg_2Y,
    receive_leg=float_leg_2Y,
    currency="EUR",
    day_count_convention="Act365Fixed",
    issuer="dummy_issuer",
    securitization_level="COLLATERALIZED",
)

# THIS CAUSES ERROR since deposits dont use the forward curve? and since only the forward curve is being changed, the brentq solver fails as
# there is no change ?...
# instruments_3M = [dep_3M, dep_6M, dep_9M, irs_1Y, irs_2Y]
# quotes_3M = [-0.003290, -0.003277, -0.003238, -0.003204, -0.002615]


# no deposits
instruments_3M = [irs_1Y, irs_2Y]
quotes_3M = [-0.003204, -0.002615]

euribor3MCurve = bootstrap_curve(
    ref_date,
    "euribor3MCurve",
    DayCounterType.Act365Fixed,
    instruments_3M,
    quotes_3M,
    curves={"discount_curve": boot_curve_ois},
    interpolation_type=InterpolationType.LINEAR_LOG,
    extrapolation_type=ExtrapolationType.LINEAR_LOG,
)


# PLotting Discount Curve

plt.figure(7)
dates_final = euribor3MCurve.get_dates()
df_final = euribor3MCurve.get_df()

# calculated df from dates
dates_new = [dates_final[0]]
days = 10  # create a smoother curve by adding dates in between
for i in range(1, len(dates_final)):
    while dates_new[-1] + dt.timedelta(days=days) < dates_final[i]:
        dates_new.append(dates_new[-1] + dt.timedelta(days=days))
    dates_new.append(dates_final[i])

print(dates_new)
values = [euribor3MCurve.value(euribor3MCurve.refdate, d) for d in dates_new]
print(values)

plt.plot(dates_final, df_final, marker="^", label="bootstrapped")
plt.plot(dates_new, values, label="interpolated")

plt.xlabel("year")
plt.ylabel("DF")
plt.legend()

# continuous compounding
plt.figure(8)
dcc = DayCounter(euribor3MCurve.daycounter)
zero_final = []
for i in range(1, len(df_final)):
    delta_t = dcc.yf(euribor3MCurve.refdate, dates_final[i])  # float((dates_new[i] - estr.refdate).days) / 365.0
    zero_final.append(-math.log(df_final[i]) / delta_t)
zero_final.insert(0, zero_final[0])  # adding first entry to account for division by zero for dt

zero_interp = []
for i in range(1, len(values)):
    delta_t = dcc.yf(euribor3MCurve.refdate, dates_new[i])  # float((dates_new[i] - estr.refdate).days) / 365.0
    zero_interp.append(-math.log(values[i]) / delta_t)
zero_interp.insert(0, zero_interp[0])

plt.plot(dates_final, zero_final, marker="^", label="bootstrapped")
# plt.scatter(end_date_deposits, quotes_deposits)
plt.plot(dates_new, zero_interp, label="interpolated")

plt.xlabel("year")
plt.ylabel("zero rates")
plt.legend()
plt.show()
