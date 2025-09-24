from rivapy.instruments.deposit_specifications import DepositSpecification
from rivapy.instruments.ir_swap_specification import (
    InterestRateSwapSpecification,
    IrFixedLegSpecification,
    IrFloatLegSpecification,
    IrOISLegSpecification,
)
from rivapy.instruments.fra_specifications import ForwardRateAgreementSpecification

from rivapy.pricing.deposit_pricing import DepositPricer
from rivapy.pricing.fra_pricing import ForwardRateAgreementPricer
from rivapy.pricing.interest_rate_swap_pricing import InterestRateSwapPricer

from rivapy.marketdata.curves import DiscountCurve
from rivapy.instruments.components import ConstNotionalStructure

from rivapy.tools.datetools import _term_to_period, DayCounter, roll_day
from rivapy.tools.enums import DayCounterType, InterpolationType, ExtrapolationType, Instrument
from datetime import date, timedelta, datetime
from rivapy.pricing.bond_pricing import SimpleCashflowPricer
import math


test_deposit = DepositSpecification(
    obj_id="test_deposit", fixing_date=date(2024, 1, 1), start_date=date(2024, 1, 1), term="6M", rate=0.05, notional=1000.0
)

test_fra = ForwardRateAgreementSpecification(
    obj_id="test_fra",
    issue_date=date(2024, 1, 1),
    maturity_date=date(2024, 6, 30),
    notional=1000.0,
    rate=0.03,
    start_date=date(2024, 3, 1),
    end_date=date(2024, 6, 1),
    udlID="EURIBOR_3M",
    rate_start_date=date(2024, 2, 28),
    rate_end_date=date(2024, 5, 30),
    currency="EUR",
)

test_ir_fixed_leg = IrFixedLegSpecification(
    fixed_rate=0.02,
    obj_id="test_fixed_leg",
    notional=1000.0,
    start_dates=[date(2024, 1, 1), date(2025, 1, 1)],
    end_dates=[date(2025, 1, 1), date(2026, 1, 1)],
    pay_dates=[date(2025, 1, 1), date(2026, 1, 1)],
    currency="EUR",
)

test_ir_float_leg = IrFloatLegSpecification(
    obj_id="test_float_leg",
    notional=1000.0,
    reset_dates=[date(2024, 1, 1), date(2025, 1, 1)],
    start_dates=[date(2024, 1, 1), date(2025, 1, 1)],
    end_dates=[date(2025, 1, 1), date(2026, 1, 1)],
    rate_start_dates=[date(2024, 1, 1), date(2025, 1, 1)],
    rate_end_dates=[date(2025, 1, 1), date(2026, 1, 1)],
    pay_dates=[date(2025, 1, 1), date(2026, 1, 1)],
    currency="EUR",
    udl_id="EURIBOR_6M",  # dummy for labelling only
    fixing_id="EURIBOR_6M_FIX",  # dummy not tested here
    spread=0.000,
)

ns = ConstNotionalStructure(100.0)
refdate = date(2024, 1, 1)
maturity_date = refdate + timedelta(600)
ir_swap = InterestRateSwapSpecification(
    obj_id="dummy_swap",
    notional=ns,
    issue_date=date(2024, 1, 1),
    maturity_date=maturity_date,
    pay_leg=test_ir_fixed_leg,
    receive_leg=test_ir_float_leg,
    currency="EUR",
    day_count_convention="EUR",
    issuer="DBK",
    securitization_level="COLLATERALIZED",
)


##############
# Set up (flat) rate curve
object_id = "TEST_CURVE"
flat_rate = 0.025

days_to_maturity = [1, 180, 365, 720, 3 * 365, 4 * 365, 10 * 365]
dates = [refdate + timedelta(days=d) for d in days_to_maturity]

df = [math.exp(-d / 365.0 * flat_rate) for d in days_to_maturity]
dc = DiscountCurve(id=object_id, refdate=refdate, dates=dates, df=df, interpolation=InterpolationType.LINEAR, extrapolation=ExtrapolationType.LINEAR)

# price deposit
print(f"Expected Cashflows: {SimpleCashflowPricer.get_expected_cashflows(test_deposit,refdate)}")
print(f"Price Deposit: {DepositPricer.get_price(refdate, test_deposit, dc)}")

# Determine fair rate
fair_rate = DepositPricer.get_implied_simply_compounded_rate(refdate, test_deposit, dc)
print(f"fair rate: {fair_rate}")

# new deposit with fair rate
fair_deposit_spec = DepositSpecification(
    obj_id="dummy_id",
    issuer="dummy_issuer",
    securitization_level="NONE",
    currency="EUR",
    fixing_date=refdate,
    start_date=refdate,
    maturity_date=maturity_date,
    notional=1000.0,
    rate=fair_rate,
    day_count_convention="Act360",
    payment_days=0,
)
print(f"fair deposit coupon: {fair_deposit_spec.coupon}")
print(DepositPricer.get_price(refdate, fair_deposit_spec, dc))

# Check fair rate
# discount factor from curve

df_curve = dc.value_fwd(refdate, fair_deposit_spec.start_date, fair_deposit_spec.maturity_date)
# discount factor implied by fair rate (simple compounding, Act/360)
dcc = DayCounter(fair_deposit_spec.day_count_convention)
delta_t = dcc.yf(fair_deposit_spec.start_date, fair_deposit_spec.maturity_date)
df_implied = 1.0 / (1.0 + delta_t * fair_rate)
print(df_curve)
print(df_implied)
