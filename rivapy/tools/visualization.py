def _map_dates_to_xcoords(dates_list):
    """
    Map dates to x-coordinates so that periods shorter than 7 days are true to scale,
    while longer periods are compressed with interruptions.
    """
    if len(dates_list) < 2:
        return {d: 0.0 for d in dates_list}
    # Calculate day differences between consecutive dates
    day_diffs = [(dates_list[i + 1] - dates_list[i]).days for i in range(len(dates_list) - 1)]
    # For each gap, if < 7 days, use true scale; if >= 7, use fixed compressed width
    true_scale_unit = 1.0  # 1 day = 1 unit (arbitrary, will normalize later)
    compressed_gap = 3.0  # fixed width for long gaps (arbitrary, will normalize later)
    x_coords = [0.0]
    for i, diff in enumerate(day_diffs):
        if diff < 7:
            x_coords.append(x_coords[-1] + diff * true_scale_unit)
        else:
            x_coords.append(x_coords[-1] + compressed_gap)
    # Normalize to [0, 1]
    min_x, max_x = min(x_coords), max(x_coords)
    if max_x - min_x == 0:
        norm_x_coords = [0.0 for _ in x_coords]
    else:
        norm_x_coords = [(x - min_x) / (max_x - min_x) for x in x_coords]
    return dict(zip(dates_list, norm_x_coords))


def _draw_interrupted_axis(ax, sorted_dates, date_to_x, y, gap=0.015, bar_height=0.15, linewidth=1, axis="x"):
    """Draw an axis with interruptions (vertical bars) for long periods (>7 days)."""
    for i in range(len(sorted_dates) - 1):
        d1, d2 = sorted_dates[i], sorted_dates[i + 1]
        x1, x2 = date_to_x[d1], date_to_x[d2]
        try:
            delta = (d2 - d1).days
        except Exception:
            delta = 0
        if delta > 7:
            mid_x = (x1 + x2) / 2
            if axis == "x":
                ax.plot([x1, mid_x - gap], [y, y], color="black", linewidth=linewidth)
                ax.plot([mid_x + gap, x2], [y, y], color="black", linewidth=linewidth)
                ax.plot([mid_x - gap / 2, mid_x - gap / 2], [y - bar_height / 2, y + bar_height / 2], color="black", linewidth=linewidth)
                ax.plot([mid_x + gap / 2, mid_x + gap / 2], [y - bar_height / 2, y + bar_height / 2], color="black", linewidth=linewidth)
            else:
                ax.plot([y, y], [x1, mid_x - gap], color="black", linewidth=linewidth)
                ax.plot([y, y], [mid_x + gap, x2], color="black", linewidth=linewidth)
                ax.plot([y - bar_height / 2, y + bar_height / 2], [mid_x - gap / 2, mid_x - gap / 2], color="black", linewidth=linewidth)
                ax.plot([y - bar_height / 2, y + bar_height / 2], [mid_x + gap / 2, mid_x + gap / 2], color="black", linewidth=linewidth)
        else:
            if axis == "x":
                ax.plot([x1, x2], [y, y], color="black", linewidth=linewidth)
            else:
                ax.plot([y, y], [x1, x2], color="black", linewidth=linewidth)


def _place_date_marker_boxes(ax, date_positions, relevant_dates, timeline_y, marker_box_width=0.10, marker_box_height=0.18, vertical_gap=0.05):
    """Place date marker boxes so that no box overlaps horizontally with any later box."""
    sorted_positions = sorted(date_positions.items(), key=lambda item: item[0])
    rightmost_edge = -float("inf")
    for x, labels in sorted_positions:
        x_placement = max(x, rightmost_edge + 0.01)
        ax.plot(x_placement, timeline_y, "ko", markersize=6)
        for i, label in enumerate(labels):
            date = relevant_dates[label]
            y_offset = 0.2 + i * (marker_box_height + vertical_gap)
            ax.annotate(
                f'{label}\n{date.strftime("%A")}\n{date.strftime("%Y-%m-%d")}',
                (x_placement, timeline_y + y_offset),
                ha="center",
                va="center",
                bbox=dict(facecolor="white", edgecolor="lightgray", alpha=0.9, pad=3),
            )
        rightmost_edge = x_placement + marker_box_width


def _add_period_brace(ax, x1, x2, label, y_levels, timeline_y, y_base=-0.2, y_step=-0.2):
    assigned_level = None
    for level in range(10):
        y = timeline_y + y_base + y_step * level
        overlap = False
        for ox1, ox2, oy in y_levels:
            # Only stagger if periods truly overlap (not just touch)
            if oy == y and not (x2 <= ox1 or x1 >= ox2):
                overlap = True
                break
        if not overlap:
            assigned_level = level
            break
    if assigned_level is None:
        assigned_level = 0
        y = timeline_y + y_base
    y_levels.append((x1, x2, y))
    mid_x = (x1 + x2) / 2
    # Draw a true curly bracket using Bezier curves
    from matplotlib.path import Path
    import matplotlib.patches as patches

    dx = x2 - x1
    height = 0.18
    control = 0.18 * np.sign(dx)
    # Define control points for a curly bracket
    verts = [
        (x1, y),
        (x1, y - height * 0.25),
        (mid_x, y - height * 0.5),
        (mid_x, y - height),
        (mid_x, y - height * 0.5),
        (x2, y - height * 0.25),
        (x2, y),
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE3,
        Path.LINETO,
    ]
    path = Path(verts, codes)
    patch = patches.PathPatch(path, edgecolor="k", lw=1.5, facecolor="none")
    ax.add_patch(patch)
    ax.text(
        mid_x,
        y - height - 0.05,
        label,
        ha="center",
        va="top",
        bbox=dict(facecolor="white", edgecolor="lightgray", alpha=0.9, pad=2),
    )


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import date2num, DateFormatter
from rivapy.pricing.bond_pricing import DeterministicCashflowPricer
from rivapy.instruments import DepositSpecification, ForwardRateAgreementSpecification
from rivapy.tools.datetools import Period, _date_to_datetime, _term_to_period, _string_to_calendar, DayCounter, Schedule, roll_day


def plot_timeline_and_cf(
    relevant_dates: dict,
    periods: list,
    cashflows,
    figsize=(12, 10),
    height_ratios=[1, 2.5],
) -> plt.Figure:
    """Generic two-panel timeline visualization with shared x-axis.
    Args:
        relevant_dates: dict of label -> datetime for date markers.
        periods: list of tuples (label, date1, date2) for curly braces.
        cashflows: list of (date, amount) tuples.
        figsize, height_ratios: matplotlib figure options.
    """
    # Create figure with two subplots sharing x-axis
    fig, (ax_cf, ax_timeline) = plt.subplots(2, 1, figsize=figsize, height_ratios=height_ratios)
    plt.subplots_adjust(hspace=0.0)

    # Get all unique dates
    all_dates = set(relevant_dates.values())
    all_dates.update(date for date, _ in cashflows)
    dates_list = sorted(list(all_dates))

    # Convert dates to x-coordinates
    date_to_x = _map_dates_to_xcoords(dates_list)

    # Plot cashflows in top subplot with y-axis
    max_cf = max(abs(cf[1]) for cf in cashflows)
    sorted_dates = sorted(dates_list)
    gap = 0.015
    ax_cf.set_ylim(min(-max_cf * 1.2, ax_cf.get_ylim()[0]), max(max_cf * 1.2, ax_cf.get_ylim()[1]))
    ax_cf.set_xticks([])
    ax_cf.set_xlabel("")
    bar_height = 0.15 * (ax_cf.get_ylim()[1] - ax_cf.get_ylim()[0])
    _draw_interrupted_axis(ax_cf, sorted_dates, date_to_x, 0, gap=gap, bar_height=bar_height, linewidth=1, axis="x")

    for date, amount in cashflows:
        x = date_to_x[date]
        color = "green" if amount > 0 else "red"
        ax_cf.add_patch(plt.Rectangle((x - 0.05, 0), 0.1, amount, facecolor=color, alpha=0.3))
        y_pos = amount + np.sign(amount) * max_cf * 0.1
        ax_cf.annotate(
            f"CF: {amount:.2f}",
            (x, y_pos),
            ha="center",
            va="bottom" if amount > 0 else "top",
            bbox=dict(facecolor="white", edgecolor="lightgray", alpha=0.9, pad=2),
        )

    for date in dates_list:
        x = date_to_x[date]
        ax_cf.plot(x, 0, "ko", markersize=6)
        ax_cf.annotate(
            date.strftime("%Y-%m-%d"),
            (x, -max_cf * 0.1),
            ha="center",
            va="top",
            rotation=45,
            fontsize=8,
            bbox=dict(facecolor="white", edgecolor="lightgray", alpha=0.9, pad=1),
        )

    ax_cf.spines["right"].set_visible(False)
    ax_cf.spines["top"].set_visible(False)
    ax_cf.spines["bottom"].set_position(("data", 0))
    ax_cf.set_title("Cashflows", pad=10)

    timeline_y = 0
    _draw_interrupted_axis(ax_timeline, sorted_dates, date_to_x, timeline_y, gap=gap, bar_height=0.15, linewidth=2, axis="x")

    date_positions = {}
    for label, date in relevant_dates.items():
        x = date_to_x[date]
        if x in date_positions:
            date_positions[x].append(label)
        else:
            date_positions[x] = [label]

    _place_date_marker_boxes(ax_timeline, date_positions, relevant_dates, timeline_y)

    # Sort periods by length (shortest first), but preserve original order for equal lengths
    period_objs = [
        {"label": label, "d1": d1, "d2": d2, "x1": date_to_x[d1], "x2": date_to_x[d2], "length": abs((d2 - d1).days)}
        for label, d1, d2 in periods
        if d1 != d2
    ]
    # Sort by length, then by x1 (for deterministic stacking)
    period_objs.sort(key=lambda p: (p["length"], p["x1"]))
    y_levels = []
    for p in period_objs:
        _add_period_brace(ax_timeline, p["x1"], p["x2"], p["label"], y_levels, timeline_y)

    ax_timeline.set_ylim(-1.0, 1.0)
    ax_timeline.set_xlim(-0.1, 1.1)
    ax_timeline.axis("off")
    ax_cf.set_xlim(ax_timeline.get_xlim())
    return fig


# Instrument-specific wrapper for deposits
def plot_deposit(spec: DepositSpecification, val_date) -> plt.Figure:
    payment_date = roll_day(
        spec._maturity_date, calendar=spec._calendar, business_day_convention=spec._business_day_convention, settle_days=spec._payment_days
    )
    relevant_dates = {
        "Fixing Date": spec.first_fixing_date,
        "Start Date": spec.start_date,
        "End Date": spec.end_date,
        "Maturity Date": spec.maturity_date,
        "Payment Date": payment_date,
    }
    periods = []
    if spec.first_fixing_date != spec.start_date:
        periods.append(("Settlement Period", spec.first_fixing_date, spec.start_date))
    if spec.start_date != spec.end_date:
        periods.append(("Accrual Period", spec.start_date, spec.end_date))
    if spec.end_date != spec.maturity_date:
        periods.append(("Mon Accrual Period", spec.end_date, spec.maturity_date))
    if spec.maturity_date != payment_date:
        periods.append(("Payment Lag)", spec.maturity_date, payment_date))
    from rivapy.pricing.bond_pricing import DeterministicCashflowPricer

    cashflows = DeterministicCashflowPricer.get_expected_cashflows(spec, val_date)
    return plot_timeline_and_cf(relevant_dates, periods, cashflows)


# Instrument-specific wrapper for forward rate agreements
def plot_fra(spec: ForwardRateAgreementSpecification, val_date, fwd_curve) -> plt.Figure:
    payment_date = roll_day(
        spec._start_date, calendar=spec._calendar, business_day_convention=spec._business_day_convention, settle_days=spec._payment_days
    )

    relevant_dates = {
        "Trade Date": spec._trade_date,
        "Fixing Date": spec._fixing_date,
        "Start Date": spec._start_date,
        "Rate Start Date": spec._rate_start_date,
        "End Date": spec._end_date,
        "Rate End Date": spec._rate_end_date,
        "Maturity Date": spec._maturity_date,
        "Payment Date": payment_date,
    }
    periods = []
    if spec._trade_date != spec._start_date:
        periods.append(("Lead Period \n(aka Start Period)", spec._trade_date, spec._start_date))
        periods.append(("Contract Period", spec._start_date, spec._end_date))
        periods.append(("End Period", spec._trade_date, spec._end_date))
    if spec._start_date != payment_date:
        periods.append(("Payment Lag", spec._start_date, payment_date))
    from rivapy.pricing import ForwardRateAgreementPricer

    cashflows = ForwardRateAgreementPricer.get_expected_cashflows(spec, val_date, fwd_curve)
    return plot_timeline_and_cf(relevant_dates, periods, cashflows, figsize=(12, 13), height_ratios=[1, 3.0])
