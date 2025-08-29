import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import date2num, DateFormatter
from rivapy.pricing.bond_pricing import SimpleCashflowPricer
from rivapy.instruments import DepositSpecification
from rivapy.tools.datetools import Period, _date_to_datetime, _term_to_period, _string_to_calendar, DayCounter, Schedule, roll_day


def plot_deposit_timeline(spec: DepositSpecification, val_date) -> plt.Figure:
    """Creates a two-panel timeline visualization with shared x-axis."""
    payment_date = roll_day(
        spec._maturity_date, calendar=spec._calendar, business_day_convention=spec._business_day_convention, settle_days=spec._payment_days
    )
    print(spec._payment_days)
    # Get relevant dates and cashflows
    relevant_dates = {
        "Fixing Date": spec.first_fixing_date,
        "Start Date": spec.start_date,
        "End Date": spec.end_date,
        "Maturity Date": spec.maturity_date,
        "Payment Date": payment_date,
    }
    cashflows = SimpleCashflowPricer.get_expected_cashflows(spec, val_date)

    # Create figure with two subplots sharing x-axis
    fig, (ax_cf, ax_timeline) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1, 1.5])
    plt.subplots_adjust(hspace=0.15)  # Remove spacing between subplots

    # Get all unique dates
    all_dates = set(relevant_dates.values())
    all_dates.update(date for date, _ in cashflows)
    dates_list = sorted(list(all_dates))

    # Convert dates to x-coordinates
    x_coords = np.linspace(0, 1, len(dates_list))
    date_to_x = dict(zip(dates_list, x_coords))
    # date_to_x = {date: date2num(date) for date in dates_list}

    # Plot cashflows in top subplot with y-axis
    max_cf = max(abs(cf[1]) for cf in cashflows)
    for date, amount in cashflows:
        x = date_to_x[date]
        color = "green" if amount > 0 else "red"

        # Draw cashflow box
        ax_cf.add_patch(plt.Rectangle((x - 0.05, 0), 0.1, amount, facecolor=color, alpha=0.3))

        # Add cashflow label
        y_pos = amount + np.sign(amount) * max_cf * 0.1
        ax_cf.annotate(
            f"CF: {amount:.2f}",
            (x, y_pos),
            ha="center",
            va="bottom" if amount > 0 else "top",
            bbox=dict(facecolor="white", edgecolor="lightgray", alpha=0.9, pad=2),
        )

    # Add timeline dots to cashflow subplot
    for date in dates_list:
        x = date_to_x[date]
        ax_cf.plot(x, 0, "ko", markersize=6)  # Smaller dots in top subplot
        ax_cf.annotate(
            date.strftime("%Y-%m-%d"),
            (x, -max_cf * 0.1),
            ha="center",
            va="top",
            rotation=45,
            fontsize=8,
            bbox=dict(facecolor="white", edgecolor="lightgray", alpha=0.9, pad=1),
        )

    # Configure cashflow subplot
    ax_cf.set_ylim(min(-max_cf * 1.2, ax_cf.get_ylim()[0]), max(max_cf * 1.2, ax_cf.get_ylim()[1]))
    ax_cf.spines["right"].set_visible(False)  # Remove right border
    ax_cf.spines["top"].set_visible(False)  # Remove top border
    ax_cf.spines["bottom"].set_position(("data", 0))
    #
    # ax_cf.set_xlabel("")  # Remove x label
    ax_cf.set_title("Cashflows", pad=10)

    # Plot timeline in bottom subplot
    timeline_y = 0
    ax_timeline.hlines(y=timeline_y, xmin=0, xmax=1, color="black", linewidth=2)

    date_positions = {}
    for label, date in relevant_dates.items():
        x = date_to_x[date]
        if x in date_positions:
            date_positions[x].append(label)
        else:
            date_positions[x] = [label]

    # Add date markers and staggered labels
    for x, labels in date_positions.items():
        # Plot single marker for this x position
        ax_timeline.plot(x, timeline_y, "ko", markersize=6)

        # Calculate offsets based on number of labels at this position
        for i, label in enumerate(labels):
            date = relevant_dates[label]
            # Only stagger if multiple labels at same position
            y_offset = 0.25 * (i + 1) if len(labels) > 1 else 0.2

            ax_timeline.annotate(
                f'{label}\n{date.strftime("%A")}\n{date.strftime("%Y-%m-%d")}',
                (x, timeline_y + y_offset),
                ha="center",
                va="center",
                bbox=dict(facecolor="white", edgecolor="lightgray", alpha=0.9, pad=3),
            )

    # Add period braces
    def add_period_brace(x1, x2, label):
        y = timeline_y - 0.2
        mid_x = (x1 + x2) / 2
        ax_timeline.plot([x1, x1, mid_x, x2, x2], [y, y - 0.1, y - 0.15, y - 0.1, y], "k-", linewidth=1.5)
        ax_timeline.text(mid_x, y - 0.2, label, ha="center", va="top", bbox=dict(facecolor="white", edgecolor="lightgray", alpha=0.9, pad=2))

    # Add period markings
    if spec.first_fixing_date != spec.start_date:
        add_period_brace(date_to_x[spec.first_fixing_date], date_to_x[spec.start_date], "Settlement Period")
    if spec.start_date != spec.end_date:
        add_period_brace(date_to_x[spec.start_date], date_to_x[spec.end_date], "Accrual Period")
    if spec.end_date != spec.maturity_date:
        add_period_brace(date_to_x[spec.end_date], date_to_x[spec.maturity_date], "Mon Accrual Period")
    if spec.maturity_date != payment_date:
        add_period_brace(date_to_x[spec.maturity_date], date_to_x[payment_date], "Final Settlement Period \n(aka Payment Lag)")

    # Configure timeline subplot
    ax_timeline.set_ylim(-1, 0.5)
    ax_timeline.set_xlim(-0.1, 1.1)
    ax_timeline.axis("off")  # Remove all axes
    # ax_timeline.set_title("Timeline", pad=10)

    # Share x-axis between subplots
    ax_cf.set_xlim(ax_timeline.get_xlim())
    # ax_cf.xaxis.set_visible(False)  # Hide x-axis but keep y-axis

    return fig
