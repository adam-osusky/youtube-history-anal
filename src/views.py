import logging

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def hourly_views(df: pd.DataFrame) -> None:
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Europe/Prague")

    # Define 3-month windows starting every month
    start = df["time"].min().floor("D")
    end = df["time"].max()
    windows = []
    offset = 6
    while start + pd.DateOffset(months=offset) <= end:
        windows.append((start, start + pd.DateOffset(months=offset)))
        start += pd.DateOffset(months=1)

    fig, ax = plt.subplots(figsize=(16, 10))

    def update(frame):
        ax.clear()
        wstart, wend = windows[frame]
        subset = df[(df["time"] >= wstart) & (df["time"] < wend)]
        counts = subset["time"].dt.hour.value_counts().sort_index()

        ax.bar(counts.index, counts.values, alpha=0.8)

        # Dynamic y-axis: just above the tallest bar this frame
        if not counts.empty:
            ax.set_ylim(0, counts.max() * 1.1)
        else:
            ax.set_ylim(0, 1)

        ax.set_xlabel("Hour of Day (Prague)")
        ax.set_ylabel("Number of Views")
        ax.set_title(f"Hourly Views: {wstart.date()} to {wend.date()}", pad=10)
        ax.grid(axis="y", linestyle="--", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    logger.info("Started creating animation for hourly views.")
    ani = animation.FuncAnimation(fig, update, frames=len(windows), interval=800)
    output_fp = "data/hourly_views.mp4"
    ani.save(output_fp, writer="ffmpeg")
    logger.info(f"Animation for hourly views saved to {output_fp}")


def daily_views(df: pd.DataFrame) -> None:
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Europe/Prague")

    # Build rolling windows: 6-month spans, sliding by 1 month
    start = df["time"].min().floor("D")
    end = df["time"].max()
    window_length = 6  # in months

    windows = []
    while start + pd.DateOffset(months=window_length) <= end:
        windows.append((start, start + pd.DateOffset(months=window_length)))
        start += pd.DateOffset(months=1)

    # Order weekdays for consistent x-axis
    ordered_days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    fig, ax = plt.subplots(figsize=(16, 10))

    def update(i):
        ax.clear()
        wstart, wend = windows[i]
        subset = df[(df["time"] >= wstart) & (df["time"] < wend)]

        # count by weekday and reindex to keep order
        counts = (
            subset["time"]
            .dt.day_name()
            .value_counts()
            .reindex(ordered_days, fill_value=0)
        )

        ax.bar(ordered_days, counts.values, alpha=0.8)

        # dynamic y-limit
        maxv = counts.max()
        ax.set_ylim(0, maxv * 1.1 if maxv > 0 else 1)

        ax.set_title(f"Views by Day of Week\n{wstart.date()} → {wend.date()}", pad=12)
        ax.set_xlabel("Day of Week (Prague)")
        ax.set_ylabel("Number of Views")

        ax.grid(axis="y", linestyle="--", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.xticks(rotation=45)

    logger.info("Started creating animation for daily views.")
    ani = animation.FuncAnimation(
        fig, update, frames=len(windows), interval=800, repeat=False
    )
    output_fp = "data/daily_views.mp4"
    ani.save(output_fp, writer="ffmpeg")
    logger.info(f"Animation for daily views saved to {output_fp}")


def yearly_views(df: pd.DataFrame) -> None:
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Europe/Prague")

    # Build rolling windows: 1-year spans, sliding by i ear
    min_year = df["time"].dt.year.min()
    start = pd.Timestamp(f"{min_year}-01-01 00:00:00", tz="Europe/Prague")
    end = df["time"].max()
    window_years = 1

    windows = []
    while start <= end:
        windows.append((start, start + pd.DateOffset(years=window_years)))
        start += pd.DateOffset(years=1)

    # Month labels 1–12
    months = list(range(1, 13))

    fig, ax = plt.subplots(figsize=(16, 10))

    def update(i) -> None:
        ax.clear()
        wstart, wend = windows[i]
        subset = df[(df["time"] >= wstart) & (df["time"] < wend)]

        # Count by month (1–12)
        counts = (
            subset["time"]
            .dt.month.value_counts()
            .sort_index()
            .reindex(months, fill_value=0)
        )

        ax.bar(months, counts.values, alpha=0.8)

        # Dynamic y-limit
        maxv = counts.max()
        ax.set_ylim(0, maxv * 1.1 if maxv > 0 else 1)

        ax.set_title(f"Views by Month\n{wstart.date()} → {wend.date()}", pad=12)
        ax.set_xlabel("Month (Prague)")
        ax.set_ylabel("Number of Views")
        ax.set_xticks(months)

        ax.grid(axis="y", linestyle="--", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    logger.info("Started creating animation for views by year.")
    ani = animation.FuncAnimation(
        fig, update, frames=len(windows), interval=2000, repeat=False
    )
    output_fp = "data/yearly_views.mp4"
    ani.save(output_fp, writer="ffmpeg")
    logger.info(f"Animation for views by year saved to {output_fp}")


def create_views_animations(df: pd.DataFrame) -> None:
    hourly_views(df.copy())
    daily_views(df.copy())
    yearly_views(df.copy())
