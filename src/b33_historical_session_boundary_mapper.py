"""
EMA-ADX TRADING SYSTEM - V3.0

B3.3 - HISTORICAL SESSION BOUNDARY MAPPER
XAUUSD ONLY

Purpose
-------
Map recurring broker/feed boundary families through the full
XAUUSD M1 history and determine:

    - exact occurrence dates
    - year
    - month
    - weekday
    - gap duration
    - clock times before/after the gap
    - historical clock-time shifts
    - consecutive/recurring behavior
    - calendar-period behavior
    - M1 candle immediately before the gap
    - M1 candle immediately after the gap

IMPORTANT
---------
This is a DIAGNOSTIC / EVIDENCE script.

It does NOT:

    - modify raw M1 data
    - fill missing candles
    - create candles
    - modify session_validator.py
    - classify gaps as valid/invalid
    - create trading signals
    - create strategy entries
    - use the 1M timeframe as a trading timeframe
    - build the final production session rules

The purpose is to gather evidence before the production
XAUUSD session model is finalized.


INPUTS
------
Raw XAUUSD M1:

    data/raw/XAUUSDm_M1_202108170000_202608131827.csv

Existing B3.1:

    data/session_model_b31/XAUUSDm/
        recurring_boundary_families.csv
        historical_clock_shifts.csv

Existing B3.2:

    data/session_boundary_evidence_b32/XAUUSDm/
        boundary_evidence.csv
        boundary_examples.csv


OUTPUT
------
    data/session_mapping_b33/XAUUSDm/

Files:

    boundary_occurrences.csv
    daily_boundary_map.csv
    yearly_boundary_map.csv
    monthly_boundary_map.csv
    weekday_boundary_map.csv
    observed_clock_shift_points.csv
    calendar_regime_map.csv
    b33_summary.csv
    B33_REPORT.txt


IMPORTANT INTERPRETATION RULE
-----------------------------
This script deliberately does NOT call the observed shifts
"Daylight Saving Time" automatically.

It reports historical clock shifts.

The next stage will determine whether those shifts are
consistent with a stable broker-session model or merely
historical feed behavior.
"""

from pathlib import Path
import math
import time

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw"

B31_DIR = (
    PROJECT_ROOT
    / "data"
    / "session_model_b31"
    / "XAUUSDm"
)

B32_DIR = (
    PROJECT_ROOT
    / "data"
    / "session_boundary_evidence_b32"
    / "XAUUSDm"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "session_mapping_b33"
    / "XAUUSDm"
)


# ============================================================
# XAUUSD ONLY
# ============================================================

SYMBOL = "XAUUSDm"

RAW_FILE = (
    DATA_DIR
    / "XAUUSDm_M1_202108170000_202608131827.csv"
)


# ============================================================
# B3.3 SETTINGS
# ============================================================

# Maximum clock-time distance from the established B3.2
# family anchor when mapping variable historical shifts.
#
# This is intentionally small. We do NOT want unrelated
# intraday gaps being absorbed into a recurring family.
MAX_CLOCK_SHIFT_MINUTES = 5


# Gap duration tolerance is derived from B3.1 family
# minimum/maximum duration where available.
#
# No hard-coded session duration is imposed.


# Minimum occurrences needed before a family/clock regime
# is considered useful for reporting.
MIN_REGIME_OCCURRENCES = 3


# Maximum number of days between consecutive observations
# before a sequence is considered broken.
#
# This is only for describing recurrence, NOT classification.
MAX_SEQUENCE_BREAK_DAYS = 14


# ============================================================
# EXISTING FAMILY FILES
# ============================================================

B31_FAMILY_FILE = (
    B31_DIR
    / "recurring_boundary_families.csv"
)

B32_EVIDENCE_FILE = (
    B32_DIR
    / "boundary_evidence.csv"
)


# ============================================================
# REQUIRED RAW COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "<DATE>",
    "<TIME>",
    "<OPEN>",
    "<HIGH>",
    "<LOW>",
    "<CLOSE>",
]


# ============================================================
# HELPERS
# ============================================================

def minute_of_day(value):
    """Convert timestamp/time into minute-of-day."""

    if pd.isna(value):
        return math.nan

    return (
        value.hour * 60
        + value.minute
    )


def minute_to_clock(value):
    """Convert minute-of-day to HH:MM."""

    if pd.isna(value):
        return ""

    value = int(value) % (24 * 60)

    hour = value // 60
    minute = value % 60

    return f"{hour:02d}:{minute:02d}"


def circular_clock_distance(a, b):
    """
    Distance between two times on a 24-hour clock.

    Example:
        23:59 and 00:01 -> 2 minutes
    """

    if pd.isna(a) or pd.isna(b):
        return math.inf

    difference = abs(
        float(a) - float(b)
    )

    return min(
        difference,
        1440 - difference,
    )


def parse_time_value(value):
    """
    Convert values such as:

        20:57
        20:57:00
        1257
        1257.0

    into minute-of-day.
    """

    if pd.isna(value):
        return math.nan

    text = str(value).strip()

    if not text:
        return math.nan

    # HH:MM
    if ":" in text:

        parts = text.split(":")

        try:

            hour = int(parts[0])
            minute = int(parts[1])

            return hour * 60 + minute

        except ValueError:
            return math.nan

    # Numeric minute-of-day
    try:

        number = float(text)

        if 0 <= number < 1440:
            return number

    except ValueError:
        pass

    return math.nan


def detect_separator(filepath):
    """Detect MT5 export separator."""

    with open(
        filepath,
        "r",
        encoding="utf-8-sig",
        errors="replace",
    ) as handle:

        first_line = handle.readline()

    if "\t" in first_line:
        return "\t", "TAB"

    if "," in first_line:
        return ",", "COMMA"

    if ";" in first_line:
        return ";", "SEMICOLON"

    raise ValueError(
        "Could not detect separator."
    )


# ============================================================
# LOAD RAW M1
# ============================================================

def load_xauusd_m1():
    """Load the existing XAUUSD M1 file."""

    print("\n" + "=" * 70)
    print("B3.3 - LOADING XAUUSD M1 DATA")
    print("=" * 70)

    if not RAW_FILE.exists():

        raise FileNotFoundError(
            f"XAUUSD raw file not found:\n{RAW_FILE}"
        )

    separator, separator_name = (
        detect_separator(RAW_FILE)
    )

    print(
        f"File: {RAW_FILE}"
    )

    print(
        f"Detected separator: {separator_name}"
    )

    df = pd.read_csv(
        RAW_FILE,
        sep=separator,
        encoding="utf-8-sig",
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    df["Datetime"] = pd.to_datetime(
        df["<DATE>"].astype(str).str.strip()
        + " "
        + df["<TIME>"].astype(str).str.strip(),
        format="%Y.%m.%d %H:%M:%S",
        errors="coerce",
    )

    invalid = df["Datetime"].isna().sum()

    if invalid:

        raise ValueError(
            f"Found {invalid:,} invalid timestamps."
        )

    df = (
        df.sort_values("Datetime")
        .reset_index(drop=True)
    )

    duplicate_count = (
        df["Datetime"]
        .duplicated()
        .sum()
    )

    if duplicate_count:

        print(
            f"WARNING: {duplicate_count:,} duplicate "
            f"timestamps detected."
        )

        df = (
            df.drop_duplicates(
                subset=["Datetime"],
                keep="first",
            )
            .reset_index(drop=True)
        )

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        f"First candle: {df['Datetime'].iloc[0]}"
    )

    print(
        f"Last candle:  {df['Datetime'].iloc[-1]}"
    )

    return df


# ============================================================
# BUILD ALL RAW GAPS
# ============================================================

def build_gap_table(m1):
    """
    Build every timestamp gap greater than one minute.

    The raw M1 data itself is not modified.
    """

    print("\nBuilding raw M1 gap table...")

    timestamps = m1["Datetime"]

    differences = timestamps.diff()

    gap_indices = differences.index[
        differences > pd.Timedelta(minutes=1)
    ]

    if len(gap_indices) == 0:

        return pd.DataFrame()

    rows = []

    for index in gap_indices:

        previous_index = index - 1

        previous_row = m1.iloc[
            previous_index
        ]

        current_row = m1.iloc[
            index
        ]

        previous_time = previous_row[
            "Datetime"
        ]

        current_time = current_row[
            "Datetime"
        ]

        gap_minutes = (
            current_time
            - previous_time
        ).total_seconds() / 60

        rows.append(
            {
                "previous": previous_time,
                "current": current_time,
                "gap_minutes": gap_minutes,

                "previous_date":
                    previous_time.date(),

                "current_date":
                    current_time.date(),

                "previous_year":
                    previous_time.year,

                "current_year":
                    current_time.year,

                "previous_month":
                    previous_time.month,

                "current_month":
                    current_time.month,

                "previous_weekday":
                    previous_time.day_name(),

                "current_weekday":
                    current_time.day_name(),

                "previous_minute":
                    minute_of_day(
                        previous_time
                    ),

                "current_minute":
                    minute_of_day(
                        current_time
                    ),

                "previous_open":
                    previous_row["<OPEN>"],

                "previous_high":
                    previous_row["<HIGH>"],

                "previous_low":
                    previous_row["<LOW>"],

                "previous_close":
                    previous_row["<CLOSE>"],

                "current_open":
                    current_row["<OPEN>"],

                "current_high":
                    current_row["<HIGH>"],

                "current_low":
                    current_row["<LOW>"],

                "current_close":
                    current_row["<CLOSE>"],
            }
        )

    gaps = pd.DataFrame(rows)

    print(
        f"Raw gaps > 1 minute: {len(gaps):,}"
    )

    return gaps


# ============================================================
# LOAD EXISTING B3 FAMILY INFORMATION
# ============================================================

def load_existing_families():
    """
    Load B3.1/B3.2 family information.

    We use the existing project evidence rather than
    recreating the previous family-discovery process.
    """

    print("\nLoading existing B3 family evidence...")

    if not B31_FAMILY_FILE.exists():

        raise FileNotFoundError(
            f"B3.1 family file not found:\n"
            f"{B31_FAMILY_FILE}"
        )

    if not B32_EVIDENCE_FILE.exists():

        raise FileNotFoundError(
            f"B3.2 evidence file not found:\n"
            f"{B32_EVIDENCE_FILE}"
        )

    b31 = pd.read_csv(
        B31_FAMILY_FILE
    )

    b32 = pd.read_csv(
        B32_EVIDENCE_FILE
    )

    b31.columns = (
        b31.columns
        .astype(str)
        .str.strip()
    )

    b32.columns = (
        b32.columns
        .astype(str)
        .str.strip()
    )

    required_b31 = {
        "boundary_family",
        "median_previous_minute",
        "median_current_minute",
        "min_gap_minutes",
        "max_gap_minutes",
    }

    missing_b31 = (
        required_b31
        - set(b31.columns)
    )

    if missing_b31:

        raise ValueError(
            "B3.1 family file is missing columns: "
            + ", ".join(
                sorted(missing_b31)
            )
        )

    required_b32 = {
        "family_id",
        "occurrences",
        "previous_time",
        "current_time",
        "median_gap_minutes",
    }

    missing_b32 = (
        required_b32
        - set(b32.columns)
    )

    if missing_b32:

        raise ValueError(
            "B3.2 evidence file is missing columns: "
            + ", ".join(
                sorted(missing_b32)
            )
        )

    # --------------------------------------------------------
    # Merge B3.1 + B3.2
    # --------------------------------------------------------

    families = b31.copy()

    families = families.rename(
        columns={
            "boundary_family": "family_id"
        }
    )

    b32_small = b32[
        [
            "family_id",
            "occurrences",
            "previous_time",
            "current_time",
            "median_gap_minutes",
            "classification",
        ]
    ].copy()

    families = families.merge(
        b32_small,
        on="family_id",
        how="left",
        suffixes=(
            "_b31",
            "_b32",
        ),
    )

    families["anchor_previous_minute"] = (
        families["previous_time"]
        .apply(parse_time_value)
    )

    families["anchor_current_minute"] = (
        families["current_time"]
        .apply(parse_time_value)
    )

    print(
        f"Existing recurring families: "
        f"{len(families)}"
    )

    print("\nFamilies used by B3.3:")

    for _, row in families.iterrows():

        print(
            f"  {row['family_id']}: "
            f"{row['previous_time']} -> "
            f"{row['current_time']} | "
            f"duration "
            f"{row['min_gap_minutes']:.1f}-"
            f"{row['max_gap_minutes']:.1f} min"
        )

    return families


# ============================================================
# MATCH RAW GAPS TO EXISTING FAMILIES
# ============================================================

def match_gap_to_family(gap_row, families):
    """
    Map one raw gap to an existing B3 family.

    Matching requires:

        1. gap duration within the family duration range
        2. previous clock time close to the family anchor
        3. current clock time close to the family anchor

    This allows historical clock shifts while preventing
    unrelated gaps from being absorbed.

    No production session rule is created here.
    """

    candidates = []

    gap_duration = (
        gap_row["gap_minutes"]
    )

    previous_minute = (
        gap_row["previous_minute"]
    )

    current_minute = (
        gap_row["current_minute"]
    )

    for _, family in families.iterrows():

        min_gap = (
            family["min_gap_minutes"]
        )

        max_gap = (
            family["max_gap_minutes"]
        )

        # Small numerical tolerance.
        if (
            gap_duration < min_gap - 0.01
            or
            gap_duration > max_gap + 0.01
        ):
            continue

        previous_distance = (
            circular_clock_distance(
                previous_minute,
                family[
                    "anchor_previous_minute"
                ],
            )
        )

        current_distance = (
            circular_clock_distance(
                current_minute,
                family[
                    "anchor_current_minute"
                ],
            )
        )

        if (
            previous_distance
            <= MAX_CLOCK_SHIFT_MINUTES
            and
            current_distance
            <= MAX_CLOCK_SHIFT_MINUTES
        ):

            total_distance = (
                previous_distance
                + current_distance
            )

            candidates.append(
                (
                    total_distance,
                    family["family_id"],
                )
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0]
    )

    return candidates[0][1]


# ============================================================
# MAP B3 FAMILIES
# ============================================================

def map_boundary_families(
    gaps,
    families,
):
    """Map historically shifted occurrences to known families."""

    print("\nMapping historical boundary occurrences...")

    if gaps.empty:
        return pd.DataFrame()

    mapped_rows = []

    for _, row in gaps.iterrows():

        family_id = match_gap_to_family(
            row,
            families,
        )

        if family_id is None:
            continue

        record = row.to_dict()

        record["family_id"] = family_id

        record["previous_clock"] = (
            minute_to_clock(
                row["previous_minute"]
            )
        )

        record["current_clock"] = (
            minute_to_clock(
                row["current_minute"]
            )
        )

        family = families[
            families["family_id"]
            == family_id
        ].iloc[0]

        record["anchor_previous_clock"] = (
            minute_to_clock(
                family[
                    "anchor_previous_minute"
                ]
            )
        )

        record["anchor_current_clock"] = (
            minute_to_clock(
                family[
                    "anchor_current_minute"
                ]
            )
        )

        record["previous_clock_shift_minutes"] = (
            row["previous_minute"]
            - family[
                "anchor_previous_minute"
            ]
        )

        record["current_clock_shift_minutes"] = (
            row["current_minute"]
            - family[
                "anchor_current_minute"
            ]
        )

        mapped_rows.append(record)

    mapped = pd.DataFrame(
        mapped_rows
    )

    if mapped.empty:

        print(
            "WARNING: No B3 families were mapped."
        )

        return mapped

    mapped = mapped.sort_values(
        "previous"
    ).reset_index(drop=True)

    print(
        f"Mapped boundary occurrences: "
        f"{len(mapped):,}"
    )

    print("\nMapped occurrences by family:")

    print(
        mapped["family_id"]
        .value_counts()
        .to_string()
    )

    return mapped


# ============================================================
# DAILY MAP
# ============================================================

def build_daily_map(mapped):
    """Create date-level boundary occurrence map."""

    if mapped.empty:
        return pd.DataFrame()

    daily = (
        mapped
        .groupby(
            [
                mapped["previous"].dt.date,
                "family_id",
            ],
            as_index=False,
        )
        .agg(
            occurrences=(
                "family_id",
                "size",
            ),
            first_previous_clock=(
                "previous_clock",
                "first",
            ),
            last_previous_clock=(
                "previous_clock",
                "last",
            ),
            first_current_clock=(
                "current_clock",
                "first",
            ),
            last_current_clock=(
                "current_clock",
                "last",
            ),
            median_gap_minutes=(
                "gap_minutes",
                "median",
            ),
            min_gap_minutes=(
                "gap_minutes",
                "min",
            ),
            max_gap_minutes=(
                "gap_minutes",
                "max",
            ),
        )
    )

    daily = daily.rename(
        columns={
            "previous": "date"
        }
    )

    daily["date"] = pd.to_datetime(
        daily["date"]
    )

    daily["year"] = (
        daily["date"].dt.year
    )

    daily["month"] = (
        daily["date"].dt.month
    )

    daily["weekday"] = (
        daily["date"].dt.day_name()
    )

    return daily.sort_values(
        ["date", "family_id"]
    )


# ============================================================
# YEARLY MAP
# ============================================================

def build_yearly_map(mapped):
    """Summarize family behavior by year."""

    if mapped.empty:
        return pd.DataFrame()

    yearly = (
        mapped
        .groupby(
            [
                mapped["previous"].dt.year,
                "family_id",
            ],
            as_index=False,
        )
        .agg(
            occurrences=(
                "family_id",
                "size",
            ),
            unique_dates=(
                "previous_date",
                "nunique",
            ),
            median_gap_minutes=(
                "gap_minutes",
                "median",
            ),
            min_gap_minutes=(
                "gap_minutes",
                "min",
            ),
            max_gap_minutes=(
                "gap_minutes",
                "max",
            ),
            median_previous_minute=(
                "previous_minute",
                "median",
            ),
            median_current_minute=(
                "current_minute",
                "median",
            ),
            min_previous_minute=(
                "previous_minute",
                "min",
            ),
            max_previous_minute=(
                "previous_minute",
                "max",
            ),
            min_current_minute=(
                "current_minute",
                "min",
            ),
            max_current_minute=(
                "current_minute",
                "max",
            ),
            unique_weekdays=(
                "previous_weekday",
                "nunique",
            ),
        )
    )

    yearly = yearly.rename(
        columns={
            "previous": "year"
        }
    )

    yearly["median_previous_clock"] = (
        yearly[
            "median_previous_minute"
        ].apply(minute_to_clock)
    )

    yearly["median_current_clock"] = (
        yearly[
            "median_current_minute"
        ].apply(minute_to_clock)
    )

    return yearly.sort_values(
        ["year", "family_id"]
    )


# ============================================================
# MONTHLY MAP
# ============================================================

def build_monthly_map(mapped):
    """Summarize family behavior by calendar month."""

    if mapped.empty:
        return pd.DataFrame()

    working = mapped.copy()

    working["year"] = (
        working["previous"].dt.year
    )

    working["month"] = (
        working["previous"].dt.month
    )

    monthly = (
        working
        .groupby(
            [
                "year",
                "month",
                "family_id",
            ],
            as_index=False,
        )
        .agg(
            occurrences=(
                "family_id",
                "size",
            ),
            unique_dates=(
                "previous_date",
                "nunique",
            ),
            median_gap_minutes=(
                "gap_minutes",
                "median",
            ),
            median_previous_minute=(
                "previous_minute",
                "median",
            ),
            median_current_minute=(
                "current_minute",
                "median",
            ),
            unique_weekdays=(
                "previous_weekday",
                "nunique",
            ),
        )
    )

    monthly["median_previous_clock"] = (
        monthly[
            "median_previous_minute"
        ].apply(minute_to_clock)
    )

    monthly["median_current_clock"] = (
        monthly[
            "median_current_minute"
        ].apply(minute_to_clock)
    )

    return monthly.sort_values(
        [
            "year",
            "month",
            "family_id",
        ]
    ).reset_index(drop=True)

    return monthly.sort_values(
    ["family_id"]
).reset_index(drop=True)


# ============================================================
# WEEKDAY MAP
# ============================================================

def build_weekday_map(mapped):
    """Summarize boundary behavior by weekday."""

    if mapped.empty:
        return pd.DataFrame()

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    weekday = (
        mapped
        .groupby(
            [
                "family_id",
                "previous_weekday",
            ],
            as_index=False,
        )
        .agg(
            occurrences=(
                "family_id",
                "size",
            ),
            unique_dates=(
                "previous_date",
                "nunique",
            ),
            median_gap_minutes=(
                "gap_minutes",
                "median",
            ),
            median_previous_minute=(
                "previous_minute",
                "median",
            ),
            median_current_minute=(
                "current_minute",
                "median",
            ),
        )
    )

    weekday = weekday.rename(
        columns={
            "previous_weekday":
                "weekday"
        }
    )

    weekday["weekday"] = pd.Categorical(
        weekday["weekday"],
        categories=weekday_order,
        ordered=True,
    )

    weekday["median_previous_clock"] = (
        weekday[
            "median_previous_minute"
        ].apply(minute_to_clock)
    )

    weekday["median_current_clock"] = (
        weekday[
            "median_current_minute"
        ].apply(minute_to_clock)
    )

    return weekday.sort_values(
        ["family_id", "weekday"]
    )


# ============================================================
# CLOCK SHIFT POINTS
# ============================================================

def build_clock_shift_points(mapped):
    """
    Identify points where the observed clock transition
    changes historically.

    A shift is an observed change from the previous
    occurrence's previous/current clock pair.

    This is evidence only.
    """

    if mapped.empty:
        return pd.DataFrame()

    rows = []

    for family_id, family_df in mapped.groupby(
        "family_id"
    ):

        family_df = (
            family_df
            .sort_values("previous")
            .reset_index(drop=True)
        )

        previous_pair = None
        previous_date = None

        for _, row in family_df.iterrows():

            current_pair = (
                row["previous_clock"],
                row["current_clock"],
            )

            if (
                previous_pair is not None
                and current_pair
                != previous_pair
            ):

                days_since_previous = (
                    row["previous"]
                    - previous_date
                ).total_seconds() / 86400

                rows.append(
                    {
                        "family_id":
                            family_id,

                        "shift_date":
                            row["previous"],

                        "previous_observed_pair":
                            f"{previous_pair[0]} -> "
                            f"{previous_pair[1]}",

                        "new_observed_pair":
                            f"{current_pair[0]} -> "
                            f"{current_pair[1]}",

                        "new_previous_clock":
                            row["previous_clock"],

                        "new_current_clock":
                            row["current_clock"],

                        "previous_occurrence_date":
                            previous_date,

                        "days_since_previous_occurrence":
                            days_since_previous,

                        "previous_year":
                            row["previous"].year,

                        "previous_month":
                            row["previous"].month,

                        "previous_weekday":
                            row[
                                "previous"
                            ].day_name(),

                        "previous_clock_shift_minutes":
                            row[
                                "previous_clock_shift_minutes"
                            ],

                        "current_clock_shift_minutes":
                            row[
                                "current_clock_shift_minutes"
                            ],

                        "gap_minutes":
                            row["gap_minutes"],
                    }
                )

            previous_pair = current_pair
            previous_date = row["previous"]

    return pd.DataFrame(rows)


# ============================================================
# CALENDAR REGIME MAP
# ============================================================

def build_calendar_regimes(mapped):
    """
    Group consecutive observations having the same observed
    clock transition.

    A regime is NOT declared to be DST.

    It simply means:

        "During this historical period, this family was
         observed using this clock transition."

    The output allows us to compare these periods against
    calendar dates later.
    """

    if mapped.empty:
        return pd.DataFrame()

    rows = []

    for family_id, family_df in mapped.groupby(
        "family_id"
    ):

        family_df = (
            family_df
            .sort_values("previous")
            .reset_index(drop=True)
        )

        family_df["clock_pair"] = (
            family_df["previous_clock"]
            + " -> "
            + family_df["current_clock"]
        )

        regime_number = 0

        start_index = 0

        for index in range(
            1,
            len(family_df),
        ):

            previous_row = (
                family_df.iloc[index - 1]
            )

            current_row = (
                family_df.iloc[index]
            )

            same_pair = (
                current_row["clock_pair"]
                ==
                previous_row["clock_pair"]
            )

            days_between = (
                current_row["previous"]
                - previous_row["previous"]
            ).total_seconds() / 86400

            sequence_broken = (
                days_between
                > MAX_SEQUENCE_BREAK_DAYS
            )

            if (
                not same_pair
                or sequence_broken
            ):

                regime = family_df.iloc[
                    start_index:index
                ]

                if len(regime) >= MIN_REGIME_OCCURRENCES:

                    regime_number += 1

                    rows.append(
                        {
                            "family_id":
                                family_id,

                            "regime_id":
                                f"{family_id}_R"
                                f"{regime_number:03d}",

                            "start_date":
                                regime[
                                    "previous"
                                ].min(),

                            "end_date":
                                regime[
                                    "previous"
                                ].max(),

                            "occurrences":
                                len(regime),

                            "unique_dates":
                                regime[
                                    "previous_date"
                                ].nunique(),

                            "years":
                                ",".join(
                                    map(
                                        str,
                                        sorted(
                                            regime[
                                                "previous"
                                            ].dt.year
                                            .unique()
                                        ),
                                    )
                                ),

                            "months":
                                ",".join(
                                    map(
                                        str,
                                        sorted(
                                            regime[
                                                "previous"
                                            ].dt.month
                                            .unique()
                                        ),
                                    )
                                ),

                            "weekdays":
                                ",".join(
                                    sorted(
                                        regime[
                                            "previous_weekday"
                                        ].unique()
                                    )
                                ),

                            "observed_previous_clock":
                                regime[
                                    "previous_clock"
                                ].mode().iloc[0],

                            "observed_current_clock":
                                regime[
                                    "current_clock"
                                ].mode().iloc[0],

                            "median_gap_minutes":
                                regime[
                                    "gap_minutes"
                                ].median(),

                            "min_gap_minutes":
                                regime[
                                    "gap_minutes"
                                ].min(),

                            "max_gap_minutes":
                                regime[
                                    "gap_minutes"
                                ].max(),

                            "median_previous_shift":
                                regime[
                                    "previous_clock_shift_minutes"
                                ].median(),

                            "median_current_shift":
                                regime[
                                    "current_clock_shift_minutes"
                                ].median(),

                            "sequence_break_before":
                                sequence_broken,
                        }
                    )

                start_index = index

        # Final regime
        regime = family_df.iloc[
            start_index:
        ]

        if len(regime) >= MIN_REGIME_OCCURRENCES:

            regime_number += 1

            rows.append(
                {
                    "family_id":
                        family_id,

                    "regime_id":
                        f"{family_id}_R"
                        f"{regime_number:03d}",

                    "start_date":
                        regime[
                            "previous"
                        ].min(),

                    "end_date":
                        regime[
                            "previous"
                        ].max(),

                    "occurrences":
                        len(regime),

                    "unique_dates":
                        regime[
                            "previous_date"
                        ].nunique(),

                    "years":
                        ",".join(
                            map(
                                str,
                                sorted(
                                    regime[
                                        "previous"
                                    ].dt.year.unique()
                                ),
                            )
                        ),

                    "months":
                        ",".join(
                            map(
                                str,
                                sorted(
                                    regime[
                                        "previous"
                                    ].dt.month.unique()
                                ),
                            )
                        ),

                    "weekdays":
                        ",".join(
                            sorted(
                                regime[
                                    "previous_weekday"
                                ].unique()
                            )
                        ),

                    "observed_previous_clock":
                        regime[
                            "previous_clock"
                        ].mode().iloc[0],

                    "observed_current_clock":
                        regime[
                            "current_clock"
                        ].mode().iloc[0],

                    "median_gap_minutes":
                        regime[
                            "gap_minutes"
                        ].median(),

                    "min_gap_minutes":
                        regime[
                            "gap_minutes"
                        ].min(),

                    "max_gap_minutes":
                        regime[
                            "gap_minutes"
                        ].max(),

                    "median_previous_shift":
                        regime[
                            "previous_clock_shift_minutes"
                        ].median(),

                    "median_current_shift":
                        regime[
                            "current_clock_shift_minutes"
                        ].median(),

                    "sequence_break_before":
                        False,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    mapped,
    families,
):
    """Create compact B3.3 summary."""

    rows = []

    for _, family in families.iterrows():

        family_id = family[
            "family_id"
        ]

        subset = mapped[
            mapped["family_id"]
            == family_id
        ]

        if subset.empty:

            rows.append(
                {
                    "family_id":
                        family_id,

                    "b31_occurrences":
                        family.get(
                            "occurrences_b31",
                            math.nan,
                        ),

                    "b32_occurrences":
                        family.get(
                            "occurrences_b32",
                            math.nan,
                        ),

                    "b33_mapped_occurrences":
                        0,

                    "unique_dates":
                        0,

                    "unique_years":
                        0,

                    "first_seen":
                        "",

                    "last_seen":
                        "",

                    "median_gap_minutes":
                        math.nan,

                    "min_gap_minutes":
                        math.nan,

                    "max_gap_minutes":
                        math.nan,

                    "clock_pairs":
                        0,

                    "classification_b32":
                        family.get(
                            "classification",
                            "",
                        ),
                }
            )

            continue

        clock_pairs = (
            subset[
                [
                    "previous_clock",
                    "current_clock",
                ]
            ]
            .drop_duplicates()
        )

        rows.append(
            {
                "family_id":
                    family_id,

                "b31_occurrences":
                    family.get(
                        "occurrences_b31",
                        math.nan,
                    ),

                "b32_occurrences":
                    family.get(
                        "occurrences_b32",
                        math.nan,
                    ),

                "b33_mapped_occurrences":
                    len(subset),

                "unique_dates":
                    subset[
                        "previous_date"
                    ].nunique(),

                "unique_years":
                    subset[
                        "previous_year"
                    ].nunique(),

                "first_seen":
                    subset[
                        "previous"
                    ].min(),

                "last_seen":
                    subset[
                        "previous"
                    ].max(),

                "median_gap_minutes":
                    subset[
                        "gap_minutes"
                    ].median(),

                "min_gap_minutes":
                    subset[
                        "gap_minutes"
                    ].min(),

                "max_gap_minutes":
                    subset[
                        "gap_minutes"
                    ].max(),

                "clock_pairs":
                    len(clock_pairs),

                "classification_b32":
                    family.get(
                        "classification",
                        "",
                    ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# REPORT
# ============================================================

def write_report(
    mapped,
    summary,
    yearly,
    monthly,
    weekday,
    shifts,
    regimes,
):
    """Write human-readable B3.3 report."""

    report_path = (
        OUTPUT_DIR
        / "B33_REPORT.txt"
    )

    lines = []

    lines.append(
        "EMA-ADX TRADING SYSTEM - V3.0"
    )

    lines.append(
        "B3.3 HISTORICAL SESSION BOUNDARY MAPPER"
    )

    lines.append(
        "INSTRUMENT: XAUUSDm"
    )

    lines.append("")
    lines.append(
        "=" * 70
    )

    lines.append(
        "PURPOSE"
    )

    lines.append(
        "Map existing B3.1/B3.2 recurring boundary families "
        "through the full XAUUSD M1 history."
    )

    lines.append("")

    lines.append(
        "IMPORTANT:"
    )

    lines.append(
        "B3.3 reports historical clock behavior."
    )

    lines.append(
        "It does NOT declare a final session rule."
    )

    lines.append(
        "It does NOT automatically identify DST."
    )

    lines.append(
        "It does NOT modify session_validator.py."
    )

    lines.append("")

    lines.append(
        "=" * 70
    )

    lines.append(
        "SUMMARY BY FAMILY"
    )

    lines.append(
        "=" * 70
    )

    if summary.empty:

        lines.append(
            "No mapped boundary occurrences."
        )

    else:

        for _, row in summary.iterrows():

            lines.append(
                f"\n{row['family_id']}"
            )

            lines.append(
                f"  B3.1 occurrences: "
                f"{row['b31_occurrences']}"
            )

            lines.append(
                f"  B3.2 occurrences: "
                f"{row['b32_occurrences']}"
            )

            lines.append(
                f"  B3.3 mapped: "
                f"{row['b33_mapped_occurrences']}"
            )

            lines.append(
                f"  Unique dates: "
                f"{row['unique_dates']}"
            )

            lines.append(
                f"  Years: "
                f"{row['unique_years']}"
            )

            lines.append(
                f"  First seen: "
                f"{row['first_seen']}"
            )

            lines.append(
                f"  Last seen: "
                f"{row['last_seen']}"
            )

            lines.append(
                f"  Gap median: "
                f"{row['median_gap_minutes']}"
            )

            lines.append(
                f"  Gap range: "
                f"{row['min_gap_minutes']} - "
                f"{row['max_gap_minutes']}"
            )

            lines.append(
                f"  Distinct clock pairs: "
                f"{row['clock_pairs']}"
            )

    lines.append("")
    lines.append(
        "=" * 70
    )

    lines.append(
        "YEARLY OBSERVATIONS"
    )

    lines.append(
        "=" * 70
    )

    if yearly.empty:

        lines.append(
            "No yearly observations."
        )

    else:

        for _, row in yearly.iterrows():

            lines.append(
                f"{int(row['year'])} | "
                f"{row['family_id']} | "
                f"{int(row['occurrences'])} occurrences | "
                f"{row['median_previous_clock']} -> "
                f"{row['median_current_clock']} | "
                f"gap={row['median_gap_minutes']:.2f}"
            )

    lines.append("")
    lines.append(
        "=" * 70
    )

    lines.append(
        "OBSERVED CLOCK SHIFT POINTS"
    )

    lines.append(
        "=" * 70
    )

    if shifts.empty:

        lines.append(
            "No clock-pair shifts detected."
        )

    else:

        for _, row in shifts.iterrows():

            lines.append(
                f"{row['family_id']} | "
                f"{row['shift_date']} | "
                f"{row['previous_observed_pair']} "
                f"-> "
                f"{row['new_observed_pair']}"
            )

    lines.append("")
    lines.append(
        "=" * 70
    )

    lines.append(
        "CALENDAR REGIMES"
    )

    lines.append(
        "=" * 70
    )

    if regimes.empty:

        lines.append(
            "No qualifying calendar regimes."
        )

    else:

        for _, row in regimes.iterrows():

            lines.append(
                f"{row['regime_id']} | "
                f"{row['start_date']} -> "
                f"{row['end_date']} | "
                f"{row['observed_previous_clock']} -> "
                f"{row['observed_current_clock']} | "
                f"{row['occurrences']} occurrences"
            )

    lines.append("")
    lines.append(
        "=" * 70
    )

    lines.append(
        "INTERPRETATION"
    )

    lines.append(
        "=" * 70
    )

    lines.append(
        "The mapped families are historical evidence only."
    )

    lines.append(
        "A clock shift does not automatically prove DST."
    )

    lines.append(
        "The next step is to determine whether the observed "
        "regimes are stable enough to form production "
        "session-validation rules."
    )

    lines.append(
        "No session_validator.py changes should be made "
        "from B3.3 alone."
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return report_path


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print("\n")
    print("=" * 70)
    print(
        "EMA-ADX TRADING SYSTEM - V3.0"
    )
    print(
        "B3.3 HISTORICAL SESSION BOUNDARY MAPPER"
    )
    print(
        "XAUUSDm ONLY"
    )
    print("=" * 70)

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    m1 = load_xauusd_m1()

    # --------------------------------------------------------
    # Build all raw gaps
    # --------------------------------------------------------

    gaps = build_gap_table(m1)

    # --------------------------------------------------------
    # Load existing family evidence
    # --------------------------------------------------------

    families = load_existing_families()

    # --------------------------------------------------------
    # Map families
    # --------------------------------------------------------

    mapped = map_boundary_families(
        gaps,
        families,
    )

    # --------------------------------------------------------
    # Generate outputs
    # --------------------------------------------------------

    daily = build_daily_map(
        mapped
    )

    yearly = build_yearly_map(
        mapped
    )

    monthly = build_monthly_map(
        mapped
    )

    weekday = build_weekday_map(
        mapped
    )

    shifts = build_clock_shift_points(
        mapped
    )

    regimes = build_calendar_regimes(
        mapped
    )

    summary = build_summary(
        mapped,
        families,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    mapped.to_csv(
        OUTPUT_DIR
        / "boundary_occurrences.csv",
        index=False,
    )

    daily.to_csv(
        OUTPUT_DIR
        / "daily_boundary_map.csv",
        index=False,
    )

    yearly.to_csv(
        OUTPUT_DIR
        / "yearly_boundary_map.csv",
        index=False,
    )

    monthly.to_csv(
        OUTPUT_DIR
        / "monthly_boundary_map.csv",
        index=False,
    )

    weekday.to_csv(
        OUTPUT_DIR
        / "weekday_boundary_map.csv",
        index=False,
    )

    shifts.to_csv(
        OUTPUT_DIR
        / "observed_clock_shift_points.csv",
        index=False,
    )

    regimes.to_csv(
        OUTPUT_DIR
        / "calendar_regime_map.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "b33_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report_path = write_report(
        mapped,
        summary,
        yearly,
        monthly,
        weekday,
        shifts,
        regimes,
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    elapsed = (
        time.time()
        - start_time
    )

    print("\n")
    print("=" * 70)
    print("B3.3 COMPLETE")
    print("=" * 70)

    print(
        f"XAUUSD M1 rows: "
        f"{len(m1):,}"
    )

    print(
        f"Raw gaps > 1 minute: "
        f"{len(gaps):,}"
    )

    print(
        f"B3.3 mapped occurrences: "
        f"{len(mapped):,}"
    )

    print("\nMapped occurrences by family:")

    if mapped.empty:

        print("  NONE")

    else:

        print(
            mapped[
                "family_id"
            ]
            .value_counts()
            .to_string()
        )

    print(
        f"\nClock shift points: "
        f"{len(shifts):,}"
    )

    print(
        f"Calendar regimes: "
        f"{len(regimes):,}"
    )

    print(
        f"\nReport:"
    )

    print(
        f"  {report_path}"
    )

    print(
        f"\nElapsed time: "
        f"{elapsed:.2f} seconds"
    )

    print("\nIMPORTANT:")
    print(
        "B3.3 is evidence only."
    )

    print(
        "DO NOT modify session_validator.py yet."
    )

    print(
        "Review the generated outputs before "
        "finalizing the XAUUSD session model."
    )


if __name__ == "__main__":
    main()