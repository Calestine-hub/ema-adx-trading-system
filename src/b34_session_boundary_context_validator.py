"""
EMA-ADX TRADING SYSTEM - V3.0

B3.4 - SESSION BOUNDARY CONTEXT VALIDATOR
XAUUSD ONLY

FINAL DIAGNOSTIC STAGE BEFORE PRODUCTION SESSION MODEL

Purpose
-------
Validate the actual M1 candle context surrounding the recurring
boundary families identified by B3.1, B3.2 and B3.3.

B3.4 investigates:

    1. P30_C35_D64
       ~64-minute recurring boundary

    2. P33_C39_D64
       ~64-minute recurring boundary with historical clock shift

    3. P41_C1_D2
       23:58 -> 00:00 recurring 2-minute gap

    4. Unclassified gaps
       Raw gaps that were NOT mapped to the known B3.3 families.

For each mapped boundary, the script examines:

    - 5 M1 candles before the gap
    - the exact gap
    - 5 M1 candles after the gap
    - timestamp continuity
    - gap duration
    - candle availability
    - OHLC context
    - weekday
    - month
    - year
    - observed clock regime

The purpose is NOT to decide trading validity.

The purpose is to determine whether the recurring boundaries
behave differently from ordinary/unclassified data gaps.

IMPORTANT
---------
This script:

    - does NOT modify raw M1 data
    - does NOT fill missing candles
    - does NOT construct 15M candles
    - does NOT construct 4H candles
    - does NOT modify session_validator.py
    - does NOT create trading signals
    - does NOT use M1 as a trading timeframe
    - does NOT automatically declare DST
    - does NOT automatically create production session rules

B3.4 is evidence only.

After B3.4 is reviewed, the production XAUUSD session model
can be defined and session_validator.py can then be updated.
"""

from pathlib import Path
import time

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw"

RAW_FILE = (
    DATA_DIR
    / "XAUUSDm_M1_202108170000_202608131827.csv"
)

B33_DIR = (
    PROJECT_ROOT
    / "data"
    / "session_mapping_b33"
    / "XAUUSDm"
)

B33_OCCURRENCES_FILE = (
    B33_DIR
    / "boundary_occurrences.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "session_context_b34"
    / "XAUUSDm"
)


# ============================================================
# SETTINGS
# ============================================================

SYMBOL = "XAUUSDm"

CONTEXT_CANDLES = 5

# Unclassified gaps are the control group.
#
# We inspect ALL unclassified gaps rather than randomly
# sampling them because the dataset is manageable.
#
# This gives us the strongest comparison between known
# recurring boundaries and everything else.

MAX_UNCLASSIFIED_CONTEXTS = None


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


def minute_of_day(timestamp):
    """Return timestamp as minute-of-day."""

    return (
        timestamp.hour * 60
        + timestamp.minute
    )


def clock_string(timestamp):
    """Return HH:MM."""

    return timestamp.strftime("%H:%M")


def safe_float(value):
    """Convert numeric value safely."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ============================================================
# LOAD RAW M1
# ============================================================

def load_xauusd_m1():
    """Load existing XAUUSD M1 data."""

    print("\n" + "=" * 70)
    print("B3.4 - LOADING XAUUSD M1 DATA")
    print("=" * 70)

    if not RAW_FILE.exists():

        raise FileNotFoundError(
            f"Raw XAUUSD file not found:\n{RAW_FILE}"
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
            f"timestamps found."
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
        f"Last candle: {df['Datetime'].iloc[-1]}"
    )

    return df


# ============================================================
# BUILD RAW GAP INDEX
# ============================================================

def build_raw_gap_table(m1):
    """
    Identify every raw M1 timestamp gap > 1 minute.

    The raw dataset remains untouched.
    """

    print("\nBuilding raw gap index...")

    timestamps = m1["Datetime"]

    differences = timestamps.diff()

    gap_indices = differences.index[
        differences > pd.Timedelta(minutes=1)
    ]

    rows = []

    for index in gap_indices:

        previous_row = m1.iloc[index - 1]
        current_row = m1.iloc[index]

        previous_time = (
            previous_row["Datetime"]
        )

        current_time = (
            current_row["Datetime"]
        )

        gap_minutes = (
            current_time
            - previous_time
        ).total_seconds() / 60

        rows.append(
            {
                "gap_index": index,

                "previous":
                    previous_time,

                "current":
                    current_time,

                "gap_minutes":
                    gap_minutes,

                "previous_date":
                    previous_time.date(),

                "current_date":
                    current_time.date(),

                "year":
                    previous_time.year,

                "month":
                    previous_time.month,

                "weekday":
                    previous_time.day_name(),

                "previous_clock":
                    clock_string(previous_time),

                "current_clock":
                    clock_string(current_time),
            }
        )

    gaps = pd.DataFrame(rows)

    print(
        f"Raw gaps > 1 minute: {len(gaps):,}"
    )

    return gaps


# ============================================================
# LOAD B3.3 OCCURRENCES
# ============================================================

def load_b33_occurrences():
    """Load actual B3.3 mapped boundary occurrences."""

    print("\nLoading B3.3 boundary occurrences...")

    if not B33_OCCURRENCES_FILE.exists():

        raise FileNotFoundError(
            "B3.3 occurrence file not found:\n"
            f"{B33_OCCURRENCES_FILE}"
        )

    mapped = pd.read_csv(
        B33_OCCURRENCES_FILE
    )

    mapped.columns = (
        mapped.columns
        .astype(str)
        .str.strip()
    )

    required = [
        "family_id",
        "previous",
        "current",
        "gap_minutes",
    ]

    missing = [
        column
        for column in required
        if column not in mapped.columns
    ]

    if missing:

        raise ValueError(
            "B3.3 occurrence file is missing: "
            + ", ".join(missing)
        )

    mapped["previous"] = pd.to_datetime(
        mapped["previous"]
    )

    mapped["current"] = pd.to_datetime(
        mapped["current"]
    )

    print(
        f"B3.3 mapped occurrences: "
        f"{len(mapped):,}"
    )

    print("\nOccurrences by family:")

    print(
        mapped["family_id"]
        .value_counts()
        .to_string()
    )

    return mapped


# ============================================================
# CREATE RAW TIMESTAMP LOOKUP
# ============================================================

def create_timestamp_lookup(m1):
    """
    Create timestamp -> positional index lookup.

    This allows fast context retrieval.
    """

    return {
        timestamp: index
        for index, timestamp
        in enumerate(m1["Datetime"])
    }


# ============================================================
# CONTEXT EXTRACTION
# ============================================================

def extract_context(
    m1,
    timestamp_lookup,
    gap_row,
    context_type,
    family_id=None,
):
    """
    Extract N candles immediately before and after a gap.

    The gap itself is represented by the difference between
    the last available timestamp before it and the first
    available timestamp after it.

    No synthetic candles are created.
    """

    previous_time = pd.Timestamp(
        gap_row["previous"]
    )

    current_time = pd.Timestamp(
        gap_row["current"]
    )

    if (
        previous_time
        not in timestamp_lookup
    ):

        return None

    if (
        current_time
        not in timestamp_lookup
    ):

        return None

    previous_index = (
        timestamp_lookup[
            previous_time
        ]
    )

    current_index = (
        timestamp_lookup[
            current_time
        ]
    )

    before_start = max(
        0,
        previous_index
        - CONTEXT_CANDLES
        + 1,
    )

    before = m1.iloc[
        before_start:
        previous_index + 1
    ]

    after_end = min(
        len(m1),
        current_index
        + CONTEXT_CANDLES,
    )

    after = m1.iloc[
        current_index:
        after_end
    ]

    # --------------------------------------------------------
    # Validate context availability
    # --------------------------------------------------------

    before_count = len(before)
    after_count = len(after)

    full_before = (
        before_count
        == CONTEXT_CANDLES
    )

    full_after = (
        after_count
        == CONTEXT_CANDLES
    )

    # --------------------------------------------------------
    # Check expected one-minute continuity INSIDE context
    # --------------------------------------------------------

    before_differences = (
        before["Datetime"].diff()
        .dropna()
    )

    after_differences = (
        after["Datetime"].diff()
        .dropna()
    )

    before_internal_gaps = (
        before_differences
        > pd.Timedelta(minutes=1)
    ).sum()

    after_internal_gaps = (
        after_differences
        > pd.Timedelta(minutes=1)
    ).sum()

    # --------------------------------------------------------
    # Context OHLC values
    # --------------------------------------------------------

    first_before = before.iloc[0]
    last_before = before.iloc[-1]

    first_after = after.iloc[0]
    last_after = after.iloc[-1]

    gap_minutes = (
        current_time
        - previous_time
    ).total_seconds() / 60

    return {
        "context_type":
            context_type,

        "family_id":
            family_id,

        "gap_previous":
            previous_time,

        "gap_current":
            current_time,

        "gap_minutes":
            gap_minutes,

        "date":
            previous_time.date(),

        "year":
            previous_time.year,

        "month":
            previous_time.month,

        "weekday":
            previous_time.day_name(),

        "previous_clock":
            clock_string(previous_time),

        "current_clock":
            clock_string(current_time),

        "before_count":
            before_count,

        "after_count":
            after_count,

        "full_before_context":
            full_before,

        "full_after_context":
            full_after,

        "before_internal_gaps":
            before_internal_gaps,

        "after_internal_gaps":
            after_internal_gaps,

        "context_is_continuous":
            (
                before_internal_gaps == 0
                and
                after_internal_gaps == 0
            ),

        "first_before_time":
            first_before["Datetime"],

        "last_before_time":
            last_before["Datetime"],

        "first_after_time":
            first_after["Datetime"],

        "last_after_time":
            last_after["Datetime"],

        "first_before_open":
            safe_float(
                first_before["<OPEN>"]
            ),

        "last_before_close":
            safe_float(
                last_before["<CLOSE>"]
            ),

        "first_after_open":
            safe_float(
                first_after["<OPEN>"]
            ),

        "last_after_close":
            safe_float(
                last_after["<CLOSE>"]
            ),

        "before_high":
            safe_float(
                before["<HIGH>"].max()
            ),

        "before_low":
            safe_float(
                before["<LOW>"].min()
            ),

        "after_high":
            safe_float(
                after["<HIGH>"].max()
            ),

        "after_low":
            safe_float(
                after["<LOW>"].min()
            ),

        "before_close_to_after_open_change":
            (
                safe_float(
                    first_after["<OPEN>"]
                )
                -
                safe_float(
                    last_before["<CLOSE>"]
                )
            ),
    }


# ============================================================
# VALIDATE KNOWN BOUNDARIES
# ============================================================

def validate_known_boundaries(
    m1,
    timestamp_lookup,
    mapped,
):
    """Extract context for every B3.3 boundary."""

    print("\nValidating known boundary contexts...")

    rows = []

    for index, row in mapped.iterrows():

        context = extract_context(
            m1=m1,
            timestamp_lookup=timestamp_lookup,
            gap_row=row,
            context_type="KNOWN_BOUNDARY",
            family_id=row["family_id"],
        )

        if context is None:

            print(
                f"WARNING: Could not extract context "
                f"for mapped occurrence {index}"
            )

            continue

        rows.append(context)

    result = pd.DataFrame(rows)

    print(
        f"Known boundary contexts validated: "
        f"{len(result):,}"
    )

    return result


# ============================================================
# IDENTIFY UNCLASSIFIED GAPS
# ============================================================

def identify_unclassified_gaps(
    raw_gaps,
    mapped,
):
    """
    Identify raw gaps that are not represented in B3.3.

    Matching is based on exact previous/current timestamps.
    """

    print("\nIdentifying unclassified gaps...")

    if raw_gaps.empty:

        return raw_gaps.copy()

    known_keys = set(
        zip(
            mapped["previous"],
            mapped["current"],
        )
    )

    raw_gaps = raw_gaps.copy()

    raw_gaps["is_known_family"] = [
        (
            previous,
            current,
        ) in known_keys
        for previous, current
        in zip(
            raw_gaps["previous"],
            raw_gaps["current"],
        )
    ]

    unclassified = raw_gaps[
        ~raw_gaps["is_known_family"]
    ].copy()

    unclassified = (
        unclassified
        .drop(columns=["is_known_family"])
        .reset_index(drop=True)
    )

    print(
        f"Unclassified gaps: "
        f"{len(unclassified):,}"
    )

    return unclassified


# ============================================================
# VALIDATE UNCLASSIFIED GAPS
# ============================================================

def validate_unclassified_boundaries(
    m1,
    timestamp_lookup,
    unclassified,
):
    """
    Extract context around all unclassified gaps.

    This forms the control group.
    """

    print(
        "\nValidating unclassified gap contexts..."
    )

    if (
        MAX_UNCLASSIFIED_CONTEXTS
        is not None
    ):

        unclassified = (
            unclassified
            .head(
                MAX_UNCLASSIFIED_CONTEXTS
            )
        )

    rows = []

    for index, row in unclassified.iterrows():

        context = extract_context(
            m1=m1,
            timestamp_lookup=timestamp_lookup,
            gap_row=row,
            context_type="UNCLASSIFIED",
            family_id=None,
        )

        if context is None:

            continue

        rows.append(context)

    result = pd.DataFrame(rows)

    print(
        f"Unclassified contexts validated: "
        f"{len(result):,}"
    )

    return result


# ============================================================
# KNOWN FAMILY SUMMARY
# ============================================================

def build_family_context_summary(
    known,
):
    """Summarize context behavior for each known family."""

    if known.empty:
        return pd.DataFrame()

    summary = (
        known
        .groupby(
            "family_id",
            as_index=False,
        )
        .agg(
            occurrences=(
                "family_id",
                "size",
            ),

            unique_dates=(
                "date",
                "nunique",
            ),

            years=(
                "year",
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

            full_before_context_pct=(
                "full_before_context",
                "mean",
            ),

            full_after_context_pct=(
                "full_after_context",
                "mean",
            ),

            continuous_context_pct=(
                "context_is_continuous",
                "mean",
            ),

            internal_before_gaps=(
                "before_internal_gaps",
                "sum",
            ),

            internal_after_gaps=(
                "after_internal_gaps",
                "sum",
            ),

            median_before_after_price_change=(
                "before_close_to_after_open_change",
                "median",
            ),
        )
    )

    summary[
        "full_before_context_pct"
    ] *= 100

    summary[
        "full_after_context_pct"
    ] *= 100

    summary[
        "continuous_context_pct"
    ] *= 100

    return summary


# ============================================================
# CONTROL GROUP SUMMARY
# ============================================================

def build_unclassified_summary(
    unclassified,
):
    """Summarize all unclassified gaps."""

    if unclassified.empty:
        return pd.DataFrame()

    summary = {
        "category":
            "UNCLASSIFIED",

        "occurrences":
            len(unclassified),

        "unique_dates":
            unclassified[
                "date"
            ].nunique(),

        "years":
            unclassified[
                "year"
            ].nunique(),

        "median_gap_minutes":
            unclassified[
                "gap_minutes"
            ].median(),

        "min_gap_minutes":
            unclassified[
                "gap_minutes"
            ].min(),

        "max_gap_minutes":
            unclassified[
                "gap_minutes"
            ].max(),

        "full_before_context_pct":
            unclassified[
                "full_before_context"
            ].mean() * 100,

        "full_after_context_pct":
            unclassified[
                "full_after_context"
            ].mean() * 100,

        "continuous_context_pct":
            unclassified[
                "context_is_continuous"
            ].mean() * 100,

        "internal_before_gaps":
            unclassified[
                "before_internal_gaps"
            ].sum(),

        "internal_after_gaps":
            unclassified[
                "after_internal_gaps"
            ].sum(),
    }

    return pd.DataFrame(
        [summary]
    )


# ============================================================
# MONTHLY CONTEXT SUMMARY
# ============================================================

def build_monthly_context_summary(
    known,
):
    """Show family behavior across calendar months."""

    if known.empty:
        return pd.DataFrame()

    working = known.copy()

    working["month"] = (
        pd.to_datetime(
            working["date"]
        ).dt.month
    )

    working["year"] = (
        pd.to_datetime(
            working["date"]
        ).dt.year
    )

    summary = (
        working
        .groupby(
            [
                "family_id",
                "year",
                "month",
            ],
            as_index=False,
        )
        .agg(
            occurrences=(
                "family_id",
                "size",
            ),

            median_gap_minutes=(
                "gap_minutes",
                "median",
            ),

            median_previous_minute=(
                "previous_clock",
                lambda values:
                pd.Series(
                    [
                        int(
                            str(value)[0:2]
                        ) * 60
                        +
                        int(
                            str(value)[3:5]
                        )
                        for value in values
                    ]
                ).median(),
            ),

            median_current_minute=(
                "current_clock",
                lambda values:
                pd.Series(
                    [
                        int(
                            str(value)[0:2]
                        ) * 60
                        +
                        int(
                            str(value)[3:5]
                        )
                        for value in values
                    ]
                ).median(),
            ),

            continuous_context_pct=(
                "context_is_continuous",
                "mean",
            ),
        )
    )

    summary[
        "continuous_context_pct"
    ] *= 100

    return summary.sort_values(
        [
            "family_id",
            "year",
            "month",
        ]
    ).reset_index(drop=True)


# ============================================================
# WEEKDAY CONTEXT SUMMARY
# ============================================================

def build_weekday_context_summary(
    known,
):
    """Show family behavior by weekday."""

    if known.empty:
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

    summary = (
        known
        .groupby(
            [
                "family_id",
                "weekday",
            ],
            as_index=False,
        )
        .agg(
            occurrences=(
                "family_id",
                "size",
            ),

            median_gap_minutes=(
                "gap_minutes",
                "median",
            ),

            continuous_context_pct=(
                "context_is_continuous",
                "mean",
            ),
        )
    )

    summary[
        "continuous_context_pct"
    ] *= 100

    summary["weekday"] = pd.Categorical(
        summary["weekday"],
        categories=weekday_order,
        ordered=True,
    )

    return summary.sort_values(
        [
            "family_id",
            "weekday",
        ]
    ).reset_index(drop=True)


# ============================================================
# CONTEXT PATTERN SUMMARY
# ============================================================

def build_context_pattern_summary(
    known,
):
    """
    Determine how many distinct observed boundary
    clock transitions exist for each family.
    """

    if known.empty:
        return pd.DataFrame()

    summary = (
        known
        .groupby(
            [
                "family_id",
                "previous_clock",
                "current_clock",
            ],
            as_index=False,
        )
        .agg(
            occurrences=(
                "family_id",
                "size",
            ),

            first_seen=(
                "gap_previous",
                "min",
            ),

            last_seen=(
                "gap_previous",
                "max",
            ),

            median_gap_minutes=(
                "gap_minutes",
                "median",
            ),

            continuous_context_pct=(
                "context_is_continuous",
                "mean",
            ),
        )
    )

    summary[
        "continuous_context_pct"
    ] *= 100

    return summary.sort_values(
        [
            "family_id",
            "occurrences",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(drop=True)


# ============================================================
# BUILD REPORT
# ============================================================

def write_report(
    known,
    unclassified,
    family_summary,
    unclassified_summary,
    monthly_summary,
    weekday_summary,
    pattern_summary,
):
    """Write final B3.4 human-readable report."""

    report_path = (
        OUTPUT_DIR
        / "session_boundary_validation_report.txt"
    )

    lines = []

    lines.append(
        "EMA-ADX TRADING SYSTEM - V3.0"
    )

    lines.append(
        "B3.4 SESSION BOUNDARY CONTEXT VALIDATION"
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
        "Validate M1 candle context around known recurring "
        "boundary families and compare them with unclassified gaps."
    )

    lines.append("")

    lines.append(
        "B3.4 IS EVIDENCE ONLY."
    )

    lines.append(
        "No production session rule is created by this script."
    )

    lines.append(
        "session_validator.py is not modified."
    )

    lines.append("")

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    lines.append(
        "=" * 70
    )

    lines.append(
        "OVERALL"
    )

    lines.append(
        "=" * 70
    )

    lines.append(
        f"Known boundary contexts: "
        f"{len(known):,}"
    )

    lines.append(
        f"Unclassified gap contexts: "
        f"{len(unclassified):,}"
    )

    if not known.empty:

        lines.append(
            f"Known contexts with full 5-candle "
            f"before context: "
            f"{known['full_before_context'].mean() * 100:.2f}%"
        )

        lines.append(
            f"Known contexts with full 5-candle "
            f"after context: "
            f"{known['full_after_context'].mean() * 100:.2f}%"
        )

        lines.append(
            f"Known contexts with continuous "
            f"surrounding data: "
            f"{known['context_is_continuous'].mean() * 100:.2f}%"
        )

    if not unclassified.empty:

        lines.append(
            f"Unclassified contexts with full 5-candle "
            f"before context: "
            f"{unclassified['full_before_context'].mean() * 100:.2f}%"
        )

        lines.append(
            f"Unclassified contexts with full 5-candle "
            f"after context: "
            f"{unclassified['full_after_context'].mean() * 100:.2f}%"
        )

        lines.append(
            f"Unclassified contexts with continuous "
            f"surrounding data: "
            f"{unclassified['context_is_continuous'].mean() * 100:.2f}%"
        )

    # --------------------------------------------------------
    # Families
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "=" * 70
    )

    lines.append(
        "KNOWN FAMILY CONTEXT"
    )

    lines.append(
        "=" * 70
    )

    if family_summary.empty:

        lines.append(
            "No known family context available."
        )

    else:

        for _, row in family_summary.iterrows():

            lines.append("")

            lines.append(
                f"Family: {row['family_id']}"
            )

            lines.append(
                f"  Occurrences: "
                f"{int(row['occurrences'])}"
            )

            lines.append(
                f"  Unique dates: "
                f"{int(row['unique_dates'])}"
            )

            lines.append(
                f"  Years: "
                f"{int(row['years'])}"
            )

            lines.append(
                f"  Gap median: "
                f"{row['median_gap_minutes']:.2f} min"
            )

            lines.append(
                f"  Gap range: "
                f"{row['min_gap_minutes']:.2f} - "
                f"{row['max_gap_minutes']:.2f} min"
            )

            lines.append(
                f"  Full before context: "
                f"{row['full_before_context_pct']:.2f}%"
            )

            lines.append(
                f"  Full after context: "
                f"{row['full_after_context_pct']:.2f}%"
            )

            lines.append(
                f"  Continuous context: "
                f"{row['continuous_context_pct']:.2f}%"
            )

            lines.append(
                f"  Internal before gaps: "
                f"{int(row['internal_before_gaps'])}"
            )

            lines.append(
                f"  Internal after gaps: "
                f"{int(row['internal_after_gaps'])}"
            )

    # --------------------------------------------------------
    # Control
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "=" * 70
    )

    lines.append(
        "UNCLASSIFIED CONTROL GROUP"
    )

    lines.append(
        "=" * 70
    )

    if unclassified_summary.empty:

        lines.append(
            "No unclassified contexts."
        )

    else:

        row = unclassified_summary.iloc[0]

        lines.append(
            f"Occurrences: "
            f"{int(row['occurrences'])}"
        )

        lines.append(
            f"Unique dates: "
            f"{int(row['unique_dates'])}"
        )

        lines.append(
            f"Years: "
            f"{int(row['years'])}"
        )

        lines.append(
            f"Gap median: "
            f"{row['median_gap_minutes']:.2f} min"
        )

        lines.append(
            f"Gap range: "
            f"{row['min_gap_minutes']:.2f} - "
            f"{row['max_gap_minutes']:.2f} min"
        )

        lines.append(
            f"Full before context: "
            f"{row['full_before_context_pct']:.2f}%"
        )

        lines.append(
            f"Full after context: "
            f"{row['full_after_context_pct']:.2f}%"
        )

        lines.append(
            f"Continuous context: "
            f"{row['continuous_context_pct']:.2f}%"
        )

    # --------------------------------------------------------
    # Conclusion guidance
    # --------------------------------------------------------

    lines.append("")
    lines.append(
        "=" * 70
    )

    lines.append(
        "B3.4 INTERPRETATION GUIDANCE"
    )

    lines.append(
        "=" * 70
    )

    lines.append(
        "1. High recurrence alone does not make a gap valid."
    )

    lines.append(
        "2. Exact clock times should not be hard-coded without "
        "historical tolerance."
    )

    lines.append(
        "3. P30 and P33 should be evaluated together because "
        "their historical clock regimes differ by approximately "
        "one hour."
    )

    lines.append(
        "4. P41 should remain a separate phenomenon until its "
        "context establishes otherwise."
    )

    lines.append(
        "5. Unclassified gaps provide the control group."
    )

    lines.append(
        "6. No data should be forward-filled or invented."
    )

    lines.append(
        "7. The final production validator should preserve "
        "unexpected gaps rather than silently deleting them."
    )

    lines.append(
        "8. The final session model should be evidence-based "
        "and should not overfit exact historical timestamps."
    )

    lines.append("")

    lines.append(
        "NEXT STAGE:"
    )

    lines.append(
        "Review B3.4 results and define the final XAUUSD "
        "session model before modifying session_validator.py."
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
        "B3.4 SESSION BOUNDARY CONTEXT VALIDATOR"
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
    # Raw gaps
    # --------------------------------------------------------

    raw_gaps = build_raw_gap_table(
        m1
    )

    # --------------------------------------------------------
    # B3.3 mapped families
    # --------------------------------------------------------

    mapped = load_b33_occurrences()

    # --------------------------------------------------------
    # Timestamp lookup
    # --------------------------------------------------------

    print(
        "\nBuilding timestamp lookup..."
    )

    timestamp_lookup = (
        create_timestamp_lookup(m1)
    )

    print(
        f"Timestamp entries: "
        f"{len(timestamp_lookup):,}"
    )

    # --------------------------------------------------------
    # Known family context
    # --------------------------------------------------------

    known_context = (
        validate_known_boundaries(
            m1=m1,
            timestamp_lookup=timestamp_lookup,
            mapped=mapped,
        )
    )

    # --------------------------------------------------------
    # Unclassified gaps
    # --------------------------------------------------------

    unclassified_gaps = (
        identify_unclassified_gaps(
            raw_gaps=raw_gaps,
            mapped=mapped,
        )
    )

    # --------------------------------------------------------
    # Unclassified context
    # --------------------------------------------------------

    unclassified_context = (
        validate_unclassified_boundaries(
            m1=m1,
            timestamp_lookup=timestamp_lookup,
            unclassified=unclassified_gaps,
        )
    )

    # --------------------------------------------------------
    # Summaries
    # --------------------------------------------------------

    family_summary = (
        build_family_context_summary(
            known_context
        )
    )

    unclassified_summary = (
        build_unclassified_summary(
            unclassified_context
        )
    )

    monthly_summary = (
        build_monthly_context_summary(
            known_context
        )
    )

    weekday_summary = (
        build_weekday_context_summary(
            known_context
        )
    )

    pattern_summary = (
        build_context_pattern_summary(
            known_context
        )
    )

    # --------------------------------------------------------
    # Save detailed outputs
    # --------------------------------------------------------

    print(
        "\nWriting B3.4 outputs..."
    )

    known_context.to_csv(
        OUTPUT_DIR
        / "boundary_context_occurrences.csv",
        index=False,
    )

    unclassified_context.to_csv(
        OUTPUT_DIR
        / "unclassified_gap_context.csv",
        index=False,
    )

    family_summary.to_csv(
        OUTPUT_DIR
        / "family_context_summary.csv",
        index=False,
    )

    unclassified_summary.to_csv(
        OUTPUT_DIR
        / "unclassified_context_summary.csv",
        index=False,
    )

    monthly_summary.to_csv(
        OUTPUT_DIR
        / "monthly_context_summary.csv",
        index=False,
    )

    weekday_summary.to_csv(
        OUTPUT_DIR
        / "weekday_context_summary.csv",
        index=False,
    )

    pattern_summary.to_csv(
        OUTPUT_DIR
        / "context_clock_pattern_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report_path = write_report(
        known=known_context,
        unclassified=unclassified_context,
        family_summary=family_summary,
        unclassified_summary=unclassified_summary,
        monthly_summary=monthly_summary,
        weekday_summary=weekday_summary,
        pattern_summary=pattern_summary,
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
    print("B3.4 COMPLETE")
    print("=" * 70)

    print(
        f"XAUUSD M1 rows: "
        f"{len(m1):,}"
    )

    print(
        f"Raw gaps > 1 minute: "
        f"{len(raw_gaps):,}"
    )

    print(
        f"Known boundary contexts: "
        f"{len(known_context):,}"
    )

    print(
        f"Unclassified gap contexts: "
        f"{len(unclassified_context):,}"
    )

    print("\nKnown boundary contexts by family:")

    if known_context.empty:

        print("  NONE")

    else:

        print(
            known_context[
                "family_id"
            ]
            .value_counts()
            .to_string()
        )

    if not known_context.empty:

        print(
            "\nKnown boundary context continuity:"
        )

        print(
            f"  Full before context: "
            f"{known_context['full_before_context'].mean() * 100:.2f}%"
        )

        print(
            f"  Full after context: "
            f"{known_context['full_after_context'].mean() * 100:.2f}%"
        )

        print(
            f"  Continuous surrounding context: "
            f"{known_context['context_is_continuous'].mean() * 100:.2f}%"
        )

    if not unclassified_context.empty:

        print(
            "\nUnclassified control continuity:"
        )

        print(
            f"  Full before context: "
            f"{unclassified_context['full_before_context'].mean() * 100:.2f}%"
        )

        print(
            f"  Full after context: "
            f"{unclassified_context['full_after_context'].mean() * 100:.2f}%"
        )

        print(
            f"  Continuous surrounding context: "
            f"{unclassified_context['context_is_continuous'].mean() * 100:.2f}%"
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
        "B3.4 is the final diagnostic stage."
    )

    print(
        "Do NOT modify session_validator.py yet."
    )

    print(
        "Review the B3.4 evidence before defining "
        "the production XAUUSD session model."
    )


if __name__ == "__main__":
    main()