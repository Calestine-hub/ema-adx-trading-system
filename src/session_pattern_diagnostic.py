
"""
EMA-ADX TRADING SYSTEM - V3.0
BROKER SESSION PATTERN DIAGNOSTIC

Purpose:
    Analyze recurring timestamp gaps in the broker's original M1 data.

IMPORTANT:
    This is a DIAGNOSTIC tool.

    It does NOT:
        - change session rules
        - classify gaps as valid/invalid
        - fill missing prices
        - remove candles
        - modify the existing Session Validator

    Its only purpose is to discover recurring broker-session patterns
    directly from the historical M1 data.

The results will later be used to build the session-aware rules in:

    src/session_validator.py
"""

import time

import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

SYMBOL_FILES = {
    "XAUUSDm": "XAUUSDm_M1_202108170000_202608131827.csv",
    "GBPUSDm": "GBPUSDm_M1_202108170000_202608131847.csv",
    "USDCHFm": "USDCHFm_M1_202108170000_202608131849.csv",
}

DATA_DIR = Path("data/raw")


REQUIRED_COLUMNS = [
    "<DATE>",
    "<TIME>",
    "<OPEN>",
    "<HIGH>",
    "<LOW>",
    "<CLOSE>",
]


# ============================================================
# LOAD M1 DATA
# ============================================================

def load_m1(filepath):
    """
    Load MetaTrader M1 data.

    Supports:
        - TAB
        - COMMA
        - SEMICOLON
    """

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"File does not exist: {filepath}"
        )

    print("\nLoading file:")
    print(f"    {filepath}")

    # --------------------------------------------------------
    # Detect separator
    # --------------------------------------------------------

    with open(
        filepath,
        "r",
        encoding="utf-8-sig",
        errors="replace",
    ) as f:

        first_line = f.readline()

    if "\t" in first_line:

        separator = "\t"
        separator_name = "TAB"

    elif "," in first_line:

        separator = ","
        separator_name = "COMMA"

    elif ";" in first_line:

        separator = ";"
        separator_name = "SEMICOLON"

    else:

        raise ValueError(
            "Could not detect CSV separator."
        )

    print(
        f"Detected separator: "
        f"{separator_name}"
    )

    # --------------------------------------------------------
    # Read data
    # --------------------------------------------------------

    df = pd.read_csv(
        filepath,
        sep=separator,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Clean column names
    # --------------------------------------------------------

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    print("\nDetected columns:")
    print(df.columns.tolist())

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # --------------------------------------------------------
    # Create Datetime
    # --------------------------------------------------------

    df["Datetime"] = pd.to_datetime(
        df["<DATE>"].astype(str).str.strip()
        + " "
        + df["<TIME>"].astype(str).str.strip(),
        format="%Y.%m.%d %H:%M:%S",
        errors="coerce",
    )

    # --------------------------------------------------------
    # Validate timestamps
    # --------------------------------------------------------

    invalid_datetime = (
        df["Datetime"]
        .isna()
        .sum()
    )

    if invalid_datetime > 0:

        raise ValueError(
            f"Found {invalid_datetime:,} "
            f"invalid Datetime values."
        )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = (
        df.sort_values("Datetime")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Remove duplicate timestamps
    # --------------------------------------------------------

    duplicates = (
        df["Datetime"]
        .duplicated()
        .sum()
    )

    if duplicates > 0:

        print(
            f"\nWARNING: "
            f"{duplicates:,} duplicate timestamps found."
        )

        df = (
            df.drop_duplicates(
                subset=["Datetime"],
                keep="first",
            )
            .reset_index(drop=True)
        )

    print(
        f"\nSuccessfully loaded "
        f"{len(df):,} M1 candles."
    )

    print(
        f"First candle: "
        f"{df['Datetime'].iloc[0]}"
    )

    print(
        f"Last candle:  "
        f"{df['Datetime'].iloc[-1]}"
    )

    return df


# ============================================================
# BUILD GAP TABLE
# ============================================================

def build_gap_table(m1):
    """
    Build a table containing every timestamp gap greater
    than one minute.

    No classification is performed here.
    """

    timestamps = m1["Datetime"]

    gaps = timestamps.diff()

    mask = gaps > pd.Timedelta(minutes=1)

    gap_indices = gaps.index[mask]

    if len(gap_indices) == 0:

        return pd.DataFrame(
            columns=[
                "previous",
                "current",
                "gap",
                "gap_minutes",
                "previous_date",
                "current_date",
                "previous_weekday",
                "current_weekday",
                "previous_time",
                "current_time",
            ]
        )

    previous_times = (
        timestamps.loc[
            gap_indices - 1
        ]
    )

    current_times = (
        timestamps.loc[
            gap_indices
        ]
    )

    gap_values = (
        gaps.loc[
            gap_indices
        ]
    )

    result = pd.DataFrame(
        {
            "previous": previous_times.values,
            "current": current_times.values,
            "gap": gap_values.values,
        }
    )

    # --------------------------------------------------------
    # Derived fields
    # --------------------------------------------------------

    result["gap_minutes"] = (
        result["gap"]
        .dt.total_seconds()
        / 60
    )

    result["previous_date"] = (
        result["previous"]
        .dt.date
    )

    result["current_date"] = (
        result["current"]
        .dt.date
    )

    result["previous_weekday"] = (
        result["previous"]
        .dt.day_name()
    )

    result["current_weekday"] = (
        result["current"]
        .dt.day_name()
    )

    result["previous_time"] = (
        result["previous"]
        .dt.strftime("%H:%M")
    )

    result["current_time"] = (
        result["current"]
        .dt.strftime("%H:%M")
    )

    return result


# ============================================================
# GAP SIZE SUMMARY
# ============================================================

def print_gap_size_summary(gaps, symbol):

    print("\n")
    print("=" * 70)
    print(
        f"GAP SIZE SUMMARY: {symbol}"
    )
    print("=" * 70)

    if gaps.empty:

        print("\nNo gaps greater than one minute.")

        return

    # --------------------------------------------------------
    # Exact gap sizes
    # --------------------------------------------------------

    counts = (
        gaps["gap_minutes"]
        .value_counts()
        .sort_index()
    )

    print("\nExact gap sizes:")
    print("-" * 70)

    for minutes, count in counts.head(30).items():

        print(
            f"{minutes:>8.0f} minute(s)"
            f"{count:>15,}"
        )

    if len(counts) > 30:

        print(
            f"\n... {len(counts) - 30:,} "
            f"additional gap sizes not displayed."
        )

    # --------------------------------------------------------
    # Broad ranges
    # --------------------------------------------------------

    bins = [
        1,
        2,
        5,
        10,
        30,
        60,
        120,
        240,
        720,
        1440,
        float("inf"),
    ]

    labels = [
        "2 minutes",
        "3-5 minutes",
        "6-10 minutes",
        "11-30 minutes",
        "31-60 minutes",
        "61-120 minutes",
        "121-240 minutes",
        "241-720 minutes",
        "721-1440 minutes",
        "Over 24 hours",
    ]

    categories = pd.cut(
        gaps["gap_minutes"],
        bins=bins,
        labels=labels,
        right=True,
    )

    range_counts = (
        categories
        .value_counts()
        .reindex(labels, fill_value=0)
    )

    print("\nGap ranges:")
    print("-" * 70)

    for label, count in range_counts.items():

        print(
            f"{label:<25}"
            f"{count:>10,}"
        )


# ============================================================
# RECURRING TIME-OF-DAY PATTERNS
# ============================================================

def analyze_time_patterns(gaps, symbol):

    print("\n")
    print("=" * 70)
    print(
        f"RECURRING TIME-OF-DAY GAP PATTERNS: {symbol}"
    )
    print("=" * 70)

    if gaps.empty:

        print("\nNo gaps detected.")

        return

    # --------------------------------------------------------
    # Exact previous -> current time transition
    # --------------------------------------------------------

    patterns = (
        gaps
        .groupby(
            [
                "previous_time",
                "current_time",
            ],
            as_index=False,
        )
        .agg(
            occurrences=("gap_minutes", "size"),
            gap_minutes=("gap_minutes", "first"),
        )
        .sort_values(
            "occurrences",
            ascending=False,
        )
    )

    print("\nMost recurring time transitions:")
    print("-" * 70)

    print(
        f"{'Previous':<12}"
        f"{'Current':<12}"
        f"{'Gap':>10}"
        f"{'Occurrences':>15}"
    )

    print("-" * 70)

    for _, row in patterns.head(30).iterrows():

        print(
            f"{row['previous_time']:<12}"
            f"{row['current_time']:<12}"
            f"{row['gap_minutes']:>8.0f} min"
            f"{row['occurrences']:>15,}"
        )


# ============================================================
# WEEKDAY PATTERNS
# ============================================================

def analyze_weekday_patterns(gaps, symbol):

    print("\n")
    print("=" * 70)
    print(
        f"WEEKDAY GAP PATTERNS: {symbol}"
    )
    print("=" * 70)

    if gaps.empty:

        print("\nNo gaps detected.")

        return

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    previous_counts = (
        gaps["previous_weekday"]
        .value_counts()
        .reindex(
            weekday_order,
            fill_value=0,
        )
    )

    current_counts = (
        gaps["current_weekday"]
        .value_counts()
        .reindex(
            weekday_order,
            fill_value=0,
        )
    )

    print("\nGap starts by previous-candle weekday:")
    print("-" * 70)

    for weekday in weekday_order:

        print(
            f"{weekday:<15}"
            f"{previous_counts[weekday]:>10,}"
        )

    print("\nGap resumes by current-candle weekday:")
    print("-" * 70)

    for weekday in weekday_order:

        print(
            f"{weekday:<15}"
            f"{current_counts[weekday]:>10,}"
        )


# ============================================================
# WEEKEND GAP ANALYSIS
# ============================================================

def analyze_weekend_gaps(gaps, symbol):

    print("\n")
    print("=" * 70)
    print(
        f"WEEKEND GAP ANALYSIS: {symbol}"
    )
    print("=" * 70)

    if gaps.empty:

        print("\nNo gaps detected.")

        return

    weekend_mask = (
        (gaps["previous_weekday"].isin(
            ["Friday", "Saturday", "Sunday"]
        ))
        |
        (gaps["current_weekday"].isin(
            ["Saturday", "Sunday", "Monday"]
        ))
    )

    weekend_gaps = gaps[
        weekend_mask
    ]

    print(
        f"\nWeekend-related gaps: "
        f"{len(weekend_gaps):,}"
    )

    if weekend_gaps.empty:

        return

    patterns = (
        weekend_gaps
        .groupby(
            [
                "previous_weekday",
                "previous_time",
                "current_weekday",
                "current_time",
            ],
            as_index=False,
        )
        .agg(
            occurrences=("gap_minutes", "size"),
            gap_minutes=("gap_minutes", "first"),
        )
        .sort_values(
            "occurrences",
            ascending=False,
        )
    )

    print("\nMost common weekend transitions:")
    print("-" * 70)

    print(
        f"{'Previous':<24}"
        f"{'Current':<24}"
        f"{'Gap':>10}"
        f"{'Occurrences':>15}"
    )

    print("-" * 70)

    for _, row in patterns.head(20).iterrows():

        previous_label = (
            f"{row['previous_weekday']} "
            f"{row['previous_time']}"
        )

        current_label = (
            f"{row['current_weekday']} "
            f"{row['current_time']}"
        )

        print(
            f"{previous_label:<24}"
            f"{current_label:<24}"
            f"{row['gap_minutes']:>8.0f} min"
            f"{row['occurrences']:>15,}"
        )


# ============================================================
# DAILY RECURRING PATTERNS
# ============================================================

def analyze_daily_patterns(gaps, symbol):

    print("\n")
    print("=" * 70)
    print(
        f"RECURRING DAILY GAP PATTERNS: {symbol}"
    )
    print("=" * 70)

    if gaps.empty:

        print("\nNo gaps detected.")

        return

    # --------------------------------------------------------
    # Exclude obvious weekend transitions
    # --------------------------------------------------------

    weekend_mask = (
        gaps["previous_weekday"].isin(
            ["Saturday", "Sunday"]
        )
        |
        gaps["current_weekday"].isin(
            ["Saturday", "Sunday"]
        )
        |
        (
            (gaps["previous_weekday"] == "Friday")
            &
            (gaps["current_weekday"] == "Monday")
        )
    )

    daily_gaps = gaps[
        ~weekend_mask
    ]

    if daily_gaps.empty:

        print("\nNo weekday-only gaps detected.")

        return

    patterns = (
        daily_gaps
        .groupby(
            [
                "previous_time",
                "current_time",
            ],
            as_index=False,
        )
        .agg(
            occurrences=("gap_minutes", "size"),
            gap_minutes=("gap_minutes", "first"),
        )
        .sort_values(
            "occurrences",
            ascending=False,
        )
    )

    print(
        "\nMost recurring weekday-only transitions:"
    )

    print("-" * 70)

    print(
        f"{'Previous':<12}"
        f"{'Current':<12}"
        f"{'Gap':>10}"
        f"{'Occurrences':>15}"
    )

    print("-" * 70)

    for _, row in patterns.head(30).iterrows():

        print(
            f"{row['previous_time']:<12}"
            f"{row['current_time']:<12}"
            f"{row['gap_minutes']:>8.0f} min"
            f"{row['occurrences']:>15,}"
        )


# ============================================================
# HISTORICAL CONSISTENCY
# ============================================================

def analyze_historical_consistency(
    gaps,
    symbol,
):
    """
    Show how frequently recurring patterns appear by year.

    This helps determine whether a pattern is stable across
    the entire historical dataset or only appears during
    certain periods.
    """

    print("\n")
    print("=" * 70)
    print(
        f"HISTORICAL PATTERN CONSISTENCY: {symbol}"
    )
    print("=" * 70)

    if gaps.empty:

        print("\nNo gaps detected.")

        return

    gaps = gaps.copy()

    gaps["year"] = (
        gaps["current"]
        .dt.year
    )

    # --------------------------------------------------------
    # Find recurring weekday transition patterns
    # --------------------------------------------------------

    weekday_mask = (
        ~(
            gaps["previous_weekday"].isin(
                ["Saturday", "Sunday"]
            )
            |
            gaps["current_weekday"].isin(
                ["Saturday", "Sunday"]
            )
        )
    )

    weekday_gaps = gaps[
        weekday_mask
    ]

    if weekday_gaps.empty:

        print("\nNo weekday patterns found.")

        return

    patterns = (
        weekday_gaps
        .groupby(
            [
                "previous_time",
                "current_time",
            ]
        )
        .size()
        .sort_values(
            ascending=False
        )
    )

    top_patterns = patterns.head(10)

    for (
        previous_time,
        current_time,
    ), total_count in top_patterns.items():

        subset = weekday_gaps[
            (
                weekday_gaps["previous_time"]
                == previous_time
            )
            &
            (
                weekday_gaps["current_time"]
                == current_time
            )
        ]

        yearly = (
            subset["year"]
            .value_counts()
            .sort_index()
        )

        print("\n")
        print(
            f"Pattern: "
            f"{previous_time} -> "
            f"{current_time}"
        )

        print(
            f"Total occurrences: "
            f"{total_count:,}"
        )

        print(
            "Yearly occurrences:"
        )

        for year, count in yearly.items():

            print(
                f"    {year}: "
                f"{count:,}"
            )


# ============================================================
# SHORT GAP ANALYSIS
# ============================================================

def analyze_short_gaps(gaps, symbol):

    print("\n")
    print("=" * 70)
    print(
        f"SHORT GAP ANALYSIS: {symbol}"
    )
    print("=" * 70)

    if gaps.empty:

        print("\nNo gaps detected.")

        return

    short_gaps = gaps[
        gaps["gap_minutes"] <= 5
    ]

    print(
        f"\nGaps of 2-5 minutes: "
        f"{len(short_gaps):,}"
    )

    if short_gaps.empty:

        print("\nNo short gaps detected.")

        return

    patterns = (
        short_gaps
        .groupby(
            [
                "previous_time",
                "current_time",
            ],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size": "occurrences"
            }
        )
        .sort_values(
            "occurrences",
            ascending=False,
        )
    )

    print("\nMost common short-gap transitions:")
    print("-" * 70)

    print(
        f"{'Previous':<12}"
        f"{'Current':<12}"
        f"{'Gap':>10}"
        f"{'Occurrences':>15}"
    )

    print("-" * 70)

    for _, row in patterns.head(30).iterrows():

        previous_time = row["previous_time"]
        current_time = row["current_time"]

        previous_minutes = (
            int(previous_time[:2]) * 60
            + int(previous_time[3:])
        )

        current_minutes = (
            int(current_time[:2]) * 60
            + int(current_time[3:])
        )

        gap_minutes = (
            current_minutes
            - previous_minutes
        )

        if gap_minutes <= 0:
            gap_minutes += 24 * 60

        print(
            f"{previous_time:<12}"
            f"{current_time:<12}"
            f"{gap_minutes:>8} min"
            f"{row['occurrences']:>15,}"
        )


# ============================================================
# PROCESS SYMBOL
# ============================================================

def process_symbol(
    symbol,
    filename,
):

    print("\n")
    print("=" * 70)
    print(
        f"SESSION PATTERN DIAGNOSTIC: {symbol}"
    )
    print("=" * 70)

    symbol_start = time.perf_counter()

    filepath = (
        DATA_DIR
        / filename
    )

    try:

        load_start = time.perf_counter()

        m1 = load_m1(
            filepath
        )

        load_time = (
            time.perf_counter()
            - load_start
        )

        print(
            f"\n[PERFORMANCE] "
            f"Load time: "
            f"{load_time:.2f} seconds"
        )

    except Exception as error:

        print(
            f"\nERROR loading {symbol}:"
        )

        print(error)

        return

    # ========================================================
    # GAP TABLE
    # ========================================================

    print("\n")
    print(
        "Building timestamp gap table..."
    )

    gap_start = time.perf_counter()

    gaps = build_gap_table(
        m1
    )

    gap_time = (
        time.perf_counter()
        - gap_start
    )

    print(
        f"Detected "
        f"{len(gaps):,} gaps greater "
        f"than one minute."
    )

    print(
        f"[PERFORMANCE] "
        f"Gap table time: "
        f"{gap_time:.2f} seconds"
    )

    # ========================================================
    # ANALYSIS
    # ========================================================

    print_gap_size_summary(
        gaps,
        symbol,
    )

    analyze_time_patterns(
        gaps,
        symbol,
    )

    analyze_weekday_patterns(
        gaps,
        symbol,
    )

    analyze_weekend_gaps(
        gaps,
        symbol,
    )

    analyze_daily_patterns(
        gaps,
        symbol,
    )

    analyze_historical_consistency(
        gaps,
        symbol,
    )

    analyze_short_gaps(
        gaps,
        symbol,
    )

    # ========================================================
    # TOTAL
    # ========================================================

    total_time = (
        time.perf_counter()
        - symbol_start
    )

    print("\n")
    print(
        f"[PERFORMANCE] "
        f"{symbol} diagnostic time: "
        f"{total_time:.2f} seconds"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "       EMA-ADX TRADING SYSTEM - V3.0"
    )

    print(
        "       BROKER SESSION PATTERN DIAGNOSTIC"
    )

    print("=" * 70)

    print("\n")
    print(
        "IMPORTANT:"
    )

    print(
        "This diagnostic does NOT modify "
        "session rules or trading data."
    )

    print(
        "It only identifies recurring "
        "timestamp-gap patterns."
    )

    # ========================================================
    # PROCESS SYMBOLS
    # ========================================================

    for symbol, filename in SYMBOL_FILES.items():

        process_symbol(
            symbol,
            filename,
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 70)

    print(
        "SESSION PATTERN DIAGNOSTIC COMPLETE"
    )

    print("=" * 70)

    print("\n")
    print(
        "No session rules were changed."
    )

    print(
        "Use these results to establish "
        "the broker-session model before "
        "modifying session_validator.py."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
