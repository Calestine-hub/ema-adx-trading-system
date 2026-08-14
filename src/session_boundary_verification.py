"""
EMA-ADX TRADING SYSTEM - V3.0
SESSION BOUNDARY VERIFICATION

Purpose:
    Verify whether recurring M1 timestamp gaps represent
    systematic broker-session boundaries or isolated missing data.

IMPORTANT:
    This script is diagnostic only.

    It does NOT:
        - modify the raw data
        - fill missing candles
        - classify candles as valid/invalid
        - modify session_validator.py
        - create trading signals

    It examines recurring gap patterns and measures their
    consistency by date, weekday, and historical period.
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
# VERIFICATION SETTINGS
# ============================================================

# Minimum number of occurrences before a gap transition
# is considered interesting enough for detailed analysis.
MIN_PATTERN_OCCURRENCES = 20

# Number of strongest recurring patterns to investigate.
TOP_PATTERNS = 15

# A recurring pattern should appear on a meaningful portion
# of the dates on which it could reasonably occur.
#
# This is NOT a final classification threshold.
# It is only used to highlight candidates.
CANDIDATE_CONSISTENCY_PERCENT = 20.0


# ============================================================
# LOAD M1 DATA
# ============================================================

def load_m1(filepath):
    """
    Load MetaTrader M1 data.

    Supports TAB, COMMA and SEMICOLON separators.
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
        f"Detected separator: {separator_name}"
    )

    # --------------------------------------------------------
    # Read
    # --------------------------------------------------------

    df = pd.read_csv(
        filepath,
        sep=separator,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Clean columns
    # --------------------------------------------------------

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Validate
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
    # Datetime
    # --------------------------------------------------------

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

    if duplicates:
        print(
            f"WARNING: removing "
            f"{duplicates:,} duplicate timestamps."
        )

        df = (
            df.drop_duplicates(
                subset=["Datetime"],
                keep="first",
            )
            .reset_index(drop=True)
        )

    print(
        f"Successfully loaded "
        f"{len(df):,} M1 candles."
    )

    print(
        f"First: {df['Datetime'].iloc[0]}"
    )

    print(
        f"Last:  {df['Datetime'].iloc[-1]}"
    )

    return df


# ============================================================
# BUILD GAP TABLE
# ============================================================

def build_gap_table(m1):
    """
    Build one row for every timestamp gap greater than one
    minute.
    """

    timestamps = m1["Datetime"]

    differences = timestamps.diff()

    mask = (
        differences
        > pd.Timedelta(minutes=1)
    )

    indices = differences.index[mask]

    if len(indices) == 0:

        return pd.DataFrame(
            columns=[
                "previous",
                "current",
                "gap_minutes",
                "previous_date",
                "current_date",
                "previous_weekday",
                "current_weekday",
                "previous_time",
                "current_time",
            ]
        )

    previous = timestamps.loc[
        indices - 1
    ].reset_index(drop=True)

    current = timestamps.loc[
        indices
    ].reset_index(drop=True)

    result = pd.DataFrame(
        {
            "previous": previous,
            "current": current,
        }
    )

    result["gap_minutes"] = (
        result["current"]
        - result["previous"]
    ).dt.total_seconds() / 60

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

    result["year"] = (
        result["current"]
        .dt.year
    )

    return result


# ============================================================
# DETERMINE AVAILABLE DATES
# ============================================================

def build_expected_weekday_dates(m1):
    """
    Determine how many dates exist for each weekday in the
    historical dataset.

    This gives us a denominator for consistency analysis.
    """

    dates = (
        m1["Datetime"]
        .dt.normalize()
        .drop_duplicates()
    )

    calendar = pd.DataFrame(
        {
            "date": dates
        }
    )

    calendar["weekday"] = (
        calendar["date"]
        .dt.day_name()
    )

    calendar["year"] = (
        calendar["date"]
        .dt.year
    )

    return calendar


# ============================================================
# FIND RECURRING PATTERNS
# ============================================================

def find_recurring_patterns(gaps):

    if gaps.empty:
        return pd.DataFrame()

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
            occurrences=(
                "gap_minutes",
                "size",
            ),
            gap_minutes=(
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
        .sort_values(
            "occurrences",
            ascending=False,
        )
    )

    return patterns


# ============================================================
# CANDIDATE PATTERN ANALYSIS
# ============================================================

def analyze_candidate_pattern(
    gaps,
    calendar,
    previous_time,
    current_time,
):
    """
    Measure how consistently one exact time transition
    occurs across the historical dates where it could occur.

    This is a diagnostic measure only.
    """

    subset = gaps[
        (
            gaps["previous_time"]
            == previous_time
        )
        &
        (
            gaps["current_time"]
            == current_time
        )
    ].copy()

    if subset.empty:
        return None

    # --------------------------------------------------------
    # Unique occurrence dates
    # --------------------------------------------------------

    occurrence_dates = (
        subset["previous_date"]
        .drop_duplicates()
    )

    occurrence_dates = pd.to_datetime(
        occurrence_dates
    )

    # --------------------------------------------------------
    # Weekday distribution
    # --------------------------------------------------------

    weekday_counts = (
        subset["previous_weekday"]
        .value_counts()
    )

    # --------------------------------------------------------
    # Year distribution
    # --------------------------------------------------------

    yearly_counts = (
        subset["year"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # Determine dominant weekday
    # --------------------------------------------------------

    if weekday_counts.empty:

        dominant_weekday = "N/A"
        dominant_weekday_count = 0

    else:

        dominant_weekday = (
            weekday_counts
            .index[0]
        )

        dominant_weekday_count = (
            weekday_counts.iloc[0]
        )

    # --------------------------------------------------------
    # Historical years
    # --------------------------------------------------------

    years_present = sorted(
        subset["year"]
        .unique()
        .tolist()
    )

    return {
        "previous_time": previous_time,
        "current_time": current_time,
        "occurrences": len(subset),
        "unique_dates": len(occurrence_dates),
        "median_gap": subset[
            "gap_minutes"
        ].median(),
        "min_gap": subset[
            "gap_minutes"
        ].min(),
        "max_gap": subset[
            "gap_minutes"
        ].max(),
        "dominant_weekday": dominant_weekday,
        "dominant_weekday_count":
            dominant_weekday_count,
        "years_present": years_present,
        "yearly_counts": yearly_counts,
        "weekday_counts": weekday_counts,
    }


# ============================================================
# PRINT CANDIDATE REPORT
# ============================================================

def print_candidate_report(
    gaps,
    calendar,
    symbol,
):
    """
    Print the strongest recurring gap candidates.
    """

    print("\n")
    print("=" * 80)
    print(
        f"SESSION BOUNDARY CANDIDATES: {symbol}"
    )
    print("=" * 80)

    if gaps.empty:

        print("\nNo gaps detected.")
        return

    patterns = find_recurring_patterns(
        gaps
    )

    patterns = patterns[
        patterns["occurrences"]
        >= MIN_PATTERN_OCCURRENCES
    ]

    if patterns.empty:

        print(
            "\nNo recurring patterns met "
            f"the minimum of "
            f"{MIN_PATTERN_OCCURRENCES} occurrences."
        )

        return

    candidates = []

    for _, pattern in patterns.head(
        TOP_PATTERNS
    ).iterrows():

        analysis = analyze_candidate_pattern(
            gaps,
            calendar,
            pattern["previous_time"],
            pattern["current_time"],
        )

        if analysis is not None:
            candidates.append(
                analysis
            )

    print(
        "\nThe following are recurring "
        "PATTERN CANDIDATES."
    )

    print(
        "They are NOT yet classified "
        "as session closures."
    )

    print("\n")

    print(
        f"{'Previous':<10}"
        f"{'Current':<10}"
        f"{'Occurrences':>13}"
        f"{'Unique Dates':>13}"
        f"{'Median Gap':>13}"
        f"{'Min':>8}"
        f"{'Max':>8}"
        f"{'Dominant Day':>15}"
    )

    print("-" * 80)

    for candidate in candidates:

        print(
            f"{candidate['previous_time']:<10}"
            f"{candidate['current_time']:<10}"
            f"{candidate['occurrences']:>13,}"
            f"{candidate['unique_dates']:>13,}"
            f"{candidate['median_gap']:>11.0f} min"
            f"{candidate['min_gap']:>8.0f}"
            f"{candidate['max_gap']:>8.0f}"
            f"{candidate['dominant_weekday']:>15}"
        )

    # ========================================================
    # DETAILED ANALYSIS
    # ========================================================

    for number, candidate in enumerate(
        candidates,
        start=1,
    ):

        print("\n")
        print("-" * 80)

        print(
            f"CANDIDATE #{number}: "
            f"{candidate['previous_time']} "
            f"-> "
            f"{candidate['current_time']}"
        )

        print("-" * 80)

        print(
            f"Occurrences: "
            f"{candidate['occurrences']:,}"
        )

        print(
            f"Unique dates: "
            f"{candidate['unique_dates']:,}"
        )

        print(
            f"Gap range: "
            f"{candidate['min_gap']:.0f} "
            f"to "
            f"{candidate['max_gap']:.0f} minutes"
        )

        print(
            f"Median gap: "
            f"{candidate['median_gap']:.0f} minutes"
        )

        print(
            f"Dominant weekday: "
            f"{candidate['dominant_weekday']}"
        )

        print("\nWeekday distribution:")

        for weekday, count in (
            candidate["weekday_counts"]
            .items()
        ):

            print(
                f"    {weekday:<12}"
                f"{count:>10,}"
            )

        print("\nYear distribution:")

        for year, count in (
            candidate["yearly_counts"]
            .items()
        ):

            print(
                f"    {year}: "
                f"{count:>10,}"
            )


# ============================================================
# DAILY SESSION-LIKE PATTERN DETECTION
# ============================================================

def detect_daily_patterns(
    gaps,
    symbol,
):
    """
    Identify patterns that recur across multiple weekdays.

    A strong daily pattern is particularly interesting because
    a broker's daily trading break commonly produces this shape.

    Again, this is diagnostic only.
    """

    print("\n")
    print("=" * 80)
    print(
        f"DAILY SESSION-LIKE PATTERNS: {symbol}"
    )
    print("=" * 80)

    if gaps.empty:
        print("\nNo gaps detected.")
        return

    # --------------------------------------------------------
    # Remove obvious weekend transitions
    # --------------------------------------------------------

    weekday_gaps = gaps[
        ~(
            gaps["previous_weekday"].isin(
                ["Saturday", "Sunday"]
            )
            |
            gaps["current_weekday"].isin(
                ["Saturday", "Sunday"]
            )
        )
    ].copy()

    if weekday_gaps.empty:
        print("\nNo weekday gaps.")
        return

    # --------------------------------------------------------
    # Count unique dates for each transition
    # --------------------------------------------------------

    pattern_stats = (
        weekday_gaps
        .groupby(
            [
                "previous_time",
                "current_time",
            ]
        )
        .agg(
            occurrences=(
                "gap_minutes",
                "size",
            ),
            unique_dates=(
                "previous_date",
                "nunique",
            ),
            weekdays=(
                "previous_weekday",
                "nunique",
            ),
            years=(
                "year",
                "nunique",
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Strong candidates
    # --------------------------------------------------------

    candidates = pattern_stats[
        (
            pattern_stats["occurrences"]
            >= MIN_PATTERN_OCCURRENCES
        )
        &
        (
            pattern_stats["weekdays"]
            >= 3
        )
        &
        (
            pattern_stats["years"]
            >= 2
        )
    ]

    candidates = candidates.sort_values(
        [
            "unique_dates",
            "occurrences",
        ],
        ascending=False,
    )

    if candidates.empty:

        print(
            "\nNo strong multi-weekday "
            "patterns found."
        )

        return

    print(
        "\nPatterns appearing across "
        "multiple weekdays and years:"
    )

    print("-" * 80)

    print(
        f"{'Previous':<10}"
        f"{'Current':<10}"
        f"{'Occurrences':>13}"
        f"{'Dates':>10}"
        f"{'Weekdays':>10}"
        f"{'Years':>10}"
    )

    print("-" * 80)

    for _, row in candidates.head(20).iterrows():

        print(
            f"{row['previous_time']:<10}"
            f"{row['current_time']:<10}"
            f"{row['occurrences']:>13,}"
            f"{row['unique_dates']:>10,}"
            f"{row['weekdays']:>10,}"
            f"{row['years']:>10,}"
        )


# ============================================================
# EXTRACT EXAMPLE DATES
# ============================================================

def print_example_dates(
    gaps,
    symbol,
):
    """
    Print actual dates for the strongest recurring patterns.

    This allows us to manually inspect the underlying M1 data
    later if necessary.
    """

    print("\n")
    print("=" * 80)
    print(
        f"EXAMPLE DATES FOR RECURRING PATTERNS: {symbol}"
    )
    print("=" * 80)

    if gaps.empty:
        return

    patterns = find_recurring_patterns(
        gaps
    )

    patterns = patterns[
        patterns["occurrences"]
        >= MIN_PATTERN_OCCURRENCES
    ]

    for _, pattern in patterns.head(10).iterrows():

        previous_time = (
            pattern["previous_time"]
        )

        current_time = (
            pattern["current_time"]
        )

        subset = gaps[
            (
                gaps["previous_time"]
                == previous_time
            )
            &
            (
                gaps["current_time"]
                == current_time
            )
        ]

        dates = (
            subset["previous_date"]
            .drop_duplicates()
            .sort_values()
        )

        print("\n")
        print(
            f"Pattern: "
            f"{previous_time} -> "
            f"{current_time}"
        )

        print(
            f"Total occurrences: "
            f"{len(subset):,}"
        )

        print(
            "First 10 dates:"
        )

        for date in dates.head(10):

            print(
                f"    {date}"
            )

        if len(dates) > 10:

            print(
                "Last 10 dates:"
            )

            for date in dates.tail(10):

                print(
                    f"    {date}"
                )


# ============================================================
# PROCESS SYMBOL
# ============================================================

def process_symbol(
    symbol,
    filename,
):

    print("\n")
    print("#" * 80)
    print(
        f"# PROCESSING {symbol}"
    )
    print("#" * 80)

    start = time.perf_counter()

    filepath = (
        DATA_DIR
        / filename
    )

    try:

        m1 = load_m1(
            filepath
        )

    except Exception as error:

        print(
            f"\nERROR: {error}"
        )

        return

    # --------------------------------------------------------
    # Calendar
    # --------------------------------------------------------

    calendar = (
        build_expected_weekday_dates(
            m1
        )
    )

    # --------------------------------------------------------
    # Gaps
    # --------------------------------------------------------

    print(
        "\nBuilding gap table..."
    )

    gaps = build_gap_table(
        m1
    )

    print(
        f"Total gaps > 1 minute: "
        f"{len(gaps):,}"
    )

    # --------------------------------------------------------
    # Candidate analysis
    # --------------------------------------------------------

    print_candidate_report(
        gaps,
        calendar,
        symbol,
    )

    # --------------------------------------------------------
    # Multi-weekday patterns
    # --------------------------------------------------------

    detect_daily_patterns(
        gaps,
        symbol,
    )

    # --------------------------------------------------------
    # Example dates
    # --------------------------------------------------------

    print_example_dates(
        gaps,
        symbol,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    print("\n")
    print(
        f"[PERFORMANCE] "
        f"{symbol}: "
        f"{elapsed:.2f} seconds"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "       EMA-ADX TRADING SYSTEM - V3.0"
    )

    print(
        "       SESSION BOUNDARY VERIFICATION"
    )

    print("=" * 80)

    print("\n")
    print(
        "This is a diagnostic-only stage."
    )

    print(
        "No session rules will be changed."
    )

    print(
        "No missing prices will be filled."
    )

    print(
        "No candles will be removed."
    )

    for symbol, filename in (
        SYMBOL_FILES.items()
    ):

        process_symbol(
            symbol,
            filename,
        )

    print("\n")
    print("=" * 80)

    print(
        "SESSION BOUNDARY VERIFICATION COMPLETE"
    )

    print("=" * 80)

    print("\n")
    print(
        "NEXT STEP:"
    )

    print(
        "Review the recurring patterns before "
        "modifying session_validator.py."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()