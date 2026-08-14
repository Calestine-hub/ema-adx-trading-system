
"""
EMA-ADX TRADING SYSTEM - V3.0
SESSION-AWARE CANDLE VALIDATOR

Purpose:
    Distinguish expected broker-session closures from unexpected
    intraday data gaps.

The validator works from the original M1 data and classifies
15-minute and 4-hour candle periods without filling missing prices.

IMPORTANT:
    MetaTrader exported files may be TAB-separated rather than
    comma-separated. This loader automatically detects the separator.

PERFORMANCE:
    This version is optimized for large M1 datasets.

    Improvements:
        1. Gap analysis is vectorized with pandas.
        2. Candle validation uses DatetimeIndex.searchsorted()
           instead of repeatedly scanning the entire M1 DataFrame.
        3. Stage execution times are displayed.
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
# SESSION RULES
# ============================================================

def classify_gap(previous_time, current_time):
    """
    Classify an M1 timestamp gap.

    We deliberately do NOT call every large gap bad data.
    Weekends and long market closures are expected.
    """

    gap = current_time - previous_time
    minutes = gap.total_seconds() / 60

    # Normal consecutive M1 candles
    if minutes == 1:
        return "NORMAL"

    # Friday -> Monday
    if (
        previous_time.weekday() == 4
        and current_time.weekday() == 0
    ):
        return "EXPECTED_SESSION_CLOSURE"

    # Saturday/Sunday related closure
    if (
        previous_time.weekday() >= 5
        or current_time.weekday() >= 5
    ):
        return "EXPECTED_SESSION_CLOSURE"

    # Large gap
    if minutes >= 24 * 60:
        return "EXPECTED_SESSION_CLOSURE"

    # Short intraday gap
    if minutes <= 5:
        return "SHORT_DATA_GAP"

    # Medium intraday gap
    if minutes <= 60:
        return "INTRADAY_DATA_GAP"

    # Long intraday gap
    return "LONG_DATA_GAP"


# ============================================================
# LOAD M1 DATA
# ============================================================

def load_m1(filepath):
    """
    Load MetaTrader exported M1 CSV.

    Supports:
        - TAB-separated files
        - comma-separated files
        - semicolon-separated files

    The files provided by the user are TAB-separated.
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
    # Read file
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
    # Combine DATE + TIME
    # --------------------------------------------------------

    df["Datetime"] = pd.to_datetime(
        df["<DATE>"].astype(str).str.strip()
        + " "
        + df["<TIME>"].astype(str).str.strip(),
        format="%Y.%m.%d %H:%M:%S",
        errors="coerce",
    )

    # --------------------------------------------------------
    # Check invalid timestamps
    # --------------------------------------------------------

    invalid_datetime = df["Datetime"].isna().sum()

    if invalid_datetime > 0:

        raise ValueError(
            f"Found {invalid_datetime:,} invalid "
            f"Datetime values."
        )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = (
        df.sort_values("Datetime")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Remove duplicate timestamps
    # --------------------------------------------------------

    duplicates = df["Datetime"].duplicated().sum()

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

    # --------------------------------------------------------
    # Convert OHLC columns to numeric
    # --------------------------------------------------------

    numeric_columns = [
        "<OPEN>",
        "<HIGH>",
        "<LOW>",
        "<CLOSE>",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Check invalid OHLC values
    # --------------------------------------------------------

    invalid_ohlc = (
        df[numeric_columns]
        .isna()
        .any(axis=1)
        .sum()
    )

    if invalid_ohlc > 0:

        raise ValueError(
            f"Found {invalid_ohlc:,} rows "
            f"with invalid OHLC values."
        )

    # --------------------------------------------------------
    # Standardize OHLC names
    # --------------------------------------------------------

    df["Open"] = df["<OPEN>"]
    df["High"] = df["<HIGH>"]
    df["Low"] = df["<LOW>"]
    df["Close"] = df["<CLOSE>"]

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
# GAP ANALYSIS - OPTIMIZED
# ============================================================

def analyze_gaps(m1):
    """
    Analyze timestamp gaps using vectorized pandas operations.

    The previous implementation performed millions of Python-level
    .iloc[] operations.

    This version calculates all timestamp differences at once and
    only iterates over the relatively small set of actual gaps.
    """

    timestamps = m1["Datetime"]

    # --------------------------------------------------------
    # Calculate all timestamp differences at once
    # --------------------------------------------------------

    gaps = timestamps.diff()

    # --------------------------------------------------------
    # Keep only gaps greater than one minute
    # --------------------------------------------------------

    gap_mask = gaps > pd.Timedelta(minutes=1)

    gap_indices = gaps.index[gap_mask]

    if len(gap_indices) == 0:

        return pd.DataFrame(
            columns=[
                "previous",
                "current",
                "gap",
                "classification",
            ]
        )

    # --------------------------------------------------------
    # Extract only actual gap rows
    # --------------------------------------------------------

    current_times = timestamps.loc[gap_indices]
    previous_times = timestamps.loc[gap_indices - 1]
    gap_values = gaps.loc[gap_indices]

    # --------------------------------------------------------
    # Classify only the actual gaps
    # --------------------------------------------------------

    classifications = [
        classify_gap(
            previous_time,
            current_time,
        )
        for previous_time, current_time
        in zip(
            previous_times,
            current_times,
        )
    ]

    # --------------------------------------------------------
    # Build result
    # --------------------------------------------------------

    results = pd.DataFrame(
        {
            "previous": previous_times.values,
            "current": current_times.values,
            "gap": gap_values.values,
            "classification": classifications,
        }
    )

    return results


# ============================================================
# PRINT GAP SUMMARY
# ============================================================

def print_gap_summary(gaps, symbol):

    print("\n")
    print("=" * 70)

    print(
        f"SESSION-AWARE GAP ANALYSIS: {symbol}"
    )

    print("=" * 70)

    if gaps.empty:

        print("\nNo timestamp gaps detected.")

        return

    print("\nGap classification:")
    print("-" * 70)

    counts = (
        gaps["classification"]
        .value_counts()
    )

    for classification, count in counts.items():

        print(
            f"{classification:<30}"
            f"{count:,}"
        )

    # --------------------------------------------------------
    # Unexpected gaps
    # --------------------------------------------------------

    print("\n")
    print("Unexpected gaps:")
    print("-" * 70)

    unexpected = gaps[
        gaps["classification"].isin(
            [
                "SHORT_DATA_GAP",
                "INTRADAY_DATA_GAP",
                "LONG_DATA_GAP",
            ]
        )
    ]

    if unexpected.empty:

        print(
            "No unexpected gaps detected."
        )

    else:

        print(
            unexpected
            .head(20)
            .to_string(index=False)
        )

        print(
            f"\nTotal unexpected gaps: "
            f"{len(unexpected):,}"
        )


# ============================================================
# CANDLE PERIOD VALIDATION - OPTIMIZED
# ============================================================

def validate_candle_period(
    timestamps,
    start_time,
    timeframe_minutes,
):
    """
    Validate one candle period using timestamp positions.

    Instead of filtering the entire M1 DataFrame, we use
    searchsorted() on the already-sorted DatetimeIndex.

    This preserves the original validation logic while making
    the operation dramatically faster.
    """

    end_time = (
        start_time
        + pd.Timedelta(
            minutes=timeframe_minutes
        )
    )

    # --------------------------------------------------------
    # Find positions of period boundaries
    # --------------------------------------------------------

    start_position = timestamps.searchsorted(
        start_time,
        side="left",
    )

    end_position = timestamps.searchsorted(
        end_time,
        side="left",
    )

    # --------------------------------------------------------
    # Number of M1 candles inside period
    # --------------------------------------------------------

    actual_minutes = (
        end_position
        - start_position
    )

    expected_minutes = timeframe_minutes

    missing_minutes = (
        expected_minutes
        - actual_minutes
    )

    # --------------------------------------------------------
    # Fully populated
    # --------------------------------------------------------

    if missing_minutes == 0:

        return {
            "status": "COMPLETE",
            "expected": expected_minutes,
            "actual": actual_minutes,
            "missing": 0,
        }

    # --------------------------------------------------------
    # No data
    # --------------------------------------------------------

    if actual_minutes == 0:

        return {
            "status": "SESSION_CLOSED",
            "expected": expected_minutes,
            "actual": 0,
            "missing": expected_minutes,
        }

    # --------------------------------------------------------
    # Check internal timestamp gaps
    # --------------------------------------------------------

    period_timestamps = timestamps[
        start_position:end_position
    ]

    if len(period_timestamps) > 1:

        internal_gaps = (
            period_timestamps[1:]
            - period_timestamps[:-1]
        )

        unexpected_internal_gap = (
            internal_gaps
            > pd.Timedelta(minutes=1)
        ).any()

    else:

        unexpected_internal_gap = False

    # --------------------------------------------------------
    # Classify
    # --------------------------------------------------------

    if unexpected_internal_gap:

        status = "INTRADAY_DATA_GAP"

    else:

        status = "SESSION_INCOMPLETE"

    return {
        "status": status,
        "expected": expected_minutes,
        "actual": actual_minutes,
        "missing": missing_minutes,
    }


# ============================================================
# BUILD VALIDATION TABLE - OPTIMIZED
# ============================================================

def validate_timeframe(
    m1,
    timeframe_minutes,
):
    """
    Validate all periods for a timeframe.

    Uses a sorted DatetimeIndex and searchsorted() so that each
    period does not require scanning the entire M1 DataFrame.
    """

    # --------------------------------------------------------
    # Create sorted DatetimeIndex once
    # --------------------------------------------------------

    timestamps = pd.DatetimeIndex(
        m1["Datetime"]
    )

    first_time = (
        timestamps.min()
        .floor(
            f"{timeframe_minutes}min"
        )
    )

    last_time = (
        timestamps.max()
        .floor(
            f"{timeframe_minutes}min"
        )
    )

    periods = pd.date_range(
        start=first_time,
        end=last_time,
        freq=f"{timeframe_minutes}min",
    )

    print(
        f"Periods to validate: "
        f"{len(periods):,}"
    )

    # --------------------------------------------------------
    # Validate periods
    # --------------------------------------------------------

    results = []

    total_periods = len(periods)

    progress_interval = max(
        1,
        total_periods // 10,
    )

    for index, start_time in enumerate(
        periods
    ):

        result = validate_candle_period(
            timestamps,
            start_time,
            timeframe_minutes,
        )

        results.append(
            {
                "Datetime": start_time,
                **result,
            }
        )

        # ----------------------------------------------------
        # Progress display
        # ----------------------------------------------------

        if (
            (index + 1) % progress_interval == 0
            or index == total_periods - 1
        ):

            percentage = (
                (index + 1)
                / total_periods
                * 100
            )

            print(
                f"  Progress: "
                f"{index + 1:,}/"
                f"{total_periods:,}"
                f" ({percentage:5.1f}%)"
            )

    return pd.DataFrame(results)


# ============================================================
# TIMEFRAME SUMMARY
# ============================================================

def print_timeframe_summary(
    validation,
    timeframe_name,
    symbol,
):

    print("\n")
    print("-" * 70)

    print(
        f"{symbol} | {timeframe_name} "
        f"SESSION VALIDATION"
    )

    print("-" * 70)

    counts = (
        validation["status"]
        .value_counts()
    )

    total = len(validation)

    for status, count in counts.items():

        percentage = (
            count
            / total
            * 100
        )

        print(
            f"{status:<25}"
            f"{count:>10,}"
            f"  ({percentage:6.2f}%)"
        )

    print("-" * 70)

    unexpected = validation[
        validation["status"]
        == "INTRADAY_DATA_GAP"
    ]

    print(
        f"Unexpected incomplete candles: "
        f"{len(unexpected):,}"
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
        "          SESSION VALIDATOR"
    )

    print("=" * 70)

    # ========================================================
    # PROCESS EACH SYMBOL
    # ========================================================

    for symbol, filename in SYMBOL_FILES.items():

        symbol_start = time.perf_counter()

        print("\n")
        print("=" * 70)

        print(
            f"Loading M1 data: {symbol}"
        )

        print("=" * 70)

        filepath = (
            DATA_DIR
            / filename
        )

        try:

            # ------------------------------------------------
            # LOAD
            # ------------------------------------------------

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

        except FileNotFoundError:

            print(
                "\nERROR: Could not find:"
            )

            print(filepath)

            continue

        except Exception as error:

            print(
                "\nERROR loading data:"
            )

            print(error)

            continue

        # ====================================================
        # GAP ANALYSIS
        # ====================================================

        print("\n")
        print(
            "Analyzing M1 timestamp gaps..."
        )

        gap_start = time.perf_counter()

        gaps = analyze_gaps(
            m1
        )

        gap_time = (
            time.perf_counter()
            - gap_start
        )

        print_gap_summary(
            gaps,
            symbol,
        )

        print(
            f"\n[PERFORMANCE] "
            f"Gap analysis time: "
            f"{gap_time:.2f} seconds"
        )

        # ====================================================
        # 15M VALIDATION
        # ====================================================

        print("\n")

        print(
            "Building 15-minute "
            "session validation..."
        )

        validation_15m_start = (
            time.perf_counter()
        )

        validation_15m = (
            validate_timeframe(
                m1,
                15,
            )
        )

        validation_15m_time = (
            time.perf_counter()
            - validation_15m_start
        )

        print_timeframe_summary(
            validation_15m,
            "15M",
            symbol,
        )

        print(
            f"\n[PERFORMANCE] "
            f"15M validation time: "
            f"{validation_15m_time:.2f} seconds"
        )

        # ====================================================
        # 4H VALIDATION
        # ====================================================

        print("\n")

        print(
            "Building 4-hour "
            "session validation..."
        )

        validation_4h_start = (
            time.perf_counter()
        )

        validation_4h = (
            validate_timeframe(
                m1,
                240,
            )
        )

        validation_4h_time = (
            time.perf_counter()
            - validation_4h_start
        )

        print_timeframe_summary(
            validation_4h,
            "4H",
            symbol,
        )

        print(
            f"\n[PERFORMANCE] "
            f"4H validation time: "
            f"{validation_4h_time:.2f} seconds"
        )

        # ====================================================
        # SYMBOL TOTAL
        # ====================================================

        symbol_time = (
            time.perf_counter()
            - symbol_start
        )

        print("\n")
        print(
            f"[PERFORMANCE] "
            f"{symbol} total processing time: "
            f"{symbol_time:.2f} seconds"
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 70)

    print(
        "SESSION VALIDATION COMPLETE"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
