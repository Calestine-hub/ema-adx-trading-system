"""
EMA-ADX TRADING SYSTEM - V3.0
B2 CONTEXTUAL GAP AUDIT

Purpose:
    Inspect the actual M1 candles immediately before and after
    recurring timestamp gaps.

This diagnostic is designed to answer:

    1. What candles exist immediately before a recurring gap?
    2. What candles exist immediately after the gap?
    3. Does the gap occur on normal trading days?
    4. Is the surrounding sequence consistent?
    5. Are additional gaps clustered around the same boundary?
    6. Does the pattern persist across multiple years?

IMPORTANT:

    This script is DIAGNOSTIC ONLY.

    It does NOT:
        - modify raw M1 data
        - fill missing prices
        - delete candles
        - modify session_validator.py
        - classify a gap as definitely valid/invalid
        - create trading signals
        - build candles for backtesting

    No trading decision should be made from this script alone.
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


# Number of strongest recurring timestamp transitions to audit.
TOP_PATTERNS = 10


# Minimum number of occurrences for a pattern to be considered.
MIN_PATTERN_OCCURRENCES = 20


# Number of real historical examples to inspect per pattern.
EXAMPLES_PER_PATTERN = 5


# Number of M1 candles to show immediately before the gap.
CANDLES_BEFORE = 5


# Number of M1 candles to show immediately after the gap.
CANDLES_AFTER = 5


# Additional search window around each gap.
#
# This lets us detect whether the gap is part of a larger
# cluster of missing candles.
CONTEXT_WINDOW_MINUTES = 15


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

    invalid_datetime = (
        df["Datetime"].isna().sum()
    )

    if invalid_datetime:

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
    # Duplicate timestamps
    # --------------------------------------------------------

    duplicates = (
        df["Datetime"]
        .duplicated()
        .sum()
    )

    if duplicates:

        print(
            f"WARNING: "
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
    # Numeric fields
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

    if "<TICKVOL>" in df.columns:

        df["<TICKVOL>"] = pd.to_numeric(
            df["<TICKVOL>"],
            errors="coerce",
        )

    if "<SPREAD>" in df.columns:

        df["<SPREAD>"] = pd.to_numeric(
            df["<SPREAD>"],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Standardized names
    # --------------------------------------------------------

    df["Open"] = df["<OPEN>"]
    df["High"] = df["<HIGH>"]
    df["Low"] = df["<LOW>"]
    df["Close"] = df["<CLOSE>"]

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
                "previous_index",
                "current_index",
                "previous",
                "current",
                "gap_minutes",
                "previous_date",
                "previous_weekday",
                "current_weekday",
                "previous_time",
                "current_time",
                "year",
            ]
        )

    previous_indices = (
        indices - 1
    )

    result = pd.DataFrame(
        {
            "previous_index":
                previous_indices,
            "current_index":
                indices,
        }
    )

    result["previous"] = (
        m1.loc[
            previous_indices,
            "Datetime"
        ].to_numpy()
    )

    result["current"] = (
        m1.loc[
            indices,
            "Datetime"
        ].to_numpy()
    )

    result["gap_minutes"] = (
        result["current"]
        - result["previous"]
    ).dt.total_seconds() / 60

    result["previous_date"] = (
        result["previous"]
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
# FIND STRONGEST PATTERNS
# ============================================================

def find_strongest_patterns(gaps):

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
            median_gap=(
                "gap_minutes",
                "median",
            ),
            min_gap=(
                "gap_minutes",
                "min",
            ),
            max_gap=(
                "gap_minutes",
                "max",
            ),
            unique_dates=(
                "previous_date",
                "nunique",
            ),
            years=(
                "year",
                "nunique",
            ),
        )
        .sort_values(
            [
                "occurrences",
                "unique_dates",
            ],
            ascending=False,
        )
    )

    patterns = patterns[
        patterns["occurrences"]
        >= MIN_PATTERN_OCCURRENCES
    ]

    return patterns.head(
        TOP_PATTERNS
    ).reset_index(drop=True)


# ============================================================
# FORMAT M1 ROW
# ============================================================

def prepare_context_table(
    context,
):

    if context.empty:
        return context

    output_columns = [
        "Datetime",
        "weekday",
        "minutes_from_gap",
        "Open",
        "High",
        "Low",
        "Close",
    ]

    context = context.copy()

    context["weekday"] = (
        context["Datetime"]
        .dt.day_name()
    )

    context["minutes_from_gap"] = (
        context["Datetime"]
        - context["_gap_current_time"]
    ).dt.total_seconds() / 60

    if "<TICKVOL>" in context.columns:

        context["TickVol"] = (
            context["<TICKVOL>"]
        )

        output_columns.append(
            "TickVol"
        )

    if "<SPREAD>" in context.columns:

        context["Spread"] = (
            context["<SPREAD>"]
        )

        output_columns.append(
            "Spread"
        )

    output_columns = [
        column
        for column in output_columns
        if column in context.columns
    ]

    return context[output_columns]


# ============================================================
# INSPECT ONE GAP
# ============================================================

def inspect_gap(
    m1,
    gap_row,
):

    previous_index = int(
        gap_row["previous_index"]
    )

    current_index = int(
        gap_row["current_index"]
    )

    previous_time = (
        gap_row["previous"]
    )

    current_time = (
        gap_row["current"]
    )

    # --------------------------------------------------------
    # Direct neighboring candles
    # --------------------------------------------------------

    before_start = max(
        0,
        previous_index
        - CANDLES_BEFORE
        + 1,
    )

    before_end = (
        previous_index + 1
    )

    after_start = (
        current_index
    )

    after_end = min(
        len(m1),
        current_index
        + CANDLES_AFTER,
    )

    before = (
        m1.iloc[
            before_start:before_end
        ].copy()
    )

    after = (
        m1.iloc[
            after_start:after_end
        ].copy()
    )

    # --------------------------------------------------------
    # Wider context
    # --------------------------------------------------------

    window_start = (
        previous_time
        - pd.Timedelta(
            minutes=CONTEXT_WINDOW_MINUTES
        )
    )

    window_end = (
        current_time
        + pd.Timedelta(
            minutes=CONTEXT_WINDOW_MINUTES
        )
    )

    context = m1[
        (
            m1["Datetime"]
            >= window_start
        )
        &
        (
            m1["Datetime"]
            <= window_end
        )
    ].copy()

    context["_gap_current_time"] = (
        current_time
    )

    context_table = (
        prepare_context_table(
            context
        )
    )

    return (
        before,
        after,
        context_table,
    )


# ============================================================
# PRINT BASIC GAP INFORMATION
# ============================================================

def print_gap_header(
    symbol,
    gap_number,
    total_examples,
    gap_row,
):

    print("\n")
    print("=" * 90)

    print(
        f"{symbol} | GAP EXAMPLE "
        f"{gap_number}/{total_examples}"
    )

    print("=" * 90)

    print(
        f"Previous candle: "
        f"{gap_row['previous']}"
    )

    print(
        f"Next candle:     "
        f"{gap_row['current']}"
    )

    print(
        f"Gap duration:    "
        f"{gap_row['gap_minutes']:.0f} minutes"
    )

    print(
        f"Previous day:    "
        f"{gap_row['previous_weekday']}"
    )

    print(
        f"Current day:     "
        f"{gap_row['current_weekday']}"
    )


# ============================================================
# PRINT CANDLE TABLE
# ============================================================

def print_candle_table(
    title,
    table,
):

    print("\n")
    print(title)
    print("-" * 90)

    if table.empty:

        print(
            "No candles found."
        )

        return

    display = table.copy()

    if "Datetime" in display.columns:

        display["Datetime"] = (
            display["Datetime"]
            .dt.strftime(
                "%Y-%m-%d %H:%M"
            )
        )

    if "minutes_from_gap" in display.columns:

        display["minutes_from_gap"] = (
            display["minutes_from_gap"]
            .map(
                lambda x:
                f"{x:+.0f}"
            )
        )

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "TickVol",
        "Spread",
    ]

    for column in numeric_columns:

        if column in display.columns:

            if column in [
                "TickVol",
                "Spread",
            ]:

                display[column] = (
                    display[column]
                    .map(
                        lambda x:
                        (
                            ""
                            if pd.isna(x)
                            else f"{x:.0f}"
                        )
                    )
                )

            else:

                display[column] = (
                    display[column]
                    .map(
                        lambda x:
                        (
                            ""
                            if pd.isna(x)
                            else f"{x:.6f}"
                        )
                    )
                )

    print(
        display.to_string(
            index=False
        )
    )


# ============================================================
# CHECK NEIGHBORING GAPS
# ============================================================

def analyze_gap_cluster(
    m1,
    gap_row,
):

    previous_time = (
        gap_row["previous"]
    )

    current_time = (
        gap_row["current"]
    )

    window_start = (
        previous_time
        - pd.Timedelta(
            minutes=CONTEXT_WINDOW_MINUTES
        )
    )

    window_end = (
        current_time
        + pd.Timedelta(
            minutes=CONTEXT_WINDOW_MINUTES
        )
    )

    local = m1[
        (
            m1["Datetime"]
            >= window_start
        )
        &
        (
            m1["Datetime"]
            <= window_end
        )
    ][
        "Datetime"
    ]

    if len(local) < 2:

        return []

    differences = (
        local.diff()
    )

    cluster_gaps = []

    for index in differences.index:

        gap = differences.loc[index]

        if (
            pd.notna(gap)
            and gap
            > pd.Timedelta(minutes=1)
        ):

            previous = local.loc[
                index - 1
            ]

            current = local.loc[
                index
            ]

            cluster_gaps.append(
                {
                    "previous": previous,
                    "current": current,
                    "gap_minutes":
                        gap.total_seconds()
                        / 60,
                }
            )

    return cluster_gaps


# ============================================================
# PRINT PATTERN SUMMARY
# ============================================================

def print_pattern_summary(
    pattern_number,
    pattern,
):

    print("\n")
    print("#" * 90)

    print(
        f"PATTERN #{pattern_number}"
    )

    print("#" * 90)

    print(
        f"Transition: "
        f"{pattern['previous_time']} "
        f"-> "
        f"{pattern['current_time']}"
    )

    print(
        f"Occurrences: "
        f"{pattern['occurrences']:,}"
    )

    print(
        f"Unique dates: "
        f"{pattern['unique_dates']:,}"
    )

    print(
        f"Years present: "
        f"{pattern['years']}"
    )

    print(
        f"Gap range: "
        f"{pattern['min_gap']:.0f}"
        f" - "
        f"{pattern['max_gap']:.0f}"
        f" minutes"
    )

    print(
        f"Median gap: "
        f"{pattern['median_gap']:.0f}"
        f" minutes"
    )


# ============================================================
# AUDIT ONE PATTERN
# ============================================================

def audit_pattern(
    m1,
    gaps,
    pattern,
    symbol,
    pattern_number,
):

    print_pattern_summary(
        pattern_number,
        pattern,
    )

    # --------------------------------------------------------
    # Find all examples
    # --------------------------------------------------------

    matching = gaps[
        (
            gaps["previous_time"]
            == pattern["previous_time"]
        )
        &
        (
            gaps["current_time"]
            == pattern["current_time"]
        )
    ].copy()

    # --------------------------------------------------------
    # Select examples across historical range
    #
    # We deliberately select dates spread through the
    # dataset rather than simply taking the first 5.
    # --------------------------------------------------------

    matching = (
        matching
        .sort_values("previous")
        .reset_index(drop=True)
    )

    if len(matching) <= EXAMPLES_PER_PATTERN:

        selected = matching

    else:

        positions = (
            pd.Series(
                range(
                    0,
                    len(matching),
                    max(
                        1,
                        len(matching)
                        // EXAMPLES_PER_PATTERN,
                    ),
                )
            )
            .head(
                EXAMPLES_PER_PATTERN
            )
            .tolist()
        )

        selected = matching.iloc[
            positions
        ]

    # --------------------------------------------------------
    # Inspect each real gap
    # --------------------------------------------------------

    total = len(selected)

    for number, (_, gap_row) in enumerate(
        selected.iterrows(),
        start=1,
    ):

        print_gap_header(
            symbol,
            number,
            total,
            gap_row,
        )

        before, after, context = (
            inspect_gap(
                m1,
                gap_row,
            )
        )

        # ----------------------------------------------------
        # Before
        # ----------------------------------------------------

        before_display = (
            before[
                [
                    "Datetime",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                ]
                + (
                    ["<TICKVOL>"]
                    if "<TICKVOL>" in before.columns
                    else []
                )
                + (
                    ["<SPREAD>"]
                    if "<SPREAD>" in before.columns
                    else []
                )
            ]
            .copy()
        )

        if "<TICKVOL>" in before_display.columns:

            before_display["TickVol"] = (
                before_display["<TICKVOL>"]
            )

            before_display = (
                before_display.drop(
                    columns=["<TICKVOL>"]
                )
            )

        if "<SPREAD>" in before_display.columns:

            before_display["Spread"] = (
                before_display["<SPREAD>"]
            )

            before_display = (
                before_display.drop(
                    columns=["<SPREAD>"]
                )
            )

        print_candle_table(
            "M1 CANDLES BEFORE GAP",
            before_display,
        )

        # ----------------------------------------------------
        # After
        # ----------------------------------------------------

        after_display = (
            after[
                [
                    "Datetime",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                ]
                + (
                    ["<TICKVOL>"]
                    if "<TICKVOL>" in after.columns
                    else []
                )
                + (
                    ["<SPREAD>"]
                    if "<SPREAD>" in after.columns
                    else []
                )
            ]
            .copy()
        )

        if "<TICKVOL>" in after_display.columns:

            after_display["TickVol"] = (
                after_display["<TICKVOL>"]
            )

            after_display = (
                after_display.drop(
                    columns=["<TICKVOL>"]
                )
            )

        if "<SPREAD>" in after_display.columns:

            after_display["Spread"] = (
                after_display["<SPREAD>"]
            )

            after_display = (
                after_display.drop(
                    columns=["<SPREAD>"]
                )
            )

        print_candle_table(
            "M1 CANDLES AFTER GAP",
            after_display,
        )

        # ----------------------------------------------------
        # Cluster analysis
        # ----------------------------------------------------

        cluster = (
            analyze_gap_cluster(
                m1,
                gap_row,
            )
        )

        print("\n")
        print(
            "GAPS INSIDE LOCAL "
            "CONTEXT WINDOW"
        )

        print("-" * 90)

        if not cluster:

            print(
                "No additional gaps detected "
                "inside the context window."
            )

        else:

            for item in cluster:

                print(
                    f"{item['previous']} "
                    f"-> "
                    f"{item['current']} "
                    f"({item['gap_minutes']:.0f} min)"
                )

        # ----------------------------------------------------
        # Wider contextual table
        # ----------------------------------------------------

        print_candle_table(
            "FULL LOCAL M1 CONTEXT",
            context,
        )


# ============================================================
# PROCESS SYMBOL
# ============================================================

def process_symbol(
    symbol,
    filename,
):

    print("\n")
    print("=" * 90)

    print(
        f"PROCESSING {symbol}"
    )

    print("=" * 90)

    start = time.perf_counter()

    filepath = (
        DATA_DIR
        / filename
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    try:

        m1 = load_m1(
            filepath
        )

    except Exception as error:

        print(
            f"\nERROR loading {symbol}:"
        )

        print(error)

        return

    # --------------------------------------------------------
    # Build gaps
    # --------------------------------------------------------

    print(
        "\nBuilding timestamp gap table..."
    )

    gaps = build_gap_table(
        m1
    )

    print(
        f"Detected "
        f"{len(gaps):,} gaps > 1 minute."
    )

    # --------------------------------------------------------
    # Find strongest patterns
    # --------------------------------------------------------

    patterns = (
        find_strongest_patterns(
            gaps
        )
    )

    if patterns.empty:

        print(
            "\nNo recurring patterns "
            "met the minimum occurrence threshold."
        )

        return

    print("\n")
    print(
        "STRONGEST RECURRING GAP PATTERNS"
    )

    print("-" * 90)

    print(
        f"{'Pattern':<8}"
        f"{'Transition':<18}"
        f"{'Occurrences':>13}"
        f"{'Dates':>10}"
        f"{'Years':>8}"
        f"{'Median':>10}"
        f"{'Range':>15}"
    )

    print("-" * 90)

    for number, (_, pattern) in enumerate(
        patterns.iterrows(),
        start=1,
    ):

        transition = (
            f"{pattern['previous_time']}"
            f" -> "
            f"{pattern['current_time']}"
        )

        gap_range = (
            f"{pattern['min_gap']:.0f}"
            f"-"
            f"{pattern['max_gap']:.0f}"
        )

        print(
            f"{number:<8}"
            f"{transition:<18}"
            f"{pattern['occurrences']:>13,}"
            f"{pattern['unique_dates']:>10,}"
            f"{pattern['years']:>8}"
            f"{pattern['median_gap']:>10.0f}"
            f"{gap_range:>15}"
        )

    # --------------------------------------------------------
    # Detailed audits
    # --------------------------------------------------------

    for number, (_, pattern) in enumerate(
        patterns.iterrows(),
        start=1,
    ):

        audit_pattern(
            m1,
            gaps,
            pattern,
            symbol,
            number,
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    print("\n")
    print("=" * 90)

    print(
        f"{symbol} CONTEXTUAL GAP AUDIT COMPLETE"
    )

    print(
        f"Runtime: {elapsed:.2f} seconds"
    )

    print("=" * 90)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)

    print(
        "       EMA-ADX TRADING SYSTEM - V3.0"
    )

    print(
        "       B2 CONTEXTUAL GAP AUDIT"
    )

    print("=" * 90)

    print("\n")
    print(
        "Diagnostic only."
    )

    print(
        "Raw data will NOT be modified."
    )

    print(
        "No missing candles will be filled."
    )

    print(
        "No session rules will be changed."
    )

    print(
        "No trading logic will be executed."
    )

    for symbol, filename in (
        SYMBOL_FILES.items()
    ):

        process_symbol(
            symbol,
            filename,
        )

    print("\n")
    print("=" * 90)

    print(
        "B2 CONTEXTUAL GAP AUDIT COMPLETE"
    )

    print("=" * 90)

    print("\n")
    print(
        "NEXT STEP:"
    )

    print(
        "Review the actual M1 sequences around "
        "the recurring gap patterns before "
        "building the historical session model."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()