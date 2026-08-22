"""
EMA-ADX TRADING SYSTEM - V3.0
SESSION VALIDATOR - PRODUCTION MODEL

Purpose
-------
This is the PRODUCTION session-aware candle validator. It replaces the
exploratory/diagnostic scripts (B3.1-B3.4) with a single evidence-based
rule set and turns that rule set into candle-level status flags that
downstream code (indicators, the strategy tester) can use directly.

This module is now the ONLY 15M/4H candle builder in the project.
candle_engine.py has been retired - it built higher-timeframe candles
without any gap-cause awareness (just raw M1_Count vs Expected_M1),
which is exactly the "might build imaginary/misleading candles" risk
this production model exists to remove. Use
build_session_aware_candles() for all future 15M/4H candle needs.

The rules below are NOT guesses. They come from B3.1-B3.4 evidence
gathered on XAUUSDm M1 data (2021-08 to 2026-08, ~1.76M M1 candles):

    EXPECTED_DAILY_MAINTENANCE
        A ~60-70 minute gap that occurs once per weekday, in the
        evening. It appears at two different clock times about an
        hour apart (broker-server DST handling, not two different
        real events):
            P30_C35_D64: ~20:57 -> ~22:01   (619 occurrences)
            P33_C39_D64: ~21:57 -> ~23:01   (319 occurrences)

    EXPECTED_DAILY_ROLLOVER
        A ~2 minute gap from 23:58 -> 00:00, present from 2023
        onward (59 occurrences). Absent 2021-2022 - this is a
        genuine platform-behaviour change, not an error.

    EXPECTED_WEEKEND_CLOSURE
        Friday close -> Sunday/Monday open, or any gap that touches
        a Saturday/Sunday, or any gap >= 20 hours (covers holiday
        closures that don't land cleanly on a weekend boundary).

    UNEXPECTED_*
        Everything else. B3.4's "unclassified" control group (426
        gaps, most of them weekend-shaped but some as short as 2
        minutes) lives here until individually reviewed. These are
        NEVER silently treated as missing-but-fine - they are
        flagged so the strategy tester can see them.

HARD RULES (do not violate)
----------------------------
    1. Never invent, forward-fill, or interpolate missing M1 data.
    2. Never treat an UNEXPECTED gap as if it were EXPECTED.
    3. Every 15M/4H candle gets an explicit Status - "COMPLETE" is
       never assumed, it is proven from the underlying M1 count.
    4. Higher-timeframe candles built from an EXPECTED gap are
       labelled EXPECTED_INCOMPLETE / SESSION_CLOSED_EXPECTED, not
       COMPLETE - they are real, but the indicator layer must decide
       (later, explicitly) whether to trust them.
"""

import time
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "session_model_production"

SYMBOL_FILES = {
    "XAUUSDm": "XAUUSDm_M1_202108170000_202608131827.csv",
    "GBPUSDm": "GBPUSDm_M1_202108170000_202608131847.csv",
    "USDCHFm": "USDCHFm_M1_202108170000_202608131849.csv",
}

REQUIRED_COLUMNS = [
    "<DATE>",
    "<TIME>",
    "<OPEN>",
    "<HIGH>",
    "<LOW>",
    "<CLOSE>",
]

TIMEFRAMES = {
    "15M": 15,
    "4H": 240,
}

# Evidence-derived thresholds (see module docstring)
MAINTENANCE_MIN_MINUTES = 55
MAINTENANCE_MAX_MINUTES = 75
MAINTENANCE_HOURS = (20, 21)          # previous candle's hour must be 20:xx or 21:xx

ROLLOVER_MAX_MINUTES = 5
ROLLOVER_HOUR = 23
ROLLOVER_MIN_MINUTE = 55              # previous candle must be >= 23:55

WEEKEND_MIN_HOURS = 20                # gaps this long are closures even off-boundary


# ============================================================
# GAP CLASSIFICATION (PRODUCTION RULES)
# ============================================================

def classify_gap(previous_time, current_time):
    """
    Classify a single M1 timestamp gap using the B3.1-B3.4 evidence.

    Returns
    -------
    (classification: str, category: str)
        category is one of "NORMAL", "EXPECTED", "UNEXPECTED".
        UNEXPECTED gaps are NEVER folded into an EXPECTED bucket -
        they must be individually reviewable.
    """

    gap = current_time - previous_time
    minutes = gap.total_seconds() / 60

    if minutes <= 1:
        return "NORMAL", "NORMAL"

    # --------------------------------------------------------
    # Weekend / holiday closure
    # --------------------------------------------------------

    spans_weekend_day = (
        previous_time.weekday() >= 5
        or current_time.weekday() >= 5
        or (previous_time.weekday() == 4 and current_time.weekday() == 0)
    )

    if spans_weekend_day or minutes >= WEEKEND_MIN_HOURS * 60:
        return "EXPECTED_WEEKEND_CLOSURE", "EXPECTED"

    # --------------------------------------------------------
    # Daily rollover (23:5x -> 00:0x, ~2 minutes)
    # --------------------------------------------------------

    if (
        minutes <= ROLLOVER_MAX_MINUTES
        and previous_time.hour == ROLLOVER_HOUR
        and previous_time.minute >= ROLLOVER_MIN_MINUTE
        and current_time.date() != previous_time.date()
    ):
        return "EXPECTED_DAILY_ROLLOVER", "EXPECTED"

    # --------------------------------------------------------
    # Daily maintenance (~60-70 minutes, evening, weekday)
    # --------------------------------------------------------

    if (
        MAINTENANCE_MIN_MINUTES <= minutes <= MAINTENANCE_MAX_MINUTES
        and previous_time.weekday() < 5
        and previous_time.hour in MAINTENANCE_HOURS
    ):
        return "EXPECTED_DAILY_MAINTENANCE", "EXPECTED"

    # --------------------------------------------------------
    # Everything else is genuinely unexpected - never disguised
    # --------------------------------------------------------

    if minutes <= 5:
        return "UNEXPECTED_SHORT_GAP", "UNEXPECTED"

    if minutes <= 60:
        return "UNEXPECTED_INTRADAY_GAP", "UNEXPECTED"

    return "UNEXPECTED_LONG_GAP", "UNEXPECTED"


# ============================================================
# LOAD M1 DATA
# ============================================================

def load_m1(filepath):
    """Load one MetaTrader-exported M1 CSV (TAB/COMMA/SEMICOLON)."""

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File does not exist: {filepath}")

    print(f"\nLoading file:\n    {filepath}")

    with open(filepath, "r", encoding="utf-8-sig", errors="replace") as f:
        first_line = f.readline()

    if "\t" in first_line:
        separator, separator_name = "\t", "TAB"
    elif "," in first_line:
        separator, separator_name = ",", "COMMA"
    elif ";" in first_line:
        separator, separator_name = ";", "SEMICOLON"
    else:
        raise ValueError("Could not detect CSV separator.")

    print(f"Detected separator: {separator_name}")

    df = pd.read_csv(filepath, sep=separator, encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.strip()

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError("Missing required columns: " + ", ".join(missing_columns))

    df["Datetime"] = pd.to_datetime(
        df["<DATE>"].astype(str).str.strip() + " " + df["<TIME>"].astype(str).str.strip(),
        format="%Y.%m.%d %H:%M:%S",
        errors="coerce",
    )

    invalid_datetime = df["Datetime"].isna().sum()
    if invalid_datetime > 0:
        raise ValueError(f"Found {invalid_datetime:,} invalid Datetime values.")

    df = df.sort_values("Datetime").reset_index(drop=True)

    duplicates = df["Datetime"].duplicated().sum()
    if duplicates > 0:
        print(f"\nWARNING: {duplicates:,} duplicate timestamps found. Keeping first.")
        df = df.drop_duplicates(subset=["Datetime"], keep="first").reset_index(drop=True)

    numeric_columns = ["<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid_ohlc = df[numeric_columns].isna().any(axis=1).sum()
    if invalid_ohlc > 0:
        raise ValueError(f"Found {invalid_ohlc:,} rows with invalid OHLC values.")

    df["Open"] = df["<OPEN>"]
    df["High"] = df["<HIGH>"]
    df["Low"] = df["<LOW>"]
    df["Close"] = df["<CLOSE>"]

    print(f"\nSuccessfully loaded {len(df):,} M1 candles.")
    print(f"First candle: {df['Datetime'].iloc[0]}")
    print(f"Last candle:  {df['Datetime'].iloc[-1]}")

    return df[["Datetime", "Open", "High", "Low", "Close"]].copy()


# ============================================================
# GAP TABLE (VECTORIZED)
# ============================================================

def build_gap_table(m1):
    """
    Build a table of every M1 timestamp gap > 1 minute, classified
    with the production rules above. This table is the single
    source of truth for "why is this candle incomplete".
    """

    timestamps = m1["Datetime"]
    deltas = timestamps.diff()

    gap_mask = deltas > pd.Timedelta(minutes=1)
    gap_positions = deltas.index[gap_mask]

    if len(gap_positions) == 0:
        return pd.DataFrame(
            columns=["previous", "current", "gap_minutes", "classification", "category"]
        )

    current_times = timestamps.loc[gap_positions].reset_index(drop=True)
    previous_times = timestamps.loc[gap_positions - 1].reset_index(drop=True)
    gap_minutes = (current_times - previous_times).dt.total_seconds() / 60

    classifications = []
    categories = []

    for previous_time, current_time in zip(previous_times, current_times):
        classification, category = classify_gap(previous_time, current_time)
        classifications.append(classification)
        categories.append(category)

    return pd.DataFrame(
        {
            "previous": previous_times,
            "current": current_times,
            "gap_minutes": gap_minutes.values,
            "classification": classifications,
            "category": categories,
        }
    )


# ============================================================
# SESSION-AWARE CANDLE BUILDER (PRODUCTION)
# ============================================================

def build_session_aware_candles(m1, gap_table, timeframe_minutes):
    """
    Build higher-timeframe candles from M1 data with an explicit,
    evidence-based Status on every single candle. No candle is ever
    fabricated - OHLC values come only from real M1 data that exists
    inside that period.

    Status values
    -------------
    COMPLETE                 full M1 count, safe to use as-is
    EXPECTED_INCOMPLETE       partial M1 count, entirely explained by
                              known expected gaps (maintenance/rollover/
                              weekend edge)
    SESSION_CLOSED_EXPECTED  zero M1 candles, entirely explained by a
                              known expected closure (weekend/holiday)
    UNEXPECTED_INCOMPLETE     partial M1 count, at least one touching
                              gap is NOT in the expected rule set
    UNEXPECTED_NO_DATA        zero M1 candles and the cause is not a
                              known expected closure - needs review
    """

    freq = f"{timeframe_minutes}min"

    # --------------------------------------------------------
    # Vectorized OHLC + M1 count per period (fast, no fabrication:
    # a period with zero M1 rows simply has NaN OHLC and is dropped
    # from OHLC purposes but kept as a period for status purposes)
    # --------------------------------------------------------

    indexed = m1.set_index("Datetime")

    ohlc = indexed.resample(freq).agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
        M1_Count=("Close", "count"),
    )

    candles = ohlc.reset_index()
    candles["Expected_M1"] = timeframe_minutes
    candles["Missing_M1"] = candles["Expected_M1"] - candles["M1_Count"]
    candles["Status"] = "COMPLETE"
    candles.loc[candles["Missing_M1"] > 0, "Status"] = "PENDING"

    # --------------------------------------------------------
    # Stamp only the periods actually touched by a real gap.
    # This loops the (small) gap table, not the (large) period
    # grid - far cheaper and ties every incomplete candle back
    # to a specific, named cause.
    # --------------------------------------------------------

    touches = {}  # period Timestamp -> set of category strings

    for row in gap_table.itertuples(index=False):
        if row.category == "NORMAL":
            continue

        first_touched = row.previous.floor(freq)
        last_touched = (row.current - pd.Timedelta(minutes=1)).floor(freq)

        if last_touched < first_touched:
            last_touched = first_touched

        touched_periods = pd.date_range(first_touched, last_touched, freq=freq)

        for period in touched_periods:
            touches.setdefault(period, set()).add(row.category)

    def resolve_status(row):
        if row["Status"] != "PENDING":
            return row["Status"]

        categories = touches.get(row["Datetime"], set())

        if row["M1_Count"] == 0:
            if categories and "UNEXPECTED" not in categories:
                return "SESSION_CLOSED_EXPECTED"
            return "UNEXPECTED_NO_DATA"
        else:
            if categories and "UNEXPECTED" not in categories:
                return "EXPECTED_INCOMPLETE"
            return "UNEXPECTED_INCOMPLETE"

    pending_mask = candles["Status"] == "PENDING"
    candles.loc[pending_mask, "Status"] = candles.loc[pending_mask].apply(resolve_status, axis=1)

    candles["GapCause"] = candles["Datetime"].map(
        lambda p: "|".join(sorted(touches.get(p, []))) if p in touches else ""
    )

    return candles[
        [
            "Datetime",
            "Open",
            "High",
            "Low",
            "Close",
            "M1_Count",
            "Expected_M1",
            "Missing_M1",
            "Status",
            "GapCause",
        ]
    ]


# ============================================================
# REPORTING
# ============================================================

def print_gap_summary(gap_table, symbol):
    print("\n" + "=" * 70)
    print(f"GAP CLASSIFICATION: {symbol}")
    print("=" * 70)

    if gap_table.empty:
        print("No timestamp gaps detected.")
        return

    print("\nBy classification:")
    print("-" * 70)
    for classification, count in gap_table["classification"].value_counts().items():
        print(f"{classification:<32}{count:,}")

    unexpected = gap_table[gap_table["category"] == "UNEXPECTED"]

    print(f"\nTotal gaps: {len(gap_table):,}")
    print(f"Unexpected gaps: {len(unexpected):,}")

    if not unexpected.empty:
        print("\nUnexpected gaps (first 20):")
        print("-" * 70)
        print(
            unexpected[["previous", "current", "gap_minutes", "classification"]]
            .head(20)
            .to_string(index=False)
        )


def print_timeframe_summary(candles, timeframe_name, symbol):
    print("\n" + "-" * 70)
    print(f"{symbol} | {timeframe_name} SESSION-AWARE CANDLES")
    print("-" * 70)

    counts = candles["Status"].value_counts()
    total = len(candles)

    for status, count in counts.items():
        print(f"{status:<28}{count:>10,}  ({count / total * 100:6.2f}%)")

    needs_review = candles[candles["Status"].isin(["UNEXPECTED_INCOMPLETE", "UNEXPECTED_NO_DATA"])]
    print(f"\nCandles needing review (unexpected): {len(needs_review):,}")


def write_report(symbol, gap_table, candles_15m, candles_4h, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    gap_table.to_csv(output_dir / "gap_classification.csv", index=False)
    candles_15m.to_csv(output_dir / "candles_15m.csv", index=False)
    candles_4h.to_csv(output_dir / "candles_4h.csv", index=False)

    lines = []
    lines.append("EMA-ADX TRADING SYSTEM - V3.0")
    lines.append(f"PRODUCTION SESSION MODEL: {symbol}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Rules applied (see session_validator.py docstring for evidence):")
    lines.append("  EXPECTED_DAILY_MAINTENANCE  55-75 min, Mon-Fri, 20:xx/21:xx start")
    lines.append("  EXPECTED_DAILY_ROLLOVER      <=5 min, 23:5x -> 00:0x")
    lines.append("  EXPECTED_WEEKEND_CLOSURE     touches Sat/Sun, Fri->Mon, or >=20h")
    lines.append("  UNEXPECTED_*                 anything else - flagged, not hidden")
    lines.append("")

    for timeframe_name, candles in [("15M", candles_15m), ("4H", candles_4h)]:
        lines.append(f"{timeframe_name} candle status:")
        counts = candles["Status"].value_counts()
        total = len(candles)
        for status, count in counts.items():
            lines.append(f"  {status:<28}{count:>10,}  ({count / total * 100:6.2f}%)")
        lines.append("")

    unexpected_gaps = gap_table[gap_table["category"] == "UNEXPECTED"]
    lines.append(f"Unexpected M1 gaps: {len(unexpected_gaps):,} (see gap_classification.csv)")
    lines.append("")
    lines.append("PRODUCTION RULE FOR THE STRATEGY TESTER:")
    lines.append("  - COMPLETE candles: safe to use.")
    lines.append("  - EXPECTED_INCOMPLETE / SESSION_CLOSED_EXPECTED: real market")
    lines.append("    behaviour (maintenance/rollover/weekend). Never fabricate the")
    lines.append("    missing minutes - the backtester must skip signal generation")
    lines.append("    where indicators would depend on data that doesn't exist.")
    lines.append("  - UNEXPECTED_INCOMPLETE / UNEXPECTED_NO_DATA: exclude from")
    lines.append("    backtesting until manually reviewed in gap_classification.csv.")

    with open(output_dir / "production_session_report.txt", "w") as f:
        f.write("\n".join(lines))

    print(f"\nWrote production outputs to: {output_dir}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("       EMA-ADX TRADING SYSTEM - V3.0")
    print("       SESSION VALIDATOR - PRODUCTION MODEL")
    print("=" * 70)

    for symbol, filename in SYMBOL_FILES.items():
        symbol_start = time.perf_counter()

        print("\n" + "=" * 70)
        print(f"PROCESSING: {symbol}")
        print("=" * 70)

        filepath = DATA_DIR / filename

        try:
            m1 = load_m1(filepath)
        except FileNotFoundError:
            print(f"\nERROR: Could not find:\n{filepath}")
            continue
        except Exception as error:
            print(f"\nERROR loading data:\n{error}")
            continue

        print("\nBuilding gap table...")
        gap_table = build_gap_table(m1)
        print_gap_summary(gap_table, symbol)

        print("\nBuilding session-aware 15M candles...")
        candles_15m = build_session_aware_candles(m1, gap_table, 15)
        print_timeframe_summary(candles_15m, "15M", symbol)

        print("\nBuilding session-aware 4H candles...")
        candles_4h = build_session_aware_candles(m1, gap_table, 240)
        print_timeframe_summary(candles_4h, "4H", symbol)

        write_report(
            symbol,
            gap_table,
            candles_15m,
            candles_4h,
            OUTPUT_DIR / symbol,
        )

        symbol_time = time.perf_counter() - symbol_start
        print(f"\n[PERFORMANCE] {symbol} total time: {symbol_time:.2f} seconds")

    print("\n" + "=" * 70)
    print("PRODUCTION SESSION MODEL COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()