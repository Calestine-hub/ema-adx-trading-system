from pathlib import Path
import pandas as pd


# ============================================================
# EMA-ADX TRADING SYSTEM - V3.0
# CANDLE ENGINE
#
# Step 3:
# Convert validated M1 data into:
#   - 15-minute candles
#   - 4-hour candles
#
# Rules:
#   1. Never invent missing M1 data.
#   2. Preserve broker/server timestamps.
#   3. Mark incomplete higher-timeframe candles.
#   4. Only completed candles are eligible for indicators.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


FILES = {
   "XAUUSDm": RAW_DATA_DIR / "XAUUSDm_M1_202108170000_202608131827.csv",
       "GBPUSDm": RAW_DATA_DIR / "GBPUSDm_M1_202108170000_202608131847.csv",
       "USDCHFm": RAW_DATA_DIR / "USDCHFm_M1_202108170000_202608131849.csv",
}


# ============================================================
# LOAD M1 DATA
# ============================================================

def load_m1_data(symbol: str, file_path: Path) -> pd.DataFrame:

    print("\n" + "=" * 70)
    print(f"Loading M1 data: {symbol}")
    print("=" * 70)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find {symbol} data:\n{file_path}"
        )

    df = pd.read_csv(file_path, sep="\t")
    print("\nRAW DATA DIAGNOSTIC:")
    print("-" * 70)
    print(df.head(10).to_string())
    print("\nRAW COLUMNS:")
    print(df.columns.tolist())
    print("\nRAW DATE VALUES:")
    print(df["<DATE>"].head(10).tolist())
    print("\nRAW TIME VALUES:")
    print(df["<TIME>"].head(10).tolist())

    # Normalize MT5 column names
    df.columns = [
        column.strip().upper()
        for column in df.columns
    ]

    column_mapping = {
        "<DATE>": "Date",
        "<TIME>": "Time",
        "<OPEN>": "Open",
        "<HIGH>": "High",
        "<LOW>": "Low",
        "<CLOSE>": "Close",
    }

    df = df.rename(columns=column_mapping)

    # Create broker/server timestamp
    df["Datetime"] = pd.to_datetime(
        df["Date"].astype(str)
        + " "
        + df["Time"].astype(str),
        errors="coerce",
    )

    # Remove invalid timestamps
    df = df.dropna(subset=["Datetime"])

    # Sort chronologically
    df = df.sort_values("Datetime").reset_index(drop=True)

    # V3 first test period: 2021 onward
    df = df[
        df["Datetime"] >= "2021-01-01"
    ].copy()

    # Convert prices
    price_columns = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    for column in price_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=price_columns
    )

    return df[
        [
            "Datetime",
            "Open",
            "High",
            "Low",
            "Close",
        ]
    ].copy()


# ============================================================
# BUILD HIGHER TIMEFRAME
# ============================================================

def build_timeframe(
    df: pd.DataFrame,
    timeframe: str,
    expected_minutes: int,
) -> pd.DataFrame:

    print("\n" + "-" * 70)
    print(f"Building {timeframe} candles")
    print("-" * 70)

    data = df.copy()

    # --------------------------------------------------------
    # Calculate time difference between consecutive M1 candles
    # --------------------------------------------------------

    data["PreviousDatetime"] = data["Datetime"].shift(1)

    data["MinuteGap"] = (
        data["Datetime"]
        - data["PreviousDatetime"]
    ).dt.total_seconds() / 60

    # --------------------------------------------------------
    # Identify a new continuous sequence
    #
    # A sequence breaks whenever the M1 gap is greater
    # than one minute.
    # --------------------------------------------------------

    data["Sequence"] = (
        data["MinuteGap"]
        > 1
    ).cumsum()

    # --------------------------------------------------------
    # Standard OHLC aggregation
    #
    # IMPORTANT:
    # This does NOT create missing M1 prices.
    # --------------------------------------------------------

    candles = (
        data
        .set_index("Datetime")
        .resample(timeframe)
        .agg(
            Open=("Open", "first"),
            High=("High", "max"),
            Low=("Low", "min"),
            Close=("Close", "last"),
            M1_Count=("Close", "count"),
        )
        .dropna(
            subset=[
                "Open",
                "High",
                "Low",
                "Close",
            ]
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Expected M1 candle count
    # --------------------------------------------------------

    candles["Expected_M1"] = (
        expected_minutes
    )

    candles["Complete"] = (
        candles["M1_Count"]
        == candles["Expected_M1"]
    )

    candles["Missing_M1"] = (
        candles["Expected_M1"]
        - candles["M1_Count"]
    )

    # --------------------------------------------------------
    # Quality classification
    # --------------------------------------------------------

    candles["Quality"] = "COMPLETE"

    candles.loc[
        candles["M1_Count"] < candles["Expected_M1"],
        "Quality",
    ] = "INCOMPLETE"

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    total = len(candles)

    complete = (
        candles["Complete"]
        .sum()
    )

    incomplete = (
        total - complete
    )

    print(f"Total {timeframe} candles: {total:,}")
    print(f"Complete candles:          {complete:,}")
    print(f"Incomplete candles:        {incomplete:,}")

    if incomplete > 0:

        print("\nIncomplete candle examples:")
        print("-" * 70)

        examples = candles[
            ~candles["Complete"]
        ].head(10)

        for _, row in examples.iterrows():

            print(
                f"{row['Datetime']} | "
                f"M1={row['M1_Count']}/"
                f"{row['Expected_M1']} | "
                f"Missing={row['Missing_M1']}"
            )

    return candles


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("       EMA-ADX TRADING SYSTEM - V3.0")
    print("               CANDLE ENGINE")
    print("=" * 70)

    for symbol, file_path in FILES.items():

        m1 = load_m1_data(
            symbol,
            file_path,
        )

        # ----------------------------------------------------
        # SESSION DIAGNOSTIC
        # Automatically inspect the FIRST DATE actually
        # present in the dataset.
        # ----------------------------------------------------

        first_date = m1["Datetime"].dt.normalize().iloc[0]

        diagnostic = m1[
            (m1["Datetime"] >= first_date)
            & (
                m1["Datetime"]
                < first_date + pd.Timedelta(days=1)
            )
        ].copy()

        print("\nSESSION DIAGNOSTIC:")
        print("-" * 70)

        print(
            diagnostic[
                [
                    "Datetime",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                ]
            ].head(30).to_string(index=False)
        )

        print(
            f"\nTotal M1 candles on "
            f"{first_date.date()}: {len(diagnostic):,}"
        )

        if len(diagnostic) > 0:

            print(
                f"First M1 candle: "
                f"{diagnostic['Datetime'].iloc[0]}"
            )

            print(
                f"Last M1 candle:  "
                f"{diagnostic['Datetime'].iloc[-1]}"
            )

        # ----------------------------------------------------
        # Build 15-minute candles
        # ----------------------------------------------------

        candles_15m = build_timeframe(
            m1,
            "15min",
            15,
        )

        # ----------------------------------------------------
        # Build 4-hour candles
        # ----------------------------------------------------

        candles_4h = build_timeframe(
            m1,
            "4h",
            240,
        )

        print("\n")
        print(
            f"{symbol} candle construction complete."
        )

    print("\n")
    print("=" * 70)
    print("CANDLE ENGINE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()