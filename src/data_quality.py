from pathlib import Path
import pandas as pd


# ============================================================
# V3.0 DATA QUALITY ENGINE
# Step 2: Analyze M1 timestamps and market gaps
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

FILES = {
    "XAUUSDm": RAW_DATA_DIR / "XAUUSDm_M1_202001020000_202608111933.csv",
    "GBPUSDm": RAW_DATA_DIR / "GBPUSDm_M1_202001020000_202608111937.csv",
    "USDCHFm": RAW_DATA_DIR / "USDCHFm_M1_202001020000_202608111940.csv",
}


def analyze_symbol(symbol: str, file_path: Path):
    print("\n" + "=" * 70)
    print(f"DATA QUALITY ANALYSIS: {symbol}")
    print("=" * 70)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find {symbol} data:\n{file_path}"
        )

    # MT5 files are tab-separated
    df = pd.read_csv(file_path, sep="\t")

    # Normalize column names
    df.columns = [column.strip().upper() for column in df.columns]

    column_mapping = {
        "<DATE>": "Date",
        "<TIME>": "Time",
        "<OPEN>": "Open",
        "<HIGH>": "High",
        "<LOW>": "Low",
        "<CLOSE>": "Close",
    }

    df = df.rename(columns=column_mapping)

    # Create timestamp
    df["Datetime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        errors="coerce",
    )

    # Sort
    df = df.sort_values("Datetime").reset_index(drop=True)

    # Use 2021 onward
    df = df[df["Datetime"] >= "2021-01-01"].copy()

    # --------------------------------------------------------
    # Calculate time difference between consecutive candles
    # --------------------------------------------------------

    df["Gap"] = df["Datetime"].diff()

    # A normal M1 sequence has a one-minute difference.
    gaps = df[df["Gap"] > pd.Timedelta(minutes=1)].copy()

    print(f"Total M1 candles: {len(df):,}")
    print(f"Timestamp gaps: {len(gaps):,}")

    # --------------------------------------------------------
    # Largest gaps
    # --------------------------------------------------------

    if len(gaps) > 0:

        largest_gaps = gaps.nlargest(15, "Gap")[
            ["Datetime", "Gap"]
        ].copy()

        print("\nLargest timestamp gaps:")
        print("-" * 70)

        for _, row in largest_gaps.iterrows():
            print(
                f"{row['Datetime']}  |  "
                f"{row['Gap']}"
            )

    # --------------------------------------------------------
    # Gap statistics
    # --------------------------------------------------------

        # --------------------------------------------------------
    # Gap distribution
    # --------------------------------------------------------

    if len(gaps) > 0:

        gap_minutes = (
            gaps["Gap"].dt.total_seconds() / 60
        )

        print("\nGap distribution:")
        print("-" * 70)

        # Category A: 2-5 minutes
        tiny_gaps = gaps[
            (gap_minutes >= 2)
            & (gap_minutes <= 5)
        ]

        # Category B: >5-15 minutes
        short_gaps = gaps[
            (gap_minutes > 5)
            & (gap_minutes <= 15)
        ]

        # Category C: >15-60 minutes
        medium_gaps = gaps[
            (gap_minutes > 15)
            & (gap_minutes <= 60)
        ]

        # Category D: >1-24 hours
        session_gaps = gaps[
            (gap_minutes > 60)
            & (gap_minutes < 1440)
        ]

        # Category E: >=24 hours
        major_gaps = gaps[
            gap_minutes >= 1440
        ]

        print(
            f"2-5 minutes:       {len(tiny_gaps):,}"
        )

        print(
            f">5-15 minutes:      {len(short_gaps):,}"
        )

        print(
            f">15-60 minutes:     {len(medium_gaps):,}"
        )

        print(
            f">1-24 hours:        {len(session_gaps):,}"
        )

        print(
            f">=24 hours:         {len(major_gaps):,}"
        )

        # ----------------------------------------------------
        # Basic statistics
        # ----------------------------------------------------

        print("\nGap statistics:")
        print("-" * 70)

        print(
            f"Smallest gap: "
            f"{gap_minutes.min():,.0f} minutes"
        )

        print(
            f"Largest gap:  "
            f"{gap_minutes.max():,.0f} minutes"
        )

        print(
            f"Average gap:  "
            f"{gap_minutes.mean():,.2f} minutes"
        )

        # ----------------------------------------------------
        # Percentage distribution
        # ----------------------------------------------------

        total_gaps = len(gaps)

        print("\nGap percentages:")
        print("-" * 70)

        print(
            f"2-5 minutes:       "
            f"{len(tiny_gaps) / total_gaps * 100:.2f}%"
        )

        print(
            f">5-15 minutes:      "
            f"{len(short_gaps) / total_gaps * 100:.2f}%"
        )

        print(
            f">15-60 minutes:     "
            f"{len(medium_gaps) / total_gaps * 100:.2f}%"
        )

        print(
            f">1-24 hours:        "
            f"{len(session_gaps) / total_gaps * 100:.2f}%"
        )

        print(
            f">=24 hours:         "
            f"{len(major_gaps) / total_gaps * 100:.2f}%"
        )

    # --------------------------------------------------------
    # Date range
    # --------------------------------------------------------

    print("\nDate range:")
    print("-" * 70)
    print(f"First candle: {df['Datetime'].iloc[0]}")
    print(f"Last candle:  {df['Datetime'].iloc[-1]}")

    return df


def main():

    print("\n")
    print("=" * 70)
    print("       EMA-ADX TRADING SYSTEM - V3.0")
    print("             DATA QUALITY ENGINE")
    print("=" * 70)

    for symbol, file_path in FILES.items():
        analyze_symbol(symbol, file_path)

    print("\n")
    print("=" * 70)
    print("DATA QUALITY ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()