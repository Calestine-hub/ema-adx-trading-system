from pathlib import Path
import pandas as pd


# ============================================================
# V3.0 DATA ENGINE
# Step 1: Load and validate raw M1 market data
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

FILES = {
    "XAUUSDm": RAW_DATA_DIR / "XAUUSDm_M1_202001020000_202608111933.csv",
    "GBPUSDm": RAW_DATA_DIR / "GBPUSDm_M1_202001020000_202608111937.csv",
    "USDCHFm": RAW_DATA_DIR / "USDCHFm_M1_202001020000_202608111940.csv",
}


def load_market_data(symbol: str, file_path: Path) -> pd.DataFrame:
    """Load one raw MT5 M1 CSV and perform basic validation."""

    print(f"\n{'=' * 60}")
    print(f"Loading {symbol}")
    print(f"{'=' * 60}")

    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find {symbol} data at:\n{file_path}"
        )

    # MT5 exports tab-separated data
    df = pd.read_csv(file_path, sep="\t")

    print(f"Rows loaded: {len(df):,}")
    print(f"Original columns: {list(df.columns)}")

    # Normalize MT5 column names
    df.columns = [column.strip().upper() for column in df.columns]

    column_mapping = {
        "<DATE>": "Date",
        "<TIME>": "Time",
        "<OPEN>": "Open",
        "<HIGH>": "High",
        "<LOW>": "Low",
        "<CLOSE>": "Close",
        "<TICKVOL>": "Tick Volume",
        "<VOL>": "Volume",
        "<SPREAD>": "Spread",
    }

    df = df.rename(columns=column_mapping)

    print(f"Normalized columns: {list(df.columns)}")

    # Required columns
    required_columns = [
        "Date",
        "Time",
        "Open",
        "High",
        "Low",
        "Close",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{symbol} is missing required columns: {missing_columns}"
        )

    # Combine Date + Time
    df["Datetime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        errors="coerce",
    )

    invalid_dates = df["Datetime"].isna().sum()

    if invalid_dates:
        raise ValueError(
            f"{symbol}: {invalid_dates:,} invalid timestamps found."
        )

    # Sort chronologically
    df = df.sort_values("Datetime").reset_index(drop=True)

    # Use 2021 onward for the first V3 test
    df = df[df["Datetime"] >= "2021-01-01"].copy()

    # Convert price columns to numeric
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

    # Check missing prices
    missing_prices = df[price_columns].isna().sum()

    if missing_prices.any():
        raise ValueError(
            f"{symbol}: Missing price values:\n{missing_prices}"
        )

    # Check duplicate timestamps
    duplicates = df["Datetime"].duplicated().sum()

    # Check OHLC integrity
    invalid_ohlc = (
        (df["High"] < df["Open"])
        | (df["High"] < df["Close"])
        | (df["High"] < df["Low"])
        | (df["Low"] > df["Open"])
        | (df["Low"] > df["Close"])
    ).sum()

    # Report
    print(f"Valid rows after 2020 removal: {len(df):,}")
    print(f"First valid candle: {df['Datetime'].iloc[0]}")
    print(f"Last candle:         {df['Datetime'].iloc[-1]}")
    print(f"Duplicate timestamps: {duplicates:,}")
    print(f"Invalid OHLC candles: {invalid_ohlc:,}")

    if duplicates:
        print("WARNING: Duplicate timestamps detected.")

    if invalid_ohlc:
        print("WARNING: Invalid OHLC candles detected.")

    # Keep useful columns
    keep_columns = [
        "Datetime",
        "Open",
        "High",
        "Low",
        "Close",
    ]

    optional_columns = [
        "Tick Volume",
        "Volume",
        "Spread",
    ]

    for column in optional_columns:
        if column in df.columns:
            keep_columns.append(column)

    return df[keep_columns].copy()


def main():
    print("\n")
    print("=" * 60)
    print("        EMA-ADX TRADING SYSTEM - V3.0")
    print("              DATA ENGINE TEST")
    print("=" * 60)

    datasets = {}

    for symbol, file_path in FILES.items():
        datasets[symbol] = load_market_data(
            symbol,
            file_path,
        )

    print("\n")
    print("=" * 60)
    print("ALL DATASETS LOADED SUCCESSFULLY")
    print("=" * 60)

    for symbol, df in datasets.items():
        print(
            f"{symbol:10} -> {len(df):,} valid M1 candles"
        )

    print("\nV3.0 data engine test complete.")


if __name__ == "__main__":
    main()
