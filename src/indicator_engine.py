"""
EMA-ADX TRADING SYSTEM - V3.0
INDICATOR ENGINE - NORMAL BRANCH (XAUUSDm, 15M entry / 4H bias)

Purpose
-------
Reproduce, on real historical candles, the indicator layer of the
Pine Script "EMA-ADX Pullback V4.0" NORMAL branch:

    - 15/50 EMA on the entry timeframe (15M)
    - ADX / DI+ / DI- (Wilder DMI, length 14) on the entry timeframe
    - ATR(14) on the entry timeframe, and the EMA "touch zone"
    - 4H 200 SMA bias, resolved WITHOUT repainting (only the close
      of a fully-closed 4H bar is ever visible to a given 15M bar -
      mirrors the Pine script's request.security(..., close[1],
      lookahead_on) anti-repaint pattern)
    - EMA15/EMA50 crossover detection + lookback

Data source
-----------
ALL candles come from session_validator.build_session_aware_candles().
candle_engine.py is retired - this is the only candle builder in the
project now.

Completeness policy (confirmed with user)
------------------------------------------
    - Indicators (EMA/ADX/DI/ATR/SMA) are computed continuously over
      COMPLETE + EXPECTED_INCOMPLETE + UNEXPECTED_INCOMPLETE bars -
      these all have real OHLC, so dropping them would carve
      artificial holes into a rolling calculation that the live
      platform itself would not have.
    - SESSION_CLOSED_EXPECTED / UNEXPECTED_NO_DATA bars have no OHLC
      at all and simply do not exist as rows (never fabricated).
    - A separate "SignalEligible" flag marks whether a bar is
      allowed to originate a NEW trading signal: TRUE only when the
      15M bar's own Status == COMPLETE AND the 4H bias bar currently
      in effect also has Status == COMPLETE. This flag does NOT
      blank out the indicator values themselves - the strategy layer
      (built next) is what actually uses it to gate entries.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import session_validator as sv


# ============================================================
# CONFIGURATION (defaults from the Pine script, NORMAL branch)
# ============================================================

EMA_FAST_LEN = 15
EMA_SLOW_LEN = 50
SMA_HTF_LEN = 200

ADX_LEN = 14

ATR_LEN = 14
EMA_ZONE_ATR = 0.15

CROSS_LOOKBACK = 10

ENTRY_TIMEFRAME_MINUTES = 15
HTF_TIMEFRAME_MINUTES = 240


# ============================================================
# WILDER-STYLE SMOOTHING (matches Pine's ta.rma / ta.dmi / ta.atr)
# ============================================================

def rma(series: pd.Series, length: int) -> pd.Series:
    """
    Wilder's moving average, exactly what ta.atr / ta.dmi / ta.rma
    use internally in Pine. alpha = 1/length, seeded with the plain
    mean of the first `length` values (Pine's actual seed behaviour).
    """

    values = series.to_numpy(dtype=float)
    out = np.full(values.shape, np.nan)

    if len(values) < length:
        return pd.Series(out, index=series.index)

    seed = np.nanmean(values[:length])
    out[length - 1] = seed

    alpha = 1.0 / length

    for i in range(length, len(values)):
        prev = out[i - 1]
        current = values[i]
        if np.isnan(prev):
            out[i] = current
        else:
            out[i] = prev + alpha * (current - prev)

    return pd.Series(out, index=series.index)


def ema(series: pd.Series, length: int) -> pd.Series:
    """
    Pine's ta.ema: seeded with the SMA of the first `length` values,
    then the standard recursive EMA formula (alpha = 2/(length+1)).
    """

    values = series.to_numpy(dtype=float)
    out = np.full(values.shape, np.nan)

    if len(values) < length:
        return pd.Series(out, index=series.index)

    seed = np.nanmean(values[:length])
    out[length - 1] = seed

    alpha = 2.0 / (length + 1)

    for i in range(length, len(values)):
        prev = out[i - 1]
        current = values[i]
        if np.isnan(prev):
            out[i] = current
        else:
            out[i] = prev + alpha * (current - prev)

    return pd.Series(out, index=series.index)


# ============================================================
# ATR
# ============================================================

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)

    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr


def atr(df: pd.DataFrame, length: int) -> pd.Series:
    tr = true_range(df)
    return rma(tr, length)


# ============================================================
# DMI / ADX (Wilder)
# ============================================================

def dmi(df: pd.DataFrame, length: int):
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    tr = true_range(df)

    tr_rma = rma(tr, length)
    plus_dm_rma = rma(plus_dm, length)
    minus_dm_rma = rma(minus_dm, length)

    di_plus = 100 * (plus_dm_rma / tr_rma)
    di_minus = 100 * (minus_dm_rma / tr_rma)

    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus)
    adx = rma(dx, length)

    return di_plus, di_minus, adx


# ============================================================
# BUILD ENTRY-TIMEFRAME INDICATORS (15M)
# ============================================================

def build_entry_indicators(candles_15m: pd.DataFrame) -> pd.DataFrame:
    """
    candles_15m must come straight from
    session_validator.build_session_aware_candles(m1, gap_table, 15).

    Bars with no OHLC (SESSION_CLOSED_EXPECTED / UNEXPECTED_NO_DATA)
    are dropped before indicator calculation - they never existed as
    real price bars, so there is nothing to compute on and nothing
    to fabricate.
    """

    df = candles_15m.copy()
    df = df[df["Close"].notna()].reset_index(drop=True)

    df["EMA_Fast"] = ema(df["Close"], EMA_FAST_LEN)
    df["EMA_Slow"] = ema(df["Close"], EMA_SLOW_LEN)

    df["ATR"] = atr(df, ATR_LEN)

    di_plus, di_minus, adx_val = dmi(df, ADX_LEN)
    df["DI_Plus"] = di_plus
    df["DI_Minus"] = di_minus
    df["ADX"] = adx_val
    df["DI_Net"] = df["DI_Plus"] - df["DI_Minus"]

    # EMA touch zone (pullback detection)
    ema_zone = df["ATR"] * EMA_ZONE_ATR

    touch_fast = (df["Low"] <= df["EMA_Fast"] + ema_zone) & (df["High"] >= df["EMA_Fast"] - ema_zone)
    touch_slow = (df["Low"] <= df["EMA_Slow"] + ema_zone) & (df["High"] >= df["EMA_Slow"] - ema_zone)
    df["Touching_EMA"] = touch_fast | touch_slow

    df["Touch_Name"] = np.select(
        [touch_fast & touch_slow, touch_fast, touch_slow],
        ["15 & 50 EMA", "15 EMA", "50 EMA"],
        default="NONE",
    )

    # EMA alignment
    df["Bullish_EMA"] = df["EMA_Fast"] > df["EMA_Slow"]
    df["Bearish_EMA"] = df["EMA_Fast"] < df["EMA_Slow"]

    # Crossover detection + lookback (matches ta.crossover/crossunder
    # + ta.barssince semantics)
    fast_above = df["EMA_Fast"] > df["EMA_Slow"]
    fast_above_prev = fast_above.shift(1)

    bullish_cross = fast_above & (~fast_above_prev.fillna(False))
    bearish_cross = (~fast_above) & fast_above_prev.fillna(False)

    df["Bullish_Cross"] = bullish_cross
    df["Bearish_Cross"] = bearish_cross

    df["Bars_Since_Bull_Cross"] = _bars_since(bullish_cross)
    df["Bars_Since_Bear_Cross"] = _bars_since(bearish_cross)

    df["Recent_Bull_Cross"] = df["Bars_Since_Bull_Cross"] <= CROSS_LOOKBACK
    df["Recent_Bear_Cross"] = df["Bars_Since_Bear_Cross"] <= CROSS_LOOKBACK

    return df


def _bars_since(event_series: pd.Series) -> pd.Series:
    """Equivalent of Pine's ta.barssince - NaN/inf until the first event."""

    out = np.full(len(event_series), np.inf)
    last_seen = -1

    events = event_series.to_numpy()

    for i in range(len(events)):
        if events[i]:
            last_seen = i
        out[i] = (i - last_seen) if last_seen >= 0 else np.inf

    return pd.Series(out, index=event_series.index)


# ============================================================
# BUILD HTF BIAS (4H 200 SMA, NO REPAINT)
# ============================================================

def build_htf_bias(candles_4h: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 4H 200 SMA on real 4H bars only (drops bars with no
    OHLC). Returns a frame indexed by "AvailableFrom" - the timestamp
    at which that 4H bar's Close/SMA/Status first becomes knowable to
    a 15M bar without repainting (i.e. the 4H bar's own close time).
    """

    df = candles_4h.copy()
    df = df[df["Close"].notna()].reset_index(drop=True)

    df["SMA200"] = df["Close"].rolling(SMA_HTF_LEN).mean()

    df["Bullish_HTF"] = df["Close"] > df["SMA200"]
    df["Bearish_HTF"] = df["Close"] < df["SMA200"]

    df["AvailableFrom"] = df["Datetime"] + pd.Timedelta(minutes=HTF_TIMEFRAME_MINUTES)

    return df[
        ["AvailableFrom", "Datetime", "Close", "SMA200", "Bullish_HTF", "Bearish_HTF", "Status"]
    ].rename(
        columns={
            "Datetime": "HTF_Datetime",
            "Close": "HTF_Close",
            "Status": "HTF_Status",
        }
    )


# ============================================================
# MERGE HTF BIAS ONTO ENTRY TIMEFRAME (ANTI-REPAINT ASOF-MERGE)
# ============================================================

def attach_htf_bias(entry_df: pd.DataFrame, htf_df: pd.DataFrame) -> pd.DataFrame:
    """
    For every 15M bar, attach the bias of the most recently FULLY
    CLOSED 4H bar as of that 15M bar's open time. This is exactly
    what the Pine script's close[1] + lookahead_on trick achieves -
    the 4H bar that is still forming is never visible.
    """

    entry_sorted = entry_df.sort_values("Datetime").reset_index(drop=True)
    htf_sorted = htf_df.sort_values("AvailableFrom").reset_index(drop=True)

    merged = pd.merge_asof(
        entry_sorted,
        htf_sorted,
        left_on="Datetime",
        right_on="AvailableFrom",
        direction="backward",
    )

    return merged


# ============================================================
# SIGNAL ELIGIBILITY (COMPLETENESS GATE)
# ============================================================

def add_signal_eligibility(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    SignalEligible = True only when the entry-TF bar AND the 4H bias
    bar currently in effect are both Status == COMPLETE. This never
    touches the indicator values themselves - it only marks which
    bars are allowed to originate a NEW trading signal. Built as its
    own explicit column so the strategy layer can decide, rather than
    the indicator layer silently deciding for it.
    """

    df = merged_df.copy()

    df["SignalEligible"] = (
        (df["Status"] == "COMPLETE") & (df["HTF_Status"] == "COMPLETE")
    )

    return df


# ============================================================
# FULL PIPELINE (ONE SYMBOL)
# ============================================================

def build_normal_branch_indicators(symbol: str = "XAUUSDm") -> pd.DataFrame:
    filepath = sv.DATA_DIR / sv.SYMBOL_FILES[symbol]

    m1 = sv.load_m1(filepath)
    gap_table = sv.build_gap_table(m1)

    candles_15m = sv.build_session_aware_candles(m1, gap_table, ENTRY_TIMEFRAME_MINUTES)
    candles_4h = sv.build_session_aware_candles(m1, gap_table, HTF_TIMEFRAME_MINUTES)

    entry_df = build_entry_indicators(candles_15m)
    htf_df = build_htf_bias(candles_4h)

    merged = attach_htf_bias(entry_df, htf_df)
    merged = add_signal_eligibility(merged)

    return merged


# ============================================================
# MAIN (DIAGNOSTIC RUN)
# ============================================================

def main():
    print("=" * 70)
    print("       EMA-ADX TRADING SYSTEM - V3.0")
    print("       INDICATOR ENGINE - NORMAL BRANCH")
    print("=" * 70)

    df = build_normal_branch_indicators("XAUUSDm")

    print(f"\nTotal 15M bars: {len(df):,}")
    print(f"Date range: {df['Datetime'].iloc[0]} -> {df['Datetime'].iloc[-1]}")

    print("\nSignalEligible breakdown:")
    print(df["SignalEligible"].value_counts())

    print("\nSample of most recent 10 bars:")
    cols = [
        "Datetime", "Close", "EMA_Fast", "EMA_Slow", "ADX", "DI_Plus", "DI_Minus",
        "Bullish_HTF", "Status", "HTF_Status", "SignalEligible",
    ]
    print(df[cols].tail(10).to_string(index=False))

    output_dir = sv.PROJECT_ROOT / "data" / "indicators_production"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "XAUUSDm_normal_15m_indicators.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()