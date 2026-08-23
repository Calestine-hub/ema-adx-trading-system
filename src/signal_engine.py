"""
EMA-ADX TRADING SYSTEM - V3.0
SIGNAL ENGINE - NORMAL BRANCH (XAUUSDm, 15M)

Purpose
-------
Turns the indicator layer (indicator_engine.py) into actual timestamped
BUY/SELL signals, reproducing the Pine script's NORMAL branch exactly:

    - Candlestick pattern detectors (pin bar, inside bar, engulfing,
      harami, morning/evening star, hanging man, three soldiers/crows)
    - Base conditions (hard gate: 4H bias + entry EMA + ADX + DI)
    - Pullback arming state machine with recency-based expiration
    - Cooldown tracking
    - Variable-strength score (pattern/DI/ADX/freshness/volume)
    - Confirmed signal + SL/TP calculation

All default parameters below are copied verbatim from the Pine
script's `input.*` defaults - see the module-level PARAMS dict.

State machine note
-------------------
Arming/cooldown/expiration are inherently sequential (the Pine script
carries `var` state bar-to-bar), so this stage runs as a single
forward pass over the bars, exactly mirroring the script's own
bar-by-bar execution model. Everything upstream (indicators, patterns,
score) is vectorized; only this final pass is a loop.

Signal eligibility
-------------------
A NEW signal can only arm/confirm on a bar where SignalEligible is
True (both the 15M bar and the 4H bias bar in effect are Status ==
COMPLETE - decided and implemented in indicator_engine.py). Bars that
are not eligible still update indicators but are treated as if no
new touch/arming/confirmation happens on them, same as if the chart
simply had no reliable bar there.
"""

import numpy as np
import pandas as pd

import indicator_engine as ie
import session_validator as sv


# ============================================================
# PARAMETERS (Pine script defaults, NORMAL branch)
# ============================================================

PARAMS = dict(
    # Trend & momentum
    adxMin=25.0,
    diMinSeparation=3.0,
    # Pullback
    emaZoneATR=0.15,
    maxPullbackBars=8,
    pinBarCLVMin=0.66,
    pinBarMinRangeATR=0.8,
    engulfingMaxOpposingWick=0.5,
    haramiMinFirstBodyATR=0.8,
    haramiMaxSecondBodyRatio=0.5,
    starMinFirstBodyATR=0.8,
    starExtremityMax=0.35,
    starThirdCLVMin=0.66,
    starThirdBodyMin=0.4,
    useFullRangeEngulfing=True,
    useFullRangeHarami=True,
    insideBarBodyMin=0.35,
    insideBarCLVMin=0.60,
    insideBarMaxOpposingWick=1.0,
    # Three soldiers / crows
    soldiersBodyMin=0.45,
    soldiersCLVMin=0.65,
    soldiersMaxOpposingWick=0.6,
    soldiersMaxOpenGap=1.0,
    # Volume
    useVolumeFilter=True,
    volumeAvgLen=20,
    volumeBoostMult=1.3,
    # Score
    minimumScore=6,
    # Entry filters
    cooldownBars=3,
    # Risk (XAUUSD)
    slPips=5.0,
    rrMultiple=3.0,
    xauPipValue=0.10,
)

MIN_TICK = 0.01  # XAUUSD-appropriate floor to avoid divide-by-zero


# ============================================================
# CANDLE MEASUREMENTS
# ============================================================

def add_candle_measurements(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Body"] = (df["Close"] - df["Open"]).abs()
    df["BodySafe"] = df["Body"].clip(lower=MIN_TICK)

    df["UpperWick"] = df["High"] - df[["Open", "Close"]].max(axis=1)
    df["LowerWick"] = df[["Open", "Close"]].min(axis=1) - df["Low"]

    df["CandleRange"] = df["High"] - df["Low"]
    df["CandleRangeSafe"] = df["CandleRange"].clip(lower=MIN_TICK)
    df["CloseLocation"] = (df["Close"] - df["Low"]) / df["CandleRangeSafe"]
    df["BodyRatio"] = df["Body"] / df["CandleRangeSafe"]

    return df


# ============================================================
# PATTERN DETECTORS (vectorized, shift-based)
# ============================================================

def add_patterns(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    df = df.copy()

    close, open_ = df["Close"], df["Open"]
    high, low = df["High"], df["Low"]
    body, body_safe = df["Body"], df["BodySafe"]
    upper, lower = df["UpperWick"], df["LowerWick"]
    close_loc = df["CloseLocation"]
    body_ratio = df["BodyRatio"]
    atr = df["ATR"]

    c1, o1, h1, l1 = close.shift(1), open_.shift(1), high.shift(1), low.shift(1)
    body1, body_safe1 = body.shift(1), body_safe.shift(1)
    close_loc1 = close_loc.shift(1)
    upper1, lower1 = upper.shift(1), lower.shift(1)

    c2, o2, h2, l2 = close.shift(2), open_.shift(2), high.shift(2), low.shift(2)
    body2 = body.shift(2)

    # ---------------- Pin bar ----------------
    pin_range_ok = df["CandleRange"] >= atr * p["pinBarMinRangeATR"]

    bullish_pin = (
        (lower >= body_safe * 2.0)
        & (upper <= body_safe)
        & (close > open_)
        & (close_loc >= p["pinBarCLVMin"])
        & pin_range_ok
    )
    bearish_pin = (
        (upper >= body_safe * 2.0)
        & (lower <= body_safe)
        & (close < open_)
        & (close_loc <= (1 - p["pinBarCLVMin"]))
        & pin_range_ok
    )

    # ---------------- Inside bar ----------------
    inside_bar = (high <= h1) & (low >= l1)

    bullish_inside = (
        inside_bar
        & (close > open_)
        & (body_ratio >= p["insideBarBodyMin"])
        & (close_loc >= p["insideBarCLVMin"])
        & (upper <= body_safe * p["insideBarMaxOpposingWick"])
    )
    bearish_inside = (
        inside_bar
        & (close < open_)
        & (body_ratio >= p["insideBarBodyMin"])
        & (close_loc <= (1 - p["insideBarCLVMin"]))
        & (lower <= body_safe * p["insideBarMaxOpposingWick"])
    )

    # ---------------- Engulfing ----------------
    bullish_engulf_body = (close > open_) & (c1 < o1) & (open_ <= c1) & (close >= o1)
    bearish_engulf_body = (close < open_) & (c1 > o1) & (open_ >= c1) & (close <= o1)

    if p["useFullRangeEngulfing"]:
        bullish_engulf_range = (high >= h1) & (low <= l1)
        bearish_engulf_range = (high <= h1) & (low >= l1)
    else:
        bullish_engulf_range = True
        bearish_engulf_range = True

    bullish_engulfing = (
        bullish_engulf_body & bullish_engulf_range & (upper <= body_safe * p["engulfingMaxOpposingWick"])
    )
    bearish_engulfing = (
        bearish_engulf_body & bearish_engulf_range & (lower <= body_safe * p["engulfingMaxOpposingWick"])
    )

    # ---------------- Harami ----------------
    harami_first_strong = body1 >= atr * p["haramiMinFirstBodyATR"]
    harami_second_shrunk = body <= body1 * p["haramiMaxSecondBodyRatio"]

    bullish_harami_body = (
        (c1 < o1)
        & (close > open_)
        & (df[["Open", "Close"]].max(axis=1) <= o1)
        & (df[["Open", "Close"]].min(axis=1) >= c1)
        & harami_first_strong
        & harami_second_shrunk
    )
    bearish_harami_body = (
        (c1 > o1)
        & (close < open_)
        & (df[["Open", "Close"]].max(axis=1) <= c1)
        & (df[["Open", "Close"]].min(axis=1) >= o1)
        & harami_first_strong
        & harami_second_shrunk
    )

    if p["useFullRangeHarami"]:
        harami_range_ok = (high <= h1) & (low >= l1)
    else:
        harami_range_ok = True

    bullish_harami = bullish_harami_body & harami_range_ok
    bearish_harami = bearish_harami_body & harami_range_ok

    # ---------------- Morning / evening star ----------------
    first_bearish = c2 < o2
    first_bullish = c2 > o2
    middle_small = (close.shift(1) - open_.shift(1)).abs() <= (c2 - o2).abs() * 0.5

    star1_range_safe = (h2 - l2).clip(lower=MIN_TICK)
    star1_strong = body2 >= atr * p["starMinFirstBodyATR"]

    star2_pos_from_low = (df[["Open", "Close"]].min(axis=1).shift(1) - l2) / star1_range_safe
    star2_near_low = star2_pos_from_low <= p["starExtremityMax"]

    star2_pos_from_high = (h2 - df[["Open", "Close"]].max(axis=1).shift(1)) / star1_range_safe
    star2_near_high = star2_pos_from_high <= p["starExtremityMax"]

    third_bullish = close > open_
    third_bearish = close < open_

    morning_star = (
        first_bearish
        & middle_small
        & star1_strong
        & star2_near_low
        & third_bullish
        & (close_loc >= p["starThirdCLVMin"])
        & (body_ratio >= p["starThirdBodyMin"])
        & (close > l2 + (h2 - l2) * (1 - p["starThirdCLVMin"]))
        & (close > (o2 + c2) / 2)
    )

    evening_star = (
        first_bullish
        & middle_small
        & star1_strong
        & star2_near_high
        & third_bearish
        & (close_loc <= (1 - p["starThirdCLVMin"]))
        & (body_ratio >= p["starThirdBodyMin"])
        & (close < l2 + (h2 - l2) * p["starThirdCLVMin"])
        & (close < (o2 + c2) / 2)
    )

    # ---------------- Hanging man ----------------
    hanging_man = (
        (lower >= body_safe * 2.0)
        & (upper <= body_safe)
        & (df[["Open", "Close"]].max(axis=1) >= low + (high - low) * 0.60)
        & pin_range_ok
    )

    # ---------------- Three soldiers / crows ----------------
    body_ratio1, body_ratio2 = body_ratio.shift(1), body_ratio.shift(2)
    close_loc2 = close_loc.shift(2)
    upper2, lower2 = upper.shift(2), lower.shift(2)
    body_safe2 = body_safe.shift(2)

    prior_top1 = df[["Open", "Close"]].max(axis=1).shift(1)
    prior_bot1 = df[["Open", "Close"]].min(axis=1).shift(1)
    prior_top2 = df[["Open", "Close"]].max(axis=1).shift(2)
    prior_bot2 = df[["Open", "Close"]].min(axis=1).shift(2)

    open_within_1 = (open_ <= prior_top1 + p["soldiersMaxOpenGap"] * body_safe1) & (
        open_ >= prior_bot1 - p["soldiersMaxOpenGap"] * body_safe1
    )
    open_within_2 = (o1 <= prior_top2 + p["soldiersMaxOpenGap"] * body_safe2) & (
        o1 >= prior_bot2 - p["soldiersMaxOpenGap"] * body_safe2
    )

    bull_q0 = (body_ratio >= p["soldiersBodyMin"]) & (close_loc >= p["soldiersCLVMin"]) & (upper <= body_safe * p["soldiersMaxOpposingWick"])
    bull_q1 = (body_ratio1 >= p["soldiersBodyMin"]) & (close_loc1 >= p["soldiersCLVMin"]) & (upper1 <= body_safe1 * p["soldiersMaxOpposingWick"])
    bull_q2 = (body_ratio2 >= p["soldiersBodyMin"]) & (close_loc2 >= p["soldiersCLVMin"]) & (upper2 <= body_safe2 * p["soldiersMaxOpposingWick"])

    bear_q0 = (body_ratio >= p["soldiersBodyMin"]) & (close_loc <= (1 - p["soldiersCLVMin"])) & (lower <= body_safe * p["soldiersMaxOpposingWick"])
    bear_q1 = (body_ratio1 >= p["soldiersBodyMin"]) & (close_loc1 <= (1 - p["soldiersCLVMin"])) & (lower1 <= body_safe1 * p["soldiersMaxOpposingWick"])
    bear_q2 = (body_ratio2 >= p["soldiersBodyMin"]) & (close_loc2 <= (1 - p["soldiersCLVMin"])) & (lower2 <= body_safe2 * p["soldiersMaxOpposingWick"])

    three_white_soldiers = (
        (close > open_) & (c1 > o1) & (c2 > o2)
        & (close > c1) & (c1 > c2)
        & bull_q0 & bull_q1 & bull_q2
        & open_within_1 & open_within_2
    )
    three_black_crows = (
        (close < open_) & (c1 < o1) & (c2 < o2)
        & (close < c1) & (c1 < c2)
        & bear_q0 & bear_q1 & bear_q2
        & open_within_1 & open_within_2
    )

    # ---------------- Pattern name resolution (priority order) ----------------
    bull_strong = three_white_soldiers | morning_star | bullish_engulfing | bullish_pin
    bear_strong = three_black_crows | evening_star | bearish_engulfing | bearish_pin

    bull_pattern = np.select(
        [three_white_soldiers, morning_star, bullish_engulfing, bullish_pin, bullish_inside, bullish_harami],
        ["Three White Soldiers", "Morning Star", "Bullish Engulfing", "Bullish Pin Bar", "Bullish Inside Bar", "Bullish Harami"],
        default="",
    )
    bear_pattern = np.select(
        [three_black_crows, evening_star, bearish_engulfing, bearish_pin, bearish_inside, bearish_harami, hanging_man],
        ["Three Black Crows", "Evening Star", "Bearish Engulfing", "Bearish Pin Bar", "Bearish Inside Bar", "Bearish Harami", "Hanging Man"],
        default="",
    )

    df["BullPattern"] = bull_pattern
    df["BearPattern"] = bear_pattern
    df["BullPatternFound"] = df["BullPattern"] != ""
    df["BearPatternFound"] = df["BearPattern"] != ""
    df["BullStrong"] = bull_strong.fillna(False)
    df["BearStrong"] = bear_strong.fillna(False)

    return df


# ============================================================
# BASE CONDITIONS + SCORE COMPONENTS (vectorized)
# ============================================================

def add_base_and_score(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    df = df.copy()

    bullish_di = (df["DI_Plus"] > df["DI_Minus"]) & (df["DI_Net"] >= p["diMinSeparation"])
    bearish_di = (df["DI_Minus"] > df["DI_Plus"]) & (-df["DI_Net"] >= p["diMinSeparation"])
    df["Bullish_DI"] = bullish_di
    df["Bearish_DI"] = bearish_di

    df["BaseBull"] = (
        df["Bullish_HTF"].fillna(False)
        & df["Bullish_EMA"]
        & (df["ADX"] > p["adxMin"])
        & bullish_di
    )
    df["BaseBear"] = (
        df["Bearish_HTF"].fillna(False)
        & df["Bearish_EMA"]
        & (df["ADX"] > p["adxMin"])
        & bearish_di
    )

    pattern_score_long = np.where(df["BullPatternFound"], np.where(df["BullStrong"], 4, 2), 0)
    pattern_score_short = np.where(df["BearPatternFound"], np.where(df["BearStrong"], 4, 2), 0)
    df["PatternScoreLong"] = pattern_score_long
    df["PatternScoreShort"] = pattern_score_short

    di_net = df["DI_Net"]
    df["DiScoreLong"] = np.select(
        [di_net >= 10, di_net >= 5, di_net >= p["diMinSeparation"]], [3, 2, 1], default=0
    )
    df["DiScoreShort"] = np.select(
        [-di_net >= 10, -di_net >= 5, -di_net >= p["diMinSeparation"]], [3, 2, 1], default=0
    )

    adx_excess = df["ADX"] - p["adxMin"]
    df["AdxScore"] = np.select([adx_excess >= 15, adx_excess >= 7], [2, 1], default=0)

    if p["useVolumeFilter"]:
        avg_volume = df["Volume"].rolling(p["volumeAvgLen"]).mean()
        df["VolumeScore"] = (df["Volume"] > avg_volume * p["volumeBoostMult"]).astype(int)
    else:
        df["VolumeScore"] = 0

    return df


# ============================================================
# STATEFUL PASS: ARMING, COOLDOWN, EXPIRATION, SCORE, CONFIRMATION
# ============================================================

def run_state_machine(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    n = len(df)

    base_bull = df["BaseBull"].to_numpy()
    base_bear = df["BaseBear"].to_numpy()
    touching = df["Touching_EMA"].to_numpy()
    eligible = df["SignalEligible"].to_numpy()

    bull_found = df["BullPatternFound"].to_numpy()
    bear_found = df["BearPatternFound"].to_numpy()

    pattern_score_long = df["PatternScoreLong"].to_numpy()
    pattern_score_short = df["PatternScoreShort"].to_numpy()
    di_score_long = df["DiScoreLong"].to_numpy()
    di_score_short = df["DiScoreShort"].to_numpy()
    adx_score = df["AdxScore"].to_numpy()
    volume_score = df["VolumeScore"].to_numpy()

    low = df["Low"].to_numpy()
    high = df["High"].to_numpy()
    close = df["Close"].to_numpy()

    pip = p["xauPipValue"]

    long_armed = False
    short_armed = False
    long_arm_bar = -1
    short_arm_bar = -1
    long_last_touch = -1
    short_last_touch = -1
    bars_since_long_signal = 10 ** 6
    bars_since_short_signal = 10 ** 6

    out_long_armed = np.zeros(n, dtype=bool)
    out_short_armed = np.zeros(n, dtype=bool)
    out_long_score = np.zeros(n, dtype=int)
    out_short_score = np.zeros(n, dtype=int)
    out_confirmed_long = np.zeros(n, dtype=bool)
    out_confirmed_short = np.zeros(n, dtype=bool)

    for i in range(n):
        bars_since_long_signal += 1
        bars_since_short_signal += 1

        long_cooldown_ok = bars_since_long_signal >= p["cooldownBars"]
        short_cooldown_ok = bars_since_short_signal >= p["cooldownBars"]

        # ---- Arming (only on eligible bars - see module docstring) ----
        if eligible[i]:
            if base_bull[i] and touching[i] and long_cooldown_ok:
                if not long_armed:
                    long_armed = True
                    short_armed = False
                    long_arm_bar = i
                long_last_touch = i

            if base_bear[i] and touching[i] and short_cooldown_ok:
                if not short_armed:
                    short_armed = True
                    long_armed = False
                    short_arm_bar = i
                short_last_touch = i

        # ---- Expiration (recency-based, always evaluated) ----
        if long_armed:
            if not base_bull[i]:
                long_armed = False
                long_arm_bar, long_last_touch = -1, -1
            elif long_last_touch >= 0 and (i - long_last_touch) > p["maxPullbackBars"]:
                long_armed = False
                long_arm_bar, long_last_touch = -1, -1

        if short_armed:
            if not base_bear[i]:
                short_armed = False
                short_arm_bar, short_last_touch = -1, -1
            elif short_last_touch >= 0 and (i - short_last_touch) > p["maxPullbackBars"]:
                short_armed = False
                short_arm_bar, short_last_touch = -1, -1

        # ---- Score ----
        freshness_long = 1 if (long_arm_bar >= 0 and (i - long_arm_bar) <= 2) else 0
        freshness_short = 1 if (short_arm_bar >= 0 and (i - short_arm_bar) <= 2) else 0

        long_score = pattern_score_long[i] + di_score_long[i] + adx_score[i] + freshness_long + volume_score[i]
        short_score = pattern_score_short[i] + di_score_short[i] + adx_score[i] + freshness_short + volume_score[i]

        out_long_armed[i] = long_armed
        out_short_armed[i] = short_armed
        out_long_score[i] = long_score
        out_short_score[i] = short_score

        # ---- Confirmation (only on eligible bars) ----
        live_long_ready = long_armed and bull_found[i] and long_score >= p["minimumScore"]
        live_short_ready = short_armed and bear_found[i] and short_score >= p["minimumScore"]

        confirmed_long = eligible[i] and live_long_ready
        confirmed_short = eligible[i] and live_short_ready

        out_confirmed_long[i] = confirmed_long
        out_confirmed_short[i] = confirmed_short

        if confirmed_long:
            long_armed = False
            long_arm_bar, long_last_touch = -1, -1
            bars_since_long_signal = 0

        if confirmed_short:
            short_armed = False
            short_arm_bar, short_last_touch = -1, -1
            bars_since_short_signal = 0

    df = df.copy()
    df["LongArmed"] = out_long_armed
    df["ShortArmed"] = out_short_armed
    df["LongScore"] = out_long_score
    df["ShortScore"] = out_short_score
    df["ConfirmedLong"] = out_confirmed_long
    df["ConfirmedShort"] = out_confirmed_short

    # ---- SL / TP (only meaningful on confirmed bars) ----
    sl_long = low - p["slPips"] * pip
    risk_long = close - sl_long
    tp_long = close + risk_long * p["rrMultiple"]

    sl_short = high + p["slPips"] * pip
    risk_short = sl_short - close
    tp_short = close - risk_short * p["rrMultiple"]

    df["SL_Long"] = sl_long
    df["TP_Long"] = tp_long
    df["SL_Short"] = sl_short
    df["TP_Short"] = tp_short

    return df


# ============================================================
# FULL PIPELINE
# ============================================================

def build_normal_branch_signals(symbol: str = "XAUUSDm") -> pd.DataFrame:
    df = ie.build_normal_branch_indicators(symbol)
    df = add_candle_measurements(df)
    df = add_patterns(df, PARAMS)
    df = add_base_and_score(df, PARAMS)
    df = run_state_machine(df, PARAMS)
    return df


def extract_trade_signals(df: pd.DataFrame) -> pd.DataFrame:
    """One row per confirmed BUY/SELL signal, ready for the trade simulator."""

    longs = df[df["ConfirmedLong"]].copy()
    longs["Direction"] = "BUY"
    longs["Pattern"] = longs["BullPattern"]
    longs["Score"] = longs["LongScore"]
    longs["SL"] = longs["SL_Long"]
    longs["TP"] = longs["TP_Long"]

    shorts = df[df["ConfirmedShort"]].copy()
    shorts["Direction"] = "SELL"
    shorts["Pattern"] = shorts["BearPattern"]
    shorts["Score"] = shorts["ShortScore"]
    shorts["SL"] = shorts["SL_Short"]
    shorts["TP"] = shorts["TP_Short"]

    cols = ["Datetime", "Direction", "Pattern", "Score", "ADX", "Close", "SL", "TP"]
    signals = pd.concat([longs[cols], shorts[cols]], ignore_index=True)
    signals = signals.sort_values("Datetime").reset_index(drop=True)

    return signals


# ============================================================
# MAIN (DIAGNOSTIC RUN)
# ============================================================

def main():
    print("=" * 70)
    print("       EMA-ADX TRADING SYSTEM - V3.0")
    print("       SIGNAL ENGINE - NORMAL BRANCH")
    print("=" * 70)

    df = build_normal_branch_signals("XAUUSDm")
    signals = extract_trade_signals(df)

    print(f"\nTotal 15M bars processed: {len(df):,}")
    print(f"Confirmed signals: {len(signals):,}")
    print(signals["Direction"].value_counts())

    print("\nScore distribution (confirmed signals only):")
    print(signals["Score"].value_counts().sort_index())

    print("\nPattern breakdown (confirmed signals only):")
    print(signals["Pattern"].value_counts())

    print("\nFirst 10 signals:")
    print(signals.head(10).to_string(index=False))

    print("\nLast 10 signals:")
    print(signals.tail(10).to_string(index=False))

    output_dir = sv.PROJECT_ROOT / "data" / "signals_production"
    output_dir.mkdir(parents=True, exist_ok=True)

    signals.to_csv(output_dir / "XAUUSDm_normal_signals.csv", index=False)
    df.to_csv(output_dir / "XAUUSDm_normal_full_state.csv", index=False)

    print(f"\nWrote: {output_dir / 'XAUUSDm_normal_signals.csv'}")
    print(f"Wrote: {output_dir / 'XAUUSDm_normal_full_state.csv'}")


if __name__ == "__main__":
    main()