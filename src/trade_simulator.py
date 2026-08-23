"""
EMA-ADX TRADING SYSTEM - V3.0
TRADE SIMULATOR - NORMAL BRANCH (XAUUSDm)

Purpose
-------
Takes the confirmed signals from signal_engine.py and walks each one
forward through real M1 data to determine what actually happens:
does SL or TP get touched first, at what price, and when. Produces a
trade log, an equity curve, and summary performance metrics.

This is the piece that turns "here is a list of signals" into
"here is how this strategy would actually have performed."

Modeling assumptions (all explicit, all changeable)
----------------------------------------------------
1. ENTRY PRICE: the open of the first M1 bar AFTER the signal's 15M
   confirmation bar closes - not the confirmation bar's own close.
   Filling exactly at the alert price is look-ahead bias; this is
   the first price you could actually have traded at.

2. SAME-BAR SL+TP CONFLICT: if one M1 bar's range touches both SL
   and TP, SL is assumed to have been hit first (conservative - we
   have no intra-minute order-flow to prove otherwise).

3. GAP SLIPPAGE: if price gaps straight through SL (e.g. a weekend
   reopen), the exit price is the actual reopening price, not the
   nominal SL level - that gap is real risk, not backtest fantasy.
   TP is treated as a resting limit order: it always fills at the
   TP price even if price gapped past it favourably.

4. ONE TRADE AT A TIME: a new signal that fires while a trade is
   still open is logged as SKIPPED_POSITION_OPEN, never stacked.

5. SIZING: results are reported in R-multiples (risk-normalised,
   independent of any sizing assumption) AND as a compounding equity
   curve assuming RISK_PER_TRADE_PCT of current balance risked per
   trade on a STARTING_BALANCE account. This sizing is NOT part of
   the original Pine script (which has no position-sizing logic) -
   it exists purely to make the equity curve readable, and both
   numbers are easy to change below.
"""

import numpy as np
import pandas as pd

import session_validator as sv
import signal_engine as se


# ============================================================
# CONFIGURATION
# ============================================================

STARTING_BALANCE = 10_000.0
RISK_PER_TRADE_PCT = 1.0     # % of current balance risked per trade
ALLOW_OVERLAPPING_TRADES = False

ENTRY_TIMEFRAME_MINUTES = 15


# ============================================================
# TRADE SIMULATION (WALK-FORWARD OVER M1)
# ============================================================

def simulate_trades(signals: pd.DataFrame, m1: pd.DataFrame) -> pd.DataFrame:
    """
    signals: output of signal_engine.extract_trade_signals() - one
             row per confirmed BUY/SELL with Datetime, SL, TP, etc.
    m1:      output of session_validator.load_m1()
    """

    m1_times = m1["Datetime"].to_numpy()
    m1_open = m1["Open"].to_numpy(dtype=float)
    m1_high = m1["High"].to_numpy(dtype=float)
    m1_low = m1["Low"].to_numpy(dtype=float)
    m1_close = m1["Close"].to_numpy(dtype=float)

    n_m1 = len(m1_times)

    trades = []
    position_open_until = None  # Datetime the current open trade resolves

    for signal in signals.itertuples(index=False):
        signal_close_time = signal.Datetime + pd.Timedelta(minutes=ENTRY_TIMEFRAME_MINUTES)

        if not ALLOW_OVERLAPPING_TRADES and position_open_until is not None:
            if signal_close_time < position_open_until:
                trades.append(
                    _skipped_trade(signal, reason="SKIPPED_POSITION_OPEN")
                )
                continue

        # ---- Find entry: first M1 bar at/after the signal bar's close ----
        entry_idx = np.searchsorted(m1_times, np.datetime64(signal_close_time), side="left")

        if entry_idx >= n_m1:
            trades.append(_skipped_trade(signal, reason="NO_DATA_AFTER_SIGNAL"))
            continue

        entry_time = m1_times[entry_idx]
        entry_price = m1_open[entry_idx]

        sl = signal.SL
        tp = signal.TP
        direction = signal.Direction

        risk_amount = (entry_price - sl) if direction == "BUY" else (sl - entry_price)

        if risk_amount <= 0:
            # Entry already gapped through the stop before we could even
            # get filled - a genuinely bad fill. Record it honestly rather
            # than silently discarding it.
            trades.append(
                _resolved_trade(
                    signal, entry_time, entry_price, sl, tp, risk_amount,
                    exit_time=entry_time, exit_price=entry_price,
                    exit_reason="INVALID_ENTRY_GAP_THROUGH_STOP",
                )
            )
            position_open_until = None
            continue

        # ---- Walk forward: does SL or TP get touched first? ----
        highs = m1_high[entry_idx:]
        lows = m1_low[entry_idx:]
        opens = m1_open[entry_idx:]

        if direction == "BUY":
            hit_sl = lows <= sl
            hit_tp = highs >= tp
        else:
            hit_sl = highs >= sl
            hit_tp = lows <= tp

        either_hit = hit_sl | hit_tp

        if not either_hit.any():
            # Trade never resolves before the data runs out
            last_idx = n_m1 - 1
            trades.append(
                _resolved_trade(
                    signal, entry_time, entry_price, sl, tp, risk_amount,
                    exit_time=m1_times[last_idx],
                    exit_price=m1_close[last_idx],
                    exit_reason="OPEN_AT_DATA_END",
                )
            )
            position_open_until = None
            continue

        rel_idx = np.argmax(either_hit)
        abs_idx = entry_idx + rel_idx

        sl_touched = hit_sl[rel_idx]
        tp_touched = hit_tp[rel_idx]
        bar_open = opens[rel_idx]

        if sl_touched and tp_touched:
            # Same-bar conflict -> conservative: SL hit first
            exit_reason = "STOP_LOSS"
            if direction == "BUY":
                exit_price = bar_open if bar_open <= sl else sl
            else:
                exit_price = bar_open if bar_open >= sl else sl
        elif sl_touched:
            exit_reason = "STOP_LOSS"
            if direction == "BUY":
                exit_price = bar_open if bar_open <= sl else sl
            else:
                exit_price = bar_open if bar_open >= sl else sl
        else:
            exit_reason = "TAKE_PROFIT"
            exit_price = tp  # limit-order assumption: always fills at TP

        if exit_reason == "STOP_LOSS" and exit_price != sl:
            exit_reason = "STOP_LOSS_GAP"

        trades.append(
            _resolved_trade(
                signal, entry_time, entry_price, sl, tp, risk_amount,
                exit_time=m1_times[abs_idx], exit_price=exit_price,
                exit_reason=exit_reason,
            )
        )

        position_open_until = pd.Timestamp(m1_times[abs_idx])

    return pd.DataFrame(trades)


def _resolved_trade(signal, entry_time, entry_price, sl, tp, risk_amount,
                     exit_time, exit_price, exit_reason):
    direction = signal.Direction

    if direction == "BUY":
        r_multiple = (exit_price - entry_price) / risk_amount if risk_amount != 0 else np.nan
    else:
        r_multiple = (entry_price - exit_price) / risk_amount if risk_amount != 0 else np.nan

    return dict(
        SignalTime=signal.Datetime,
        Direction=direction,
        Pattern=signal.Pattern,
        Score=signal.Score,
        EntryTime=pd.Timestamp(entry_time),
        EntryPrice=entry_price,
        SL=sl,
        TP=tp,
        RiskAmount=risk_amount,
        ExitTime=pd.Timestamp(exit_time),
        ExitPrice=exit_price,
        ExitReason=exit_reason,
        R=r_multiple,
        Status="RESOLVED",
    )


def _skipped_trade(signal, reason):
    return dict(
        SignalTime=signal.Datetime,
        Direction=signal.Direction,
        Pattern=signal.Pattern,
        Score=signal.Score,
        EntryTime=pd.NaT,
        EntryPrice=np.nan,
        SL=signal.SL,
        TP=signal.TP,
        RiskAmount=np.nan,
        ExitTime=pd.NaT,
        ExitPrice=np.nan,
        ExitReason=reason,
        R=np.nan,
        Status="SKIPPED",
    )


# ============================================================
# EQUITY CURVE
# ============================================================

def build_equity_curve(trades: pd.DataFrame) -> pd.DataFrame:
    resolved = trades[trades["Status"] == "RESOLVED"].copy()
    resolved = resolved[resolved["ExitReason"] != "OPEN_AT_DATA_END"]
    resolved = resolved.sort_values("ExitTime").reset_index(drop=True)

    balance = STARTING_BALANCE
    equity_rows = []

    for row in resolved.itertuples(index=False):
        risk_dollars = balance * (RISK_PER_TRADE_PCT / 100)
        pnl_dollars = risk_dollars * row.R
        balance += pnl_dollars

        equity_rows.append(
            dict(
                ExitTime=row.ExitTime,
                Direction=row.Direction,
                R=row.R,
                RiskDollars=risk_dollars,
                PnLDollars=pnl_dollars,
                Balance=balance,
            )
        )

    equity = pd.DataFrame(equity_rows)

    if not equity.empty:
        running_max = equity["Balance"].cummax()
        equity["DrawdownPct"] = (equity["Balance"] - running_max) / running_max * 100

    return equity


# ============================================================
# METRICS
# ============================================================

def compute_metrics(trades: pd.DataFrame, equity: pd.DataFrame) -> dict:
    resolved = trades[trades["Status"] == "RESOLVED"]
    closed = resolved[resolved["ExitReason"] != "OPEN_AT_DATA_END"]

    wins = closed[closed["R"] > 0]
    losses = closed[closed["R"] <= 0]

    gross_win_r = wins["R"].sum()
    gross_loss_r = losses["R"].sum()  # negative

    metrics = {
        "total_signals": len(trades),
        "skipped_position_open": (trades["ExitReason"] == "SKIPPED_POSITION_OPEN").sum(),
        "skipped_no_data": (trades["ExitReason"] == "NO_DATA_AFTER_SIGNAL").sum(),
        "invalid_entry_gap": (trades["ExitReason"] == "INVALID_ENTRY_GAP_THROUGH_STOP").sum(),
        "open_at_data_end": (resolved["ExitReason"] == "OPEN_AT_DATA_END").sum(),
        "closed_trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": (len(wins) / len(closed) * 100) if len(closed) else np.nan,
        "avg_r": closed["R"].mean() if len(closed) else np.nan,
        "total_r": closed["R"].sum() if len(closed) else np.nan,
        "expectancy_r": closed["R"].mean() if len(closed) else np.nan,
        "profit_factor": (gross_win_r / abs(gross_loss_r)) if gross_loss_r != 0 else np.nan,
        "best_trade_r": closed["R"].max() if len(closed) else np.nan,
        "worst_trade_r": closed["R"].min() if len(closed) else np.nan,
        "stop_loss_gap_count": (closed["ExitReason"] == "STOP_LOSS_GAP").sum(),
    }

    if not equity.empty:
        metrics["final_balance"] = equity["Balance"].iloc[-1]
        metrics["total_return_pct"] = (equity["Balance"].iloc[-1] / STARTING_BALANCE - 1) * 100
        metrics["max_drawdown_pct"] = equity["DrawdownPct"].min()
    else:
        metrics["final_balance"] = STARTING_BALANCE
        metrics["total_return_pct"] = 0.0
        metrics["max_drawdown_pct"] = 0.0

    metrics["longest_win_streak"] = _longest_streak(closed["R"] > 0)
    metrics["longest_loss_streak"] = _longest_streak(closed["R"] <= 0)

    return metrics


def _longest_streak(bool_series: pd.Series) -> int:
    longest = current = 0
    for value in bool_series:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


# ============================================================
# REPORTING
# ============================================================

def print_report(metrics: dict):
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY - NORMAL BRANCH - XAUUSDm")
    print("=" * 70)

    print(f"\nTotal signals generated:        {metrics['total_signals']:,}")
    print(f"  Skipped (position open):      {metrics['skipped_position_open']:,}")
    print(f"  Skipped (no data after):      {metrics['skipped_no_data']:,}")
    print(f"  Invalid entry (gapped stop):  {metrics['invalid_entry_gap']:,}")
    print(f"  Still open at data end:       {metrics['open_at_data_end']:,}")
    print(f"Closed trades:                  {metrics['closed_trades']:,}")

    print(f"\nWins / Losses:                  {metrics['wins']:,} / {metrics['losses']:,}")
    print(f"Win rate:                       {metrics['win_rate_pct']:.2f}%")
    print(f"Average R per trade:             {metrics['avg_r']:.3f}")
    print(f"Total R:                         {metrics['total_r']:.2f}")
    print(f"Profit factor:                   {metrics['profit_factor']:.2f}")
    print(f"Best trade:                      {metrics['best_trade_r']:.2f}R")
    print(f"Worst trade:                     {metrics['worst_trade_r']:.2f}R")
    print(f"Trades with gap slippage on SL:  {metrics['stop_loss_gap_count']:,}")
    print(f"Longest win streak:              {metrics['longest_win_streak']}")
    print(f"Longest loss streak:             {metrics['longest_loss_streak']}")

    print(f"\nStarting balance:                ${STARTING_BALANCE:,.2f}")
    print(f"Final balance:                    ${metrics['final_balance']:,.2f}")
    print(f"Total return:                     {metrics['total_return_pct']:.2f}%")
    print(f"Max drawdown:                     {metrics['max_drawdown_pct']:.2f}%")
    print(f"(Sizing assumption: {RISK_PER_TRADE_PCT}% of balance risked per trade, compounding)")


def write_report(trades, equity, metrics, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    trades.to_csv(output_dir / "trade_log.csv", index=False)
    equity.to_csv(output_dir / "equity_curve.csv", index=False)

    lines = ["EMA-ADX TRADING SYSTEM - V3.0", "BACKTEST RESULTS: XAUUSDm NORMAL BRANCH", "=" * 70, ""]
    for key, value in metrics.items():
        lines.append(f"{key:<28}{value}")

    with open(output_dir / "performance_summary.txt", "w") as f:
        f.write("\n".join(lines))

    print(f"\nWrote: {output_dir / 'trade_log.csv'}")
    print(f"Wrote: {output_dir / 'equity_curve.csv'}")
    print(f"Wrote: {output_dir / 'performance_summary.txt'}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("       EMA-ADX TRADING SYSTEM - V3.0")
    print("       TRADE SIMULATOR - NORMAL BRANCH")
    print("=" * 70)

    df = se.build_normal_branch_signals("XAUUSDm")
    signals = se.extract_trade_signals(df)
    print(f"\nConfirmed signals: {len(signals):,}")

    m1 = sv.load_m1(sv.DATA_DIR / sv.SYMBOL_FILES["XAUUSDm"])

    print("\nSimulating trades (walk-forward over M1)...")
    trades = simulate_trades(signals, m1)

    equity = build_equity_curve(trades)
    metrics = compute_metrics(trades, equity)

    print_report(metrics)

    output_dir = sv.PROJECT_ROOT / "data" / "backtest_production" / "XAUUSDm"
    write_report(trades, equity, metrics, output_dir)


if __name__ == "__main__":
    main()