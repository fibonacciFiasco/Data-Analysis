"""
analysis.py
-----------
All statistical analysis and summary functions for the
Bitcoin Sentiment vs Trader Performance project.

Usage:
    from src.analysis import run_statistical_tests, get_summary_table, print_insights
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple

SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]


# ─────────────────────────────────────────────────────────────────────────────
# Summary Statistics
# ─────────────────────────────────────────────────────────────────────────────

def get_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a summary table of trading performance per sentiment regime.

    Returns a DataFrame with: trade count, win rate, avg PnL,
    median PnL, total PnL, and avg size USD per sentiment.
    """
    summary = (
        df.groupby("sentiment", observed=True)
        .agg(
            Trades      =("Closed PnL",  "count"),
            Win_Rate_Pct=("is_profit",   lambda x: round(x.mean() * 100, 2)),
            Avg_PnL     =("Closed PnL",  lambda x: round(x.mean(), 4)),
            Median_PnL  =("Closed PnL",  lambda x: round(x.median(), 4)),
            Std_PnL     =("Closed PnL",  lambda x: round(x.std(), 4)),
            Total_PnL   =("Closed PnL",  lambda x: round(x.sum(), 2)),
            Avg_Fee     =("Fee",          lambda x: round(x.mean(), 4)),
            Net_PnL_Avg =("net_pnl",     lambda x: round(x.mean(), 4)),
        )
        .reindex(SENTIMENT_ORDER)
    )
    return summary


def get_trader_summary(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Per-trader performance aggregated across all sentiments.
    """
    return (
        df.groupby("Account")
        .agg(
            Trades    =("Closed PnL", "count"),
            Win_Rate  =("is_profit",  lambda x: round(x.mean() * 100, 2)),
            Total_PnL =("Closed PnL", "sum"),
            Avg_PnL   =("Closed PnL", "mean"),
            Best_Trade=("Closed PnL", "max"),
            Worst_Trade=("Closed PnL","min"),
        )
        .sort_values("Total_PnL", ascending=False)
        .head(top_n)
    )


def get_coin_summary(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    Per-coin performance aggregated across all sentiments.
    """
    return (
        df.groupby("Coin")
        .agg(
            Trades   =("Closed PnL", "count"),
            Total_PnL=("Closed PnL", "sum"),
            Avg_PnL  =("Closed PnL", "mean"),
            Win_Rate =("is_profit",  lambda x: round(x.mean() * 100, 2)),
        )
        .sort_values("Total_PnL", ascending=False)
        .head(top_n)
    )


def get_side_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    LONG vs SHORT performance breakdown per sentiment.
    """
    return (
        df.groupby(["sentiment", "side_clean"], observed=True)
        .agg(
            Trades  =("Closed PnL", "count"),
            Avg_PnL =("Closed PnL", "mean"),
            Win_Rate=("is_profit",  lambda x: round(x.mean() * 100, 2)),
        )
        .reset_index()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Statistical Tests
# ─────────────────────────────────────────────────────────────────────────────

def run_kruskal_wallis(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Kruskal-Wallis H-test: tests whether PnL distributions differ
    significantly across sentiment groups.

    Returns
    -------
    (H statistic, p-value)
    """
    groups = [
        df[df["sentiment"] == s]["Closed PnL"].dropna()
        for s in SENTIMENT_ORDER
        if s in df["sentiment"].values
    ]
    h_stat, p_val = stats.kruskal(*groups)
    return h_stat, p_val


def run_pearson_correlation(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Pearson correlation between the numeric sentiment score (1–5)
    and individual trade PnL.

    Returns
    -------
    (r coefficient, p-value)
    """
    sent_map = {s: i+1 for i, s in enumerate(SENTIMENT_ORDER)}
    df2 = df.copy()
    df2["sent_num"] = df2["sentiment"].map(sent_map)
    valid = df2.dropna(subset=["sent_num", "Closed PnL"])
    r, p = stats.pearsonr(valid["sent_num"], valid["Closed PnL"])
    return r, p


def run_pairwise_mannwhitney(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise Mann-Whitney U tests between all pairs of sentiment groups.
    Useful for identifying which specific pairs are significantly different.

    Returns
    -------
    pd.DataFrame with columns: Sentiment_A, Sentiment_B, U_stat, p_value, significant
    """
    results = []
    sentiments = [s for s in SENTIMENT_ORDER if s in df["sentiment"].values]

    for i in range(len(sentiments)):
        for j in range(i + 1, len(sentiments)):
            a = df[df["sentiment"] == sentiments[i]]["Closed PnL"].dropna()
            b = df[df["sentiment"] == sentiments[j]]["Closed PnL"].dropna()
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            results.append({
                "Sentiment_A": sentiments[i],
                "Sentiment_B": sentiments[j],
                "U_stat":      round(u, 2),
                "p_value":     round(p, 6),
                "significant": "✔" if p < 0.05 else "✗",
            })

    return pd.DataFrame(results)


def run_statistical_tests(df: pd.DataFrame) -> Dict:
    """
    Run all statistical tests and return a results dictionary.
    Also prints a formatted summary to console.
    """
    print("\n" + "=" * 60)
    print("  STATISTICAL ANALYSIS")
    print("=" * 60)

    # Kruskal-Wallis
    h, p = run_kruskal_wallis(df)
    print(f"\n1. Kruskal-Wallis Test (PnL across sentiments)")
    print(f"   H = {h:.4f},  p = {p:.2e}")
    print(f"   Result: {'✔ SIGNIFICANT (p < 0.05)' if p < 0.05 else '✗ Not significant'}")

    # Pearson r
    r, p_r = run_pearson_correlation(df)
    print(f"\n2. Pearson Correlation (sentiment score vs PnL)")
    print(f"   r = {r:.4f},  p = {p_r:.2e}")
    direction = "positive" if r > 0 else "negative"
    strength  = "weak" if abs(r) < 0.3 else ("moderate" if abs(r) < 0.6 else "strong")
    print(f"   Result: {strength.capitalize()} {direction} relationship")

    # Pairwise
    print(f"\n3. Pairwise Mann-Whitney U Tests")
    pairs = run_pairwise_mannwhitney(df)
    print(pairs.to_string(index=False))

    return {
        "kruskal_h": h, "kruskal_p": p,
        "pearson_r": r, "pearson_p": p_r,
        "pairwise":  pairs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Print Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    """Print high-level dataset overview."""
    print("\n" + "=" * 60)
    print("  DATASET OVERVIEW")
    print("=" * 60)
    print(f"  Total trades     : {len(df):,}")
    print(f"  Unique traders   : {df['Account'].nunique()}")
    print(f"  Unique coins     : {df['Coin'].nunique()}")
    print(f"  Date range       : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Overall win rate : {df['is_profit'].mean()*100:.1f}%")
    print(f"  Total PnL        : ${df['Closed PnL'].sum():,.2f}")
    print(f"  Avg PnL / trade  : ${df['Closed PnL'].mean():.4f}")
    print()
    print("  Performance by Sentiment:")
    print(get_summary_table(df).to_string())


def print_insights(df: pd.DataFrame):
    """Print the key actionable insights derived from the analysis."""
    wr  = df.groupby("sentiment", observed=True)["is_profit"].mean().reindex(SENTIMENT_ORDER)
    avg = df.groupby("sentiment", observed=True)["Closed PnL"].mean().reindex(SENTIMENT_ORDER)
    tot = df.groupby("sentiment", observed=True)["Closed PnL"].sum().reindex(SENTIMENT_ORDER)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           KEY INSIGHTS — SENTIMENT vs PERFORMANCE           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Win-Rate                                                    ║
║    ✦ Best  : {wr.idxmax():<20} ({wr.max()*100:.1f}%)              ║
║    ✦ Worst : {wr.idxmin():<20} ({wr.min()*100:.1f}%)              ║
║                                                              ║
║  Avg PnL Per Trade                                           ║
║    ✦ Best  : {avg.idxmax():<20} (${avg.max():.2f})               ║
║    ✦ Worst : {avg.idxmin():<20} (${avg.min():.2f})               ║
║                                                              ║
║  Total PnL Generated                                         ║
║    ✦ Most  : {tot.idxmax():<20} (${tot.max():,.0f})              ║
║                                                              ║
║  Recommendations                                             ║
║    1. Scale up positions during Extreme Greed                ║
║    2. Use tighter stops during Extreme Fear                  ║
║    3. Watch for Fear→Greed regime transitions                ║
║    4. Study all-weather traders (Chart 11)                   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")
