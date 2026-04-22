"""
visualisation.py
----------------
All chart generation functions for the Bitcoin Sentiment
vs Trader Performance analysis project.

Usage:
    from src.visualisation import generate_all_charts
    generate_all_charts(df, output_dir="charts/")
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
PALETTE  = ["#d73027", "#fc8d59", "#fee090", "#91bfdb", "#4575b4"]
PAL_DICT = dict(zip(SENTIMENT_ORDER, PALETTE))
BG, CARD = "#0f0f1a", "#1a1a2e"


def _set_style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": CARD,
        "axes.edgecolor":   "#333355", "axes.labelcolor": "white",
        "xtick.color":      "white",   "ytick.color":     "white",
        "text.color":       "white",   "grid.color":      "#2a2a4a",
        "grid.linestyle":   "--",      "grid.alpha":       0.5,
        "font.family":      "DejaVu Sans",
    })


def _save(fig, path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✔ Saved {os.path.basename(path)}")


# ─── Chart 1: Sentiment Distribution ─────────────────────────────────────────
def chart_sentiment_distribution(df: pd.DataFrame, out: str):
    _set_style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Market Sentiment Overview  |  May 2023 – May 2025",
                 fontsize=16, color="white", fontweight="bold", y=1.01)

    counts = df["sentiment"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
    ax = axes[0]; ax.set_facecolor(CARD)
    bars = ax.bar(counts.index, counts.values, color=PALETTE, edgecolor="#222240", width=0.6)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+300,
                f"{val:,}", ha="center", va="bottom", fontsize=10, color="white", fontweight="bold")
    ax.set_title("Number of Trades per Sentiment Regime", fontsize=13, color="white")
    ax.set_ylabel("Trade Count"); ax.grid(axis="y"); ax.set_ylim(0, counts.max()*1.15)
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right", fontsize=10)

    ax2 = axes[1]; ax2.set_facecolor(CARD)
    wedges, texts, autotexts = ax2.pie(
        counts.values, labels=counts.index, colors=PALETTE,
        autopct="%1.1f%%", startangle=140, pctdistance=0.78,
        wedgeprops=dict(edgecolor=BG, linewidth=2))
    for t in texts:     t.set_color("white"); t.set_fontsize(10)
    for t in autotexts: t.set_color("white"); t.set_fontsize(9); t.set_fontweight("bold")
    ax2.set_title("Sentiment Share of All Trades (%)", fontsize=13, color="white")
    plt.tight_layout(); _save(fig, out)


# ─── Chart 2: PnL & Win Rate by Sentiment ────────────────────────────────────
def chart_pnl_winrate(df: pd.DataFrame, out: str):
    _set_style()
    agg = df.groupby("sentiment", observed=True).agg(
        avg_pnl   =("Closed PnL", "mean"),
        median_pnl=("Closed PnL", "median"),
        win_rate  =("is_profit",  "mean"),
    ).reindex(SENTIMENT_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Trader Performance vs Market Sentiment",
                 fontsize=16, color="white", fontweight="bold", y=1.01)

    x = np.arange(len(agg)); w = 0.38
    pal = [PALETTE[SENTIMENT_ORDER.index(s)] for s in agg.index]

    ax = axes[0]; ax.set_facecolor(CARD)
    ax.bar(x-w/2, agg["avg_pnl"],    w, label="Mean PnL",   color=pal, alpha=0.9,  edgecolor="#222240")
    ax.bar(x+w/2, agg["median_pnl"], w, label="Median PnL", color=pal, alpha=0.45, edgecolor="#222240")
    ax.axhline(0, color="white", lw=1, ls="--", alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(SENTIMENT_ORDER, rotation=18, ha="right")
    ax.set_title("Mean vs Median Closed PnL", fontsize=12, color="white")
    ax.set_ylabel("Closed PnL (USD)"); ax.grid(axis="y")
    ax.legend(facecolor=CARD, labelcolor="white", fontsize=9)

    ax2 = axes[1]; ax2.set_facecolor(CARD)
    bars = ax2.bar(agg.index, agg["win_rate"]*100, color=pal, edgecolor="#222240", alpha=0.9)
    for bar, val in zip(bars, agg["win_rate"]*100):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=10, color="white", fontweight="bold")
    ax2.axhline(50, color="white", lw=1, ls="--", alpha=0.6, label="50% benchmark")
    ax2.set_title("Win Rate by Sentiment", fontsize=12, color="white")
    ax2.set_ylabel("Win Rate (%)"); ax2.set_ylim(0, 70)
    ax2.legend(facecolor=CARD, labelcolor="white"); ax2.grid(axis="y")
    plt.setp(ax2.get_xticklabels(), rotation=18, ha="right")
    plt.tight_layout(); _save(fig, out)


# ─── Chart 3: Violin Plot ─────────────────────────────────────────────────────
def chart_violin(df: pd.DataFrame, out: str):
    _set_style()
    q01 = df["Closed PnL"].quantile(0.01); q99 = df["Closed PnL"].quantile(0.99)
    clipped = df[df["Closed PnL"].between(q01, q99)].copy()

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(BG); ax.set_facecolor(CARD)
    sns.violinplot(data=clipped, x="sentiment", y="Closed PnL",
                   palette=PAL_DICT, order=SENTIMENT_ORDER,
                   inner="box", cut=0, ax=ax, linewidth=0.9)
    ax.axhline(0, color="white", lw=1.2, ls="--", alpha=0.7)
    for i, s in enumerate(SENTIMENT_ORDER):
        med = clipped[clipped["sentiment"] == s]["Closed PnL"].median()
        ax.text(i, med+1, f"${med:.1f}", ha="center", va="bottom", fontsize=8.5,
                color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#333355", alpha=0.7))
    ax.set_title("PnL Distribution by Sentiment  (clipped 1st–99th pct)",
                 fontsize=14, color="white", fontweight="bold")
    ax.set_xlabel("Sentiment"); ax.set_ylabel("Closed PnL (USD)"); ax.grid(axis="y")
    plt.tight_layout(); _save(fig, out)


# ─── Chart 4: Cumulative PnL ──────────────────────────────────────────────────
def chart_cumulative_pnl(df: pd.DataFrame, out: str):
    _set_style()
    daily = df.groupby(["date", "sentiment"], observed=True)["Closed PnL"].sum().reset_index().sort_values("date")

    fig, ax = plt.subplots(figsize=(15, 7))
    fig.patch.set_facecolor(BG); ax.set_facecolor(CARD)
    for sent, color in zip(SENTIMENT_ORDER, PALETTE):
        sub = daily[daily["sentiment"] == sent].copy()
        if sub.empty: continue
        sub["cum"] = sub["Closed PnL"].cumsum()
        ax.plot(sub["date"], sub["cum"], label=sent, color=color, lw=2.2, alpha=0.9)
        ax.annotate(f"${sub['cum'].iloc[-1]:,.0f}",
                    xy=(sub["date"].iloc[-1], sub["cum"].iloc[-1]),
                    xytext=(5, 0), textcoords="offset points",
                    fontsize=8, color=color, va="center")
    ax.axhline(0, color="white", lw=0.8, ls="--", alpha=0.5)
    ax.set_title("Cumulative Closed PnL Over Time — by Sentiment",
                 fontsize=14, color="white", fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Cumulative PnL (USD)")
    ax.legend(facecolor=CARD, labelcolor="white", fontsize=10, loc="upper left")
    ax.grid(); fig.autofmt_xdate()
    plt.tight_layout(); _save(fig, out)


# ─── Chart 5: Heatmap Sentiment × Hour ───────────────────────────────────────
def chart_heatmap_hour(df: pd.DataFrame, out: str):
    _set_style()
    pivot = (df.groupby(["sentiment", "hour"], observed=True)["Closed PnL"]
               .mean().unstack(fill_value=0).reindex(SENTIMENT_ORDER))

    fig, ax = plt.subplots(figsize=(18, 5))
    fig.patch.set_facecolor(BG)
    sns.heatmap(pivot, cmap="RdYlGn", center=0, linewidths=0.3, linecolor=BG,
                ax=ax, cbar_kws={"label": "Avg PnL (USD)"},
                annot=True, fmt=".0f", annot_kws={"size": 7})
    ax.set_title("Average PnL by Sentiment × Hour of Day (IST)",
                 fontsize=13, color="white", fontweight="bold")
    ax.set_xlabel("Hour (IST)"); ax.set_ylabel("Sentiment")
    plt.tight_layout(); _save(fig, out)


# ─── Chart 6: Long vs Short ───────────────────────────────────────────────────
def chart_long_short(df: pd.DataFrame, out: str):
    _set_style()
    side_pnl = (df.groupby(["sentiment", "side_clean"], observed=True)["Closed PnL"]
                  .mean().unstack(fill_value=0).reindex(SENTIMENT_ORDER))
    side_wr  = (df.groupby(["sentiment", "side_clean"], observed=True)["is_profit"]
                  .mean().unstack(fill_value=0).reindex(SENTIMENT_ORDER) * 100)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor(BG)
    fig.suptitle("LONG vs SHORT Performance by Sentiment",
                 fontsize=15, color="white", fontweight="bold", y=1.01)

    x = np.arange(len(side_pnl)); w = 0.38
    for ax, data, title, ylabel in zip(
            axes,
            [side_pnl, side_wr],
            ["Avg PnL: LONG vs SHORT", "Win Rate: LONG vs SHORT (%)"],
            ["Avg Closed PnL (USD)", "Win Rate (%)"]):
        ax.set_facecolor(CARD)
        if "LONG"  in data.columns: ax.bar(x-w/2, data["LONG"],  w, label="LONG",  color="#2ecc71", alpha=0.85, edgecolor="#222240")
        if "SHORT" in data.columns: ax.bar(x+w/2, data["SHORT"], w, label="SHORT", color="#e74c3c", alpha=0.85, edgecolor="#222240")
        ax.axhline(0, color="white", lw=0.8, ls="--", alpha=0.5)
        ax.set_xticks(x); ax.set_xticklabels(SENTIMENT_ORDER, rotation=18, ha="right")
        ax.set_title(title, fontsize=12, color="white"); ax.set_ylabel(ylabel)
        ax.legend(facecolor=CARD, labelcolor="white"); ax.grid(axis="y")
    plt.tight_layout(); _save(fig, out)


# ─── Chart 7: Top Traders ─────────────────────────────────────────────────────
def chart_top_traders(df: pd.DataFrame, out: str):
    _set_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Top 10 Traders by Total PnL", fontsize=15, color="white", fontweight="bold", y=1.01)

    configs = [
        ("All Sentiments", df,                                  "#8e44ad"),
        ("Extreme Fear",   df[df["sentiment"]=="Extreme Fear"], "#d73027"),
        ("Extreme Greed",  df[df["sentiment"]=="Extreme Greed"],"#4575b4"),
    ]
    for ax, (title, sub, color) in zip(axes, configs):
        ax.set_facecolor(CARD)
        if sub.empty: ax.set_visible(False); continue
        top = sub.groupby("Account")["Closed PnL"].sum().nlargest(10).sort_values()
        labels = [a[:10]+"…" if len(a) > 12 else a for a in top.index.astype(str)]
        ax.barh(labels, top.values, color=color, edgecolor="#222240", alpha=0.85)
        for i, (label, val) in enumerate(zip(labels, top.values)):
            ax.text(val + abs(top.values).max()*0.01, i,
                    f"${val:,.0f}", va="center", fontsize=8, color="white")
        ax.axvline(0, color="white", lw=0.8, ls="--", alpha=0.5)
        ax.set_title(title, fontsize=12, color="white")
        ax.set_xlabel("Total PnL (USD)"); ax.grid(axis="x")
    plt.tight_layout(); _save(fig, out)


# ─── Chart 8: Top Coins ───────────────────────────────────────────────────────
def chart_top_coins(df: pd.DataFrame, out: str):
    _set_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Top 10 Coins by Total PnL in Extreme Regimes",
                 fontsize=14, color="white", fontweight="bold", y=1.01)

    for ax, (sent, color) in zip(axes, [("Extreme Fear","#d73027"),("Extreme Greed","#4575b4")]):
        ax.set_facecolor(CARD)
        sub = df[df["sentiment"] == sent]
        top = sub.groupby("Coin")["Closed PnL"].sum().nlargest(10).sort_values()
        bar_colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in top.values]
        ax.barh(top.index.astype(str), top.values, color=bar_colors, edgecolor="#222240", alpha=0.85)
        for i, val in enumerate(top.values):
            ax.text(val + abs(top.values).max()*0.01, i,
                    f"${val:,.0f}", va="center", fontsize=8.5, color="white")
        ax.axvline(0, color="white", lw=0.8, ls="--", alpha=0.5)
        ax.set_title(f"Top Coins During {sent}", fontsize=12, color="white")
        ax.set_xlabel("Total PnL (USD)"); ax.grid(axis="x")
    plt.tight_layout(); _save(fig, out)


# ─── Chart 9: Monthly Heatmap ─────────────────────────────────────────────────
def chart_monthly_heatmap(df: pd.DataFrame, out: str):
    _set_style()
    df2 = df.copy()
    df2["year"]  = df2["date"].dt.year
    df2["month"] = df2["date"].dt.month
    pivot = df2.groupby(["year","month"])["Closed PnL"].sum().unstack(fill_value=0)
    pivot.columns = [pd.Timestamp(2000, m, 1).strftime("%b") for m in pivot.columns]

    fig, ax = plt.subplots(figsize=(15, 4))
    fig.patch.set_facecolor(BG)
    sns.heatmap(pivot, cmap="RdYlGn", center=0, linewidths=0.5, linecolor=BG,
                annot=True, fmt=".0f", annot_kws={"size": 9},
                cbar_kws={"label": "Total PnL (USD)"}, ax=ax)
    ax.set_title("Monthly Total Closed PnL Heatmap", fontsize=13, color="white", fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("Year")
    plt.tight_layout(); _save(fig, out)


# ─── Chart 10: FG Value vs Daily PnL Scatter ─────────────────────────────────
def chart_fg_scatter(df: pd.DataFrame, out: str):
    _set_style()
    daily = df.groupby(["date","fg_value","sentiment"], observed=True)["Closed PnL"].sum().reset_index()
    daily = daily.dropna(subset=["fg_value"])

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(BG); ax.set_facecolor(CARD)
    for sent, color in zip(SENTIMENT_ORDER, PALETTE):
        sub = daily[daily["sentiment"] == sent]
        ax.scatter(sub["fg_value"], sub["Closed PnL"], alpha=0.5, s=30,
                   color=color, label=sent, edgecolors="none")
    x_v = daily["fg_value"].values; y_v = daily["Closed PnL"].values
    mask = np.isfinite(x_v) & np.isfinite(y_v)
    if mask.sum() > 2:
        m, b, r, p, _ = stats.linregress(x_v[mask], y_v[mask])
        xs = np.linspace(x_v[mask].min(), x_v[mask].max(), 100)
        ax.plot(xs, m*xs+b, color="white", lw=2, ls="--", alpha=0.8,
                label=f"Trend (r={r:.2f}, p={p:.3f})")
    ax.axhline(0, color="white", lw=0.7, ls=":", alpha=0.5)
    ax.set_title("Fear & Greed Index Value vs Daily Total PnL",
                 fontsize=13, color="white", fontweight="bold")
    ax.set_xlabel("Fear & Greed Index (0 = Extreme Fear, 100 = Extreme Greed)")
    ax.set_ylabel("Daily Total Closed PnL (USD)")
    ax.legend(facecolor=CARD, labelcolor="white", fontsize=9); ax.grid()
    plt.tight_layout(); _save(fig, out)


# ─── Chart 11: Trader × Sentiment Heatmap ────────────────────────────────────
def chart_trader_heatmap(df: pd.DataFrame, out: str):
    _set_style()
    pivot = (df.groupby(["Account","sentiment"], observed=True)["Closed PnL"]
               .sum().unstack(fill_value=0)[SENTIMENT_ORDER])
    top20 = df.groupby("Account")["Closed PnL"].sum().nlargest(20).index
    pivot = pivot.loc[pivot.index.isin(top20)]
    pivot.index = [i[:12]+"…" if len(i) > 14 else i for i in pivot.index.astype(str)]

    fig, ax = plt.subplots(figsize=(13, 9))
    fig.patch.set_facecolor(BG)
    sns.heatmap(pivot, cmap="RdYlGn", center=0, linewidths=0.4, linecolor=BG,
                annot=True, fmt=".0f", annot_kws={"size": 8},
                cbar_kws={"label": "Total PnL (USD)"}, ax=ax)
    ax.set_title("Top 20 Traders — PnL per Sentiment Regime",
                 fontsize=13, color="white", fontweight="bold")
    ax.set_xlabel("Sentiment"); ax.set_ylabel("Trader (Account)")
    plt.tight_layout(); _save(fig, out)


# ─── Master function ──────────────────────────────────────────────────────────
def generate_all_charts(df: pd.DataFrame, output_dir: str = "charts/"):
    """Generate all 11 charts and save them to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[visualisation] Generating charts → {output_dir}")

    chart_sentiment_distribution(df, os.path.join(output_dir, "chart1_sentiment_distribution.png"))
    chart_pnl_winrate           (df, os.path.join(output_dir, "chart2_pnl_winrate.png"))
    chart_violin                (df, os.path.join(output_dir, "chart3_violin.png"))
    chart_cumulative_pnl        (df, os.path.join(output_dir, "chart4_cumulative_pnl.png"))
    chart_heatmap_hour          (df, os.path.join(output_dir, "chart5_heatmap_hour.png"))
    chart_long_short            (df, os.path.join(output_dir, "chart6_long_short.png"))
    chart_top_traders           (df, os.path.join(output_dir, "chart7_top_traders.png"))
    chart_top_coins             (df, os.path.join(output_dir, "chart8_top_coins.png"))
    chart_monthly_heatmap       (df, os.path.join(output_dir, "chart9_monthly_heatmap.png"))
    chart_fg_scatter            (df, os.path.join(output_dir, "chart10_fg_scatter.png"))
    chart_trader_heatmap        (df, os.path.join(output_dir, "chart11_trader_sentiment_heatmap.png"))

    print(f"[visualisation] All 11 charts saved.\n")
