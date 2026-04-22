"""
data_loader.py
--------------
Loads, cleans, and merges the Hyperliquid trader dataset
with the Bitcoin Fear & Greed Index.

Usage:
    from src.data_loader import load_and_merge
    df = load_and_merge("data/raw/historical_data.csv",
                        "data/raw/fear_greed_index.csv")
"""

import pandas as pd
import numpy as np
import os


SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]


def load_trader_data(filepath: str) -> pd.DataFrame:
    """
    Load and clean the Hyperliquid historical trade data.

    Parameters
    ----------
    filepath : str
        Path to historical_data.csv

    Returns
    -------
    pd.DataFrame
        Cleaned trader DataFrame with parsed datetime and derived columns.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Trader data not found at: {filepath}")

    df = pd.read_csv(filepath)
    print(f"[data_loader] Trader data loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

    # Parse timestamp
    df["datetime"] = pd.to_datetime(
        df["Timestamp IST"], format="%d-%m-%Y %H:%M", errors="coerce"
    )
    df["date"] = pd.to_datetime(df["datetime"].dt.date)
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.day_name()

    # Clean PnL
    df["Closed PnL"] = pd.to_numeric(df["Closed PnL"], errors="coerce").fillna(0)
    df["Fee"]        = pd.to_numeric(df["Fee"], errors="coerce").fillna(0)
    df["Size USD"]   = pd.to_numeric(df["Size USD"], errors="coerce")
    df["net_pnl"]    = df["Closed PnL"] - df["Fee"]

    # Derived flags
    df["is_profit"]  = df["Closed PnL"] > 0
    df["is_loss"]    = df["Closed PnL"] < 0
    df["side_clean"] = df["Side"].str.upper().str.strip().replace(
        {"BUY": "LONG", "SELL": "SHORT"}
    )

    # Drop rows where datetime failed to parse
    n_bad = df["datetime"].isna().sum()
    if n_bad > 0:
        print(f"[data_loader] Warning: dropped {n_bad} rows with unparseable timestamps")
    df = df.dropna(subset=["datetime"])

    print(f"[data_loader] Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"[data_loader] Unique traders: {df['Account'].nunique()}")
    print(f"[data_loader] Unique coins  : {df['Coin'].nunique()}")
    return df


def load_fear_greed(filepath: str) -> pd.DataFrame:
    """
    Load and clean the Fear & Greed Index dataset.

    Parameters
    ----------
    filepath : str
        Path to fear_greed_index.csv

    Returns
    -------
    pd.DataFrame
        Cleaned sentiment DataFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fear & Greed data not found at: {filepath}")

    fg = pd.read_csv(filepath)
    print(f"[data_loader] Sentiment data loaded: {fg.shape[0]:,} rows")

    fg["date"]       = pd.to_datetime(fg["date"], format="%d-%m-%Y", errors="coerce")
    fg["sentiment"]  = fg["classification"].str.strip().str.title()
    fg["fg_value"]   = pd.to_numeric(fg["value"], errors="coerce")
    fg = fg.dropna(subset=["date"])

    # Validate categories
    unexpected = set(fg["sentiment"].unique()) - set(SENTIMENT_ORDER)
    if unexpected:
        print(f"[data_loader] Warning: unexpected sentiment values: {unexpected}")

    return fg[["date", "sentiment", "fg_value"]]


def load_and_merge(trader_path: str, sentiment_path: str,
                   save_processed: bool = False,
                   processed_path: str = "data/processed/merged_data.csv") -> pd.DataFrame:
    """
    Full pipeline: load both datasets, merge on date, add categorical sentiment.

    Parameters
    ----------
    trader_path     : str  — path to historical_data.csv
    sentiment_path  : str  — path to fear_greed_index.csv
    save_processed  : bool — if True, saves merged CSV to processed_path
    processed_path  : str  — output path for merged CSV

    Returns
    -------
    pd.DataFrame
        Merged and enriched DataFrame ready for analysis.
    """
    trader_df   = load_trader_data(trader_path)
    sentiment_df = load_fear_greed(sentiment_path)

    merged = pd.merge(trader_df, sentiment_df, on="date", how="left")
    merged["sentiment"] = merged["sentiment"].fillna("Neutral")
    merged["sentiment"] = pd.Categorical(
        merged["sentiment"], categories=SENTIMENT_ORDER, ordered=True
    )

    # Overlap check
    overlap = merged["fg_value"].notna().sum()
    print(f"[data_loader] Merged: {len(merged):,} rows "
          f"({overlap:,} with sentiment match, "
          f"{len(merged)-overlap} filled as Neutral)")

    if save_processed:
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        merged.to_csv(processed_path, index=False)
        print(f"[data_loader] Saved processed data → {processed_path}")

    return merged
