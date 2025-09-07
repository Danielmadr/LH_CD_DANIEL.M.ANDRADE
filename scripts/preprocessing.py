import pandas as pd
import numpy as np
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from scripts.preprocess import preprocess_movies_df

from config import (
    CURRENT_YEAR,
    COLS_TO_DROP,
    RECENT_MOVIE_THRESHOLD,
    CLASSIC_MOVIE_THRESHOLD,
    LONG_MOVIE_THRESHOLD,
    SHORT_MOVIE_THRESHOLD,
)

def basic_clean(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    for col in COLS_TO_DROP:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    df = df.dropna()
    df = preprocess_movies_df(df)

    if "Released_Year" in df.columns:
        df["Released_Year"] = pd.to_numeric(df["Released_Year"], errors="coerce")
        df["Released_Year"] = df["Released_Year"].fillna(df["Released_Year"].median())
        df["Movie_Age"] = CURRENT_YEAR - df["Released_Year"]
        df["Is_Recent"] = (df["Movie_Age"] <= RECENT_MOVIE_THRESHOLD).astype(int)
        df["Is_Classic"] = (df["Movie_Age"] >= CLASSIC_MOVIE_THRESHOLD).astype(int)

    if "No_of_Votes" in df.columns:
        df["Log_Votes"] = np.log1p(df["No_of_Votes"].fillna(0))
        df["High_Votes"] = (
            df["No_of_Votes"] > df["No_of_Votes"].quantile(0.75)
        ).astype(int)

    if "Gross" in df.columns:
        filled_gross = pd.to_numeric(df["Gross"], errors="coerce").fillna(0)
        df["Log_Gross"] = np.log1p(filled_gross)
        df["Has_Gross"] = df["Gross"].notna().astype(int)
        df["Gross"] = filled_gross

    if "Meta_score" in df.columns:
        df["Has_Meta_Score"] = df["Meta_score"].notna().astype(int)
        filled_meta_score = pd.to_numeric(df["Meta_score"], errors="coerce")
        meta_median = filled_meta_score.median()
        df["Meta_score"] = filled_meta_score.fillna(meta_median)

    if "Runtime" in df.columns:
        filled_runtime = pd.to_numeric(df["Runtime"], errors="coerce")
        runtime_median = filled_runtime.median()
        df["Runtime_filled"] = filled_runtime.fillna(runtime_median)
        df["Is_Long_Movie"] = (df["Runtime_filled"] > LONG_MOVIE_THRESHOLD).astype(int)
        df["Is_Short_Movie"] = (df["Runtime_filled"] < SHORT_MOVIE_THRESHOLD).astype(
            int
        )
        df["Runtime"] = filled_runtime.fillna(runtime_median)

    if "Genre" in df.columns:
        df["Genre"] = df["Genre"].fillna("Unknown")
        df["Main_Genre"] = df["Genre"].astype(str).str.split(",").str[0]
        genre_dummies = pd.get_dummies(df["Genre"].astype(str), prefix="Genre")
        df = pd.concat([df, genre_dummies], axis=1)

    for col in ["Director", "Star1", "Star2", "Star3", "Star4"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
            freq = df[col].value_counts()
            df[f"{col}_Freq"] = df[col].map(freq).fillna(0)

    if "Released_Year" in df.columns:
        df["Decade"] = (df["Released_Year"] // 10) * 10

    for col in [
        "Certificate",
        "Overview",
        "Genre",
        "Director",
        "Star1",
        "Star2",
        "Star3",
        "Star4",
    ]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    for col in df.columns:
        if col != "IMDB_Rating" and df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "IMDB_Rating" in df.columns:
        df = df.dropna(subset=["IMDB_Rating"])

    df = df.fillna(0)
    df = df.infer_objects(copy=False)
    return df


def split_X_y(df: pd.DataFrame):
    X = df.drop(columns=["IMDB_Rating"])
    y = df["IMDB_Rating"]
    return X, y
