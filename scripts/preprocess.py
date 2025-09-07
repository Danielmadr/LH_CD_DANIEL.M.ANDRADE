import pandas as pd
import numpy as np


def preprocess_movies_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 1. Series_Title: remover espaços extras
    if "Series_Title" in df.columns:
        df["Series_Title"] = df["Series_Title"].astype(str).str.strip()

    # 2. Released_Year: converter para inteiro, tratar ausentes
    if "Released_Year" in df.columns:
        df["Released_Year"] = pd.to_numeric(
            df["Released_Year"], errors="coerce"
        ).astype("Int64")

    # 3. Certificate: padronizar valores, preencher ausentes
    if "Certificate" in df.columns:
        df["Certificate"] = df["Certificate"].fillna("Unknown").astype(str).str.strip()

    # 4. Runtime: extrair minutos, converter para inteiro
    if "Runtime" in df.columns:
        df["Runtime"] = df["Runtime"].astype(str).str.extract(r"(\d+)").astype(float)

    # 5. Genre: remover espaços extras após vírgula
    if "Genre" in df.columns:
        df["Genre"] = df["Genre"].astype(str).str.replace(", ", ",", regex=False)

    # 6. IMDB_Rating: converter para float
    if "IMDB_Rating" in df.columns:
        df["IMDB_Rating"] = pd.to_numeric(df["IMDB_Rating"], errors="coerce")

    # 7. Overview: remover quebras de linha e espaços extras
    if "Overview" in df.columns:
        df["Overview"] = (
            df["Overview"].astype(str).str.replace("\n", " ", regex=False).str.strip()
        )

    # 8. Meta_score: converter para float
    if "Meta_score" in df.columns:
        df["Meta_score"] = pd.to_numeric(df["Meta_score"], errors="coerce")

    # 9. Director: remover espaços extras
    if "Director" in df.columns:
        df["Director"] = df["Director"].astype(str).str.strip()

    # 10-13. Star1-4: remover espaços extras
    for star in ["Star1", "Star2", "Star3", "Star4"]:
        if star in df.columns:
            df[star] = df[star].astype(str).str.strip()

    # 14. No_of_Votes: remover vírgulas, converter para inteiro
    if "No_of_Votes" in df.columns:
        df["No_of_Votes"] = (
            df["No_of_Votes"].astype(str).str.replace(",", "", regex=False)
        )
        df["No_of_Votes"] = pd.to_numeric(df["No_of_Votes"], errors="coerce").astype(
            "Int64"
        )

    # 15. Gross: remover vírgulas, converter para float
    if "Gross" in df.columns:
        df["Gross"] = df["Gross"].astype(str).str.replace(",", "", regex=False)
        df["Gross"] = pd.to_numeric(df["Gross"], errors="coerce")

    # Remover duplicatas
    df = df.drop_duplicates()

    # Padronizar valores ausentes para None
    df = df.replace({np.nan: None})

    return df
