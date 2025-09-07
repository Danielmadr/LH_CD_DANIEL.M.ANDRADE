import argparse
import sqlite3
import sys
from pathlib import Path
import pandas as pd

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from config import DATABASE_PATH, RAW_DATA_PATH
from scripts.preprocess import preprocess_movies_df

TABLE_NAME = "movies"


def create_table(conn: sqlite3.Connection):
    """Cria (se não existir) a tabela principal para armazenar filmes."""
    cur = conn.cursor()
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,  -- identificador único gerado automaticamente
            Series_Title TEXT,                     -- título da obra
            Released_Year REAL,                    -- ano de lançamento 
            Certificate TEXT,                      -- classificação indicativa
            Runtime REAL,                          -- duração (minutos)
            Genre TEXT,                            -- gêneros separados por vírgula
            IMDB_Rating REAL,                      -- nota no IMDb
            Overview TEXT,                         -- descrição/resumo
            Meta_score REAL,                       -- nota de metacritic
            Director TEXT,                         -- nome do diretor
            Star1 TEXT,                            -- elenco principal (4 colunas)
            Star2 TEXT,
            Star3 TEXT,
            Star4 TEXT,
            No_of_Votes REAL,                      -- número de votos do IMDb
            Gross REAL                             -- Faturamento
        );
    """
    )
    conn.commit()


def insert_dataframe(conn: sqlite3.Connection, df: pd.DataFrame):
    expected_cols = [
        "Series_Title",
        "Released_Year",
        "Certificate",
        "Runtime",
        "Genre",
        "IMDB_Rating",
        "Overview",
        "Meta_score",
        "Director",
        "Star1",
        "Star2",
        "Star3",
        "Star4",
        "No_of_Votes",
        "Gross",
    ]
    df = df.copy()
    for c in expected_cols:
        if c not in df.columns:
            df[c] = None
    df = df[expected_cols]
    df.to_sql(TABLE_NAME, conn, if_exists="append", index=False)


def main(csv_path=None, out_dir=None):
    if csv_path is None:
        csv_path = str(RAW_DATA_PATH)
    if out_dir is None:
        out_dir = DATABASE_PATH.parent

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_raw = pd.read_csv(
        csv_path, sep=",", quotechar='"', encoding="utf-8", low_memory=False
    )
    df = preprocess_movies_df(df_raw)

    prod_path = out_dir / "production.db"
    if prod_path.exists():
        prod_path.unlink()
    conn = sqlite3.connect(str(prod_path))
    try:
        create_table(conn)
        insert_dataframe(conn, df)
    finally:
        conn.close()
    print("Bancos criados em:", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", help=f"Caminho para o CSV bruto. Padrão: {RAW_DATA_PATH}"
    )
    parser.add_argument(
        "--out-dir",
        help=f"Diretório de saída para os .db. Padrão: {DATABASE_PATH.parent}",
    )
    args = parser.parse_args()
    main(args.csv, args.out_dir)
