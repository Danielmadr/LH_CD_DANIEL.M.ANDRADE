import pandas as pd
import json
import sys
from preprocessing import basic_clean
from utils import load_model

def predict(model_path, input_json, reference_csv="data/raw/desafio_indicium_imdb.csv"):
    # Carregar modelo
    model = load_model(model_path)

    # Carregar JSON como DataFrame
    if isinstance(input_json, str):
        input_data = json.loads(input_json)
    else:
        input_data = input_json

    df_input = pd.DataFrame([input_data])

    # 🔹 Carregar dataset de treino (para alinhar colunas)
    df_ref = pd.read_csv(reference_csv)
    df_ref = basic_clean(df_ref)
    train_cols = df_ref.drop(columns=["IMDB_Rating"]).columns

    # 🔹 Pré-processar entrada
    df_input_clean = basic_clean(df_input)

    # Garantir mesmas colunas que treino
    missing_cols = [col for col in train_cols if col not in df_input_clean.columns]
    if missing_cols:
        df_missing = pd.DataFrame(0, index=df_input_clean.index, columns=missing_cols)
        df_input_clean = pd.concat([df_input_clean, df_missing], axis=1)

    df_input_clean = df_input_clean[train_cols]

    # Fazer predição
    pred = model.predict(df_input_clean.astype("float32"))[0]
    return pred


if __name__ == "__main__":
    # Exemplo: python scripts/predict.py models/xgb_model.pkl
    shawshank = {
        'Series_Title': 'The Shawshank Redemption',
        'Released_Year': 1994,
        'Certificate': 'A',
        'Runtime': '142 min',
        'Genre': 'Drama',
        'Overview': 'Two imprisoned men bond over a number of years.',
        'Meta_score': 80.0,
        'Director': 'Frank Darabont',
        'Star1': 'Tim Robbins',
        'Star2': 'Morgan Freeman',
        'Star3': 'Bob Gunton',
        'Star4': 'William Sadler',
        'No_of_Votes': 2343110,
        'Gross': '28341469'
    }

    if len(sys.argv) < 2:
        print("⚠️ Uso: python scripts/predict.py <caminho_modelo.pkl>")
        sys.exit(1)

    model_path = sys.argv[1]
    pred = predict(model_path, shawshank)
    print(f"🎬 Predição para Shawshank: {pred:.2f}")
