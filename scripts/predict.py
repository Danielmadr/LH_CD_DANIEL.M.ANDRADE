import pandas as pd
import json
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from config import RAW_DATA_PATH
from scripts.preprocessing import basic_clean
from scripts.utils import load_model

def predict(model_path, input_json, reference_csv=None):
    if reference_csv is None:
        reference_csv = str(RAW_DATA_PATH)
    model = load_model(model_path)

    if isinstance(input_json, str):
        input_data = json.loads(input_json)
    else:
        input_data = input_json

    df_input = pd.DataFrame([input_data])

    df_ref = pd.read_csv(reference_csv)
    df_ref = basic_clean(df_ref)
    train_cols = df_ref.drop(columns=["IMDB_Rating"]).columns

    df_input_clean = basic_clean(df_input)

    missing_cols = [col for col in train_cols if col not in df_input_clean.columns]
    if missing_cols:
        df_missing = pd.DataFrame(0, index=df_input_clean.index, columns=missing_cols)
        df_input_clean = pd.concat([df_input_clean, df_missing], axis=1)

    df_input_clean = df_input_clean[train_cols]

    pred = model.predict(df_input_clean.astype("float32"))[0]
    return pred


if __name__ == "__main__":
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
        'Gross': '28,341,469'
    }

    if len(sys.argv) < 2:
        print("⚠️ Uso: python scripts/predict.py <caminho_modelo.pkl>")
        sys.exit(1)

    model_path = sys.argv[1]
    pred = predict(model_path, shawshank)
    print(f"🎬 Predição para Shawshank: {pred:.2f}")
