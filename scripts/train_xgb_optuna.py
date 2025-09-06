import pandas as pd
from preprocessing import basic_clean, split_X_y
from utils import save_model
from optuna_tuner import run_optuna_two_phase


def train_xgb_optuna(csv_path, model_path="models/xgb_optuna_model.pkl"):
    df = pd.read_csv(csv_path)
    df = basic_clean(df)
    X, y = split_X_y(df)

    model = run_optuna_two_phase(X, y)
    save_model(model, model_path)
    print(f"✅ XGBoost (Optuna) salvo em {model_path}")

if __name__ == "__main__":
    train_xgb_optuna("data/raw/desafio_indicium_imdb.csv")
