import pandas as pd
import sys
from pathlib import Path

# Adicionar pasta raiz ao path para importar config
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from config import MODEL_PATHS, RAW_DATA_PATH
from scripts.preprocessing import basic_clean, split_X_y
from scripts.utils import save_model
from scripts.optuna_tuner import run_optuna_two_phase

def train_xgb_optuna(csv_path=None, model_path=None):
    if csv_path is None:
        csv_path = str(RAW_DATA_PATH)
    if model_path is None:
        model_path = str(MODEL_PATHS["XGBoost (Optuna)"])
        
    df = pd.read_csv(csv_path)
    df = basic_clean(df)
    X, y = split_X_y(df)

    model = run_optuna_two_phase(X, y)
    save_model(model, model_path)
    print(f"✅ XGBoost (Optuna) salvo em {model_path}")

if __name__ == "__main__":
    train_xgb_optuna()
