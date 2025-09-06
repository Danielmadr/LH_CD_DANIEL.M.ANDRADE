import pandas as pd
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Adicionar pasta raiz ao path para importar config
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from config import MODEL_PATHS, RAW_DATA_PATH, TEST_SIZE, RANDOM_STATE
from scripts.preprocessing import basic_clean, split_X_y
from scripts.utils import save_model

def train_xgb(csv_path=None, model_path=None):
    if csv_path is None:
        csv_path = str(RAW_DATA_PATH)
    if model_path is None:
        model_path = str(MODEL_PATHS["XGBoost"])
        
    df = pd.read_csv(csv_path)
    df = basic_clean(df)
    X, y = split_X_y(df)

    X_train, _, y_train, _ = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    xgb = XGBRegressor(
        n_estimators=600,
        learning_rate=0.09,
        max_depth=6,
        subsample=0.94,
        colsample_bytree=1,
        random_state=RANDOM_STATE,
        reg_lambda=1.0,
        reg_alpha=0.09,
        verbosity=0, 
        tree_method='hist', 
        n_jobs=-1, 
        objective='reg:squarederror',
    )
    xgb.fit(X_train.astype("float32"), y_train)

    save_model(xgb, model_path)
    print(f"✅ XGBoost salvo em {model_path}")

if __name__ == "__main__":
    train_xgb()
