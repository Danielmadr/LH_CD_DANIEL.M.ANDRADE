import pandas as pd
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Adicionar pasta raiz ao path para importar config
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from config import MODEL_PATHS, RAW_DATA_PATH, TEST_SIZE, RANDOM_STATE
from scripts.preprocessing import basic_clean, split_X_y
from scripts.utils import save_model

def train_rf(csv_path=None, model_path=None):
    if csv_path is None:
        csv_path = str(RAW_DATA_PATH)
    if model_path is None:
        model_path = str(MODEL_PATHS["RandomForest"])
        
    df = pd.read_csv(csv_path)
    df = basic_clean(df)
    X, y = split_X_y(df)

    X_train, _, y_train, _ = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    rf = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)

    save_model(rf, model_path)
    print(f"✅ RandomForest salvo em {model_path}")

if __name__ == "__main__":
    train_rf()
