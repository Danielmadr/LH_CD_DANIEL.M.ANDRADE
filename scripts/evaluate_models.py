import pandas as pd
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

# Adicionar pasta raiz ao path para importar config
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from config import MODEL_PATHS, RAW_DATA_PATH, TEST_SIZE, RANDOM_STATE
from scripts.preprocessing import basic_clean, split_X_y
from scripts.utils import load_model

def evaluate(csv_path=None, models_paths=None):
    if csv_path is None:
        csv_path = str(RAW_DATA_PATH)
    if models_paths is None:
        models_paths = {name: str(path) for name, path in MODEL_PATHS.items()}
        
    df = pd.read_csv(csv_path)
    df = basic_clean(df)
    X, y = split_X_y(df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    for name, path in models_paths.items():
        model = load_model(path)
        preds = model.predict(X_test.astype("float32"))
        print(f"\n📊 {name} - Avaliação:")
        print(f"   RMSE = {root_mean_squared_error(y_test, preds):.4f}")
        print(f"   MAE  = {mean_absolute_error(y_test, preds):.4f}")
        print(f"   R²   = {r2_score(y_test, preds):.4f}")
        #feature importances
        print("   Feature Importances:")
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(5)

        for i, (_, row) in enumerate(importances.iterrows(), 1):
            print(f"   {i}. {row['feature']}: {row['importance']:.4f}")

if __name__ == "__main__":
    evaluate()
