import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from preprocessing import basic_clean, split_X_y
from utils import load_model

def evaluate(csv_path, models_paths):
    df = pd.read_csv(csv_path)
    df = basic_clean(df)
    X, y = split_X_y(df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    models = {
        "RandomForest": "models/rf_model.pkl",
        "XGBoost": "models/xgb_model.pkl",
        "XGBoost (Optuna)": "models/xgb_optuna_model.pkl",
    }
    evaluate("data/raw/desafio_indicium_imdb.csv", models)
