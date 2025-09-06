import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from preprocessing import basic_clean, split_X_y
from utils import save_model

def train_xgb(csv_path, model_path="models/xgb_model.pkl"):
    df = pd.read_csv(csv_path)
    df = basic_clean(df)
    X, y = split_X_y(df)

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb = XGBRegressor(
        n_estimators=600,
        learning_rate=0.09,
        max_depth=6,
        subsample=0.94,
        colsample_bytree=1,
        random_state=42,
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
    train_xgb("data/raw/desafio_indicium_imdb.csv")
