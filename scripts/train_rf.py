import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from preprocessing import basic_clean, split_X_y
from utils import save_model

def train_rf(csv_path, model_path="models/rf_model.pkl"):
    df = pd.read_csv(csv_path)
    df = basic_clean(df)
    X, y = split_X_y(df)

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    save_model(rf, model_path)
    print(f"✅ RandomForest salvo em {model_path}")

if __name__ == "__main__":
    train_rf("data/raw/desafio_indicium_imdb.csv")
