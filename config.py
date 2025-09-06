"""
Arquivo de configuração centralizada do projeto.
"""

from datetime import datetime
from pathlib import Path

# Constantes temporais
CURRENT_YEAR = datetime.now().year

# Caminhos dos arquivos
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Base de dados
DATABASE_PATH = DATA_DIR / "production.db"
RAW_DATA_PATH = DATA_DIR / "raw" / "desafio_indicium_imdb.csv"

# Modelos
MODEL_PATHS = {
    "RandomForest": MODELS_DIR / "rf_model.pkl",
    "XGBoost": MODELS_DIR / "xgb_model.pkl", 
    "XGBoost (Optuna)": MODELS_DIR / "xgb_optuna_model.pkl"
}

# Configurações de treino
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Configurações de features
COLS_TO_DROP = ["id", "Series_Title", "Unnamed: 0"]

# Thresholds para features derivadas
RECENT_MOVIE_THRESHOLD = 5  # anos
CLASSIC_MOVIE_THRESHOLD = 30  # anos
LONG_MOVIE_THRESHOLD = 120  # minutos
SHORT_MOVIE_THRESHOLD = 90  # minutos

# Configurações de certificação
COMMON_CERTIFICATES = ["G", "PG", "PG-13", "R", "NC-17", "A", "U", "UA", "Unknown"]

# Métricas dos modelos (para exibição na interface)
MODEL_METRICS = {
    "RandomForest": {
        "RMSE": 0.1937,
        "MAE": 0.1534,
        "R²": 0.5536
    },
    "XGBoost": {
        "RMSE": 0.1802,
        "MAE": 0.1453,
        "R²": 0.6136
    },
    "XGBoost (Optuna)": {
        "RMSE": 0.0883,
        "MAE": 0.0697,
        "R²": 0.9073
    }
}

# Top features por modelo (para exibição)
MODEL_TOP_FEATURES = {
    "RandomForest": [
        "Log_Votes: 0.2643",
        "No_of_Votes: 0.2539", 
        "Meta_score: 0.0933",
        "Released_Year: 0.0612",
        "Movie_Age: 0.0610"
    ],
    "XGBoost": [
        "Genre_Drama,Horror,Mystery: 0.1110",
        "No_of_Votes: 0.0838",
        "Genre_Drama,Thriller,War: 0.0727",
        "Genre_Action,Sci-Fi,Thriller: 0.0641",
        "Genre_Action,Sci-Fi: 0.0613"
    ],
    "XGBoost (Optuna)": [
        "High_Votes: 0.2094",
        "No_of_Votes: 0.0708",
        "Log_Votes: 0.0546",
        "Decade: 0.0535",
        "Is_Classic: 0.0487"
    ]
}
