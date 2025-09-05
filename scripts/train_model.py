
import argparse
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import sqlite3
from data_processor import DataProcessor

def load_csv(path: str | Path) -> pd.DataFrame:
    """Carrega o arquivo CSV."""
    return pd.read_csv(path, sep=",", quotechar='"', encoding="utf-8", low_memory=False)

def load_sqlite(path: str | Path, table: str = "movies") -> pd.DataFrame:
    """Carrega dados de um banco SQLite."""
    conn = sqlite3.connect(str(path))
    try:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)
    finally:
        conn.close()

def load_data(source: str | Path) -> pd.DataFrame:
    """
    Carrega dados de diferentes fontes.
    """
    source = str(source).strip()
    if source.lower().endswith(".db"):
        return load_sqlite(source)
    else:
        return load_csv(source)

class ModelTrainer:
    """
    Classe responsável pelo treinamento e avaliação do modelo.
    """
    
    def __init__(self, model_params: dict = None):
        """
        Inicializa o trainer com parâmetros do modelo.
        """
        default_params = {
            'n_estimators': 300,
            'random_state': 42,
            'n_jobs': -1,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2
        }
        
        self.model_params = model_params or default_params
        self.model = None
        self.feature_names = None
        
    def train(self, X_train, y_train):
        """
        Treina o modelo Random Forest.
        """
        print("Iniciando treinamento do modelo...")
        print(f"Shape dos dados de treino: {X_train.shape}")
        print(f"Parâmetros do modelo: {self.model_params}")
        
        # Salvar nomes das features
        self.feature_names = list(X_train.columns)
        
        # Criar e treinar o modelo
        self.model = RandomForestRegressor(**self.model_params)
        self.model.fit(X_train, y_train)
        
        print("Treinamento concluído!")
        return self.model
    
    def evaluate(self, X_test, y_test):
        """
        Avalia o modelo nos dados de teste.
        """
        if self.model is None:
            raise ValueError("Modelo precisa ser treinado antes da avaliação.")
        
        y_pred = self.model.predict(X_test)
        
        # Calcular métricas
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Mostrar resultados
        print("\n" + "="*50)
        print("AVALIAÇÃO DO MODELO")
        print("="*50)
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"MSE:  {mse:.4f}")
        print(f"R²:   {r2:.4f}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            print("\n" + "="*30)
            print("TOP 10 FEATURES MAIS IMPORTANTES")
            print("="*30)
            
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
                print(f"{i+1:2d}. {row['feature']:<20}: {row['importance']:.4f}")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'predictions': y_pred
        }
    
    def save_model(self, model_path: str):
        """
        Salva o modelo treinado.
        """
        if self.model is None:
            raise ValueError("Modelo precisa ser treinado antes de ser salvo.")
        
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'model_params': self.model_params
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Modelo salvo em: {model_path}")
    
    @classmethod
    def load_model(cls, model_path: str):
        """
        Carrega um modelo salvo.
        """
        trainer = cls()
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        trainer.model = model_data['model']
        trainer.feature_names = model_data['feature_names']
        trainer.model_params = model_data['model_params']
        
        return trainer

def train_and_save(source: str, out_dir: str = "models", test_size: float = 0.2):
    """
    Pipeline principal de treinamento.
    """
    print("="*60)
    print("INICIANDO PIPELINE DE TREINAMENTO")
    print("="*60)
    
    # 1. Carregar dados
    print(f"\n1. Carregando dados de: {source}")
    df = load_data(source)
    print(f"   Dados carregados: {df.shape}")
    
    # 2. Preparar dados
    print(f"\n2. Preparando dados com DataProcessor...")
    processor = DataProcessor()
    X_train, X_test, y_train, y_test = processor.prepare_train_test_split(
        df, test_size=test_size, random_state=42
    )
    
    print(f"   Treino: {X_train.shape}, Teste: {X_test.shape}")
    print(f"   Features utilizadas: {len(processor.feature_columns)}")
    
    # 3. Treinar modelo
    print(f"\n3. Treinando modelo...")
    trainer = ModelTrainer()
    trainer.train(X_train, y_train)
    
    # 4. Avaliar modelo
    print(f"\n4. Avaliando modelo...")
    results = trainer.evaluate(X_test, y_test)
    
    # 5. Salvar modelo e processor
    print(f"\n5. Salvando modelo e processor...")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    model_path = Path(out_dir) / "rf_model.pkl"
    processor_path = Path(out_dir) / "data_processor.pkl"
    
    trainer.save_model(model_path)
    processor.save(processor_path)
    
    print(f"   Processor salvo em: {processor_path}")
    
    print("\n" + "="*60)
    print("PIPELINE CONCLUÍDO COM SUCESSO!")
    print("="*60)
    
    return trainer, processor, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina modelo de predição de rating IMDB")
    parser.add_argument("--db", help="Caminho para SQLite.", default=None)
    parser.add_argument("--csv", help="Caminho para CSV.", default=None)
    parser.add_argument("--out-dir", default="models", help="Diretório de saída")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporção para teste")
    
    args = parser.parse_args()
    
    source = args.db or args.csv
    if not source:
        raise SystemExit("Forneça --db OU --csv.")
    
    train_and_save(source, out_dir=args.out_dir, test_size=args.test_size)
