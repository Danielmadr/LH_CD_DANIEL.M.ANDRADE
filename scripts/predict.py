"""
Script para fazer predições usando o modelo e processor treinados.
Demonstra como usar a nova estrutura separada.
"""

import argparse
import pandas as pd
from pathlib import Path
from data_processor import DataProcessor
from scripts.train_model import ModelTrainer

class MoviePredictor:
    """
    Classe para fazer predições de rating IMDB.
    """
    
    def __init__(self, model_path: str, processor_path: str):
        """
        Inicializa o preditor carregando modelo e processor.
        """
        self.model_trainer = ModelTrainer.load_model(model_path)
        self.processor = DataProcessor.load(processor_path)
        
        print(f"Modelo carregado de: {model_path}")
        print(f"Processor carregado de: {processor_path}")
        print(f"Features do modelo: {len(self.model_trainer.feature_names)}")
    
    def predict_single(self, movie_data: dict) -> float:
        """
        Faz predição para um único filme.
        
        Args:
            movie_data: Dicionário com dados do filme
            
        Returns:
            Rating predito
        """
        # Converter para DataFrame
        df = pd.DataFrame([movie_data])
        
        # Aplicar transformações
        X = self.processor.transform(df)
        
        # Fazer predição
        prediction = self.model_trainer.model.predict(X)[0]
        
        return prediction
    
    def predict_batch(self, df: pd.DataFrame) -> pd.Series:
        """
        Faz predições para múltiplos filmes.
        
        Args:
            df: DataFrame com dados dos filmes
            
        Returns:
            Series com predições
        """
        # Aplicar transformações
        X = self.processor.transform(df)
        
        # Fazer predições
        predictions = self.model_trainer.model.predict(X)
        
        return pd.Series(predictions, index=df.index)
    
    def explain_prediction(self, movie_data: dict, top_features: int = 5):
        """
        Explica uma predição mostrando as features mais importantes.
        """
        # Fazer predição
        prediction = self.predict_single(movie_data)
        
        # Preparar dados transformados
        df = pd.DataFrame([movie_data])
        X = self.processor.transform(df)
        
        # Obter importâncias das features
        if hasattr(self.model_trainer.model, 'feature_importances_'):
            X_row = X[self.model_trainer.feature_names].iloc[0]  # garante alinhamento
            feature_importance = pd.DataFrame({
                'feature': self.model_trainer.feature_names,
                'importance': self.model_trainer.model.feature_importances_,
                'value': X_row.values
            }).sort_values('importance', ascending=False)
            
            print(f"\nPredição: {prediction:.2f}")
            print(f"\nTop {top_features} features mais importantes:")
            print("-" * 50)
            
            for i, (_, row) in enumerate(feature_importance.head(top_features).iterrows()):
                print(f"{i+1}. {row['feature']:<20}: {row['value']:<10.2f} (imp: {row['importance']:.4f})")
        
        return prediction

def demo_prediction():
    """
    Demonstração de como usar o preditor.
    """
    # Caminhos dos arquivos
    model_path = "models/rf_model.pkl"
    processor_path = "models/data_processor.pkl"
    
    # Verificar se arquivos existem
    if not Path(model_path).exists() or not Path(processor_path).exists():
        print("Erro: Modelo ou processor não encontrados.")
        print("Execute primeiro o treinamento com train_model_v2.py")
        return
    
    # Inicializar preditor
    predictor = MoviePredictor(model_path, processor_path)
    
    # Dados de teste - The Shawshank Redemption
    test_movie = {
        'Series_Title': 'The Shawshank Redemption',
        'Released_Year': 1994,
        'Certificate': 'A',
        'Runtime': '142 min',
        'Genre': 'Drama',
        'Overview': 'Two imprisoned men bond over a number of years.',
        'Meta_score': 80.0,
        'Director': 'Frank Darabont',
        'Star1': 'Tim Robbins',
        'Star2': 'Morgan Freeman',
        'Star3': 'Bob Gunton',
        'Star4': 'William Sadler',
        'No_of_Votes': 2343110,
        'Gross': '28341469'
    }
    
    print("="*60)
    print("DEMONSTRAÇÃO DE PREDIÇÃO")
    print("="*60)
    
    print(f"Filme: {test_movie['Series_Title']}")
    print(f"Ano: {test_movie['Released_Year']}")
    print(f"Gênero: {test_movie['Genre']}")
    
    # Fazer predição com explicação
    prediction = predictor.explain_prediction(test_movie, top_features=8)
    
    print(f"\nRating IMDB real: 9.3")
    print(f"Rating predito: {prediction:.2f}")
    print(f"Diferença: {abs(9.3 - prediction):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Faz predições usando modelo treinado")
    parser.add_argument("--demo", action="store_true", help="Executa demonstração")
    parser.add_argument("--model", default="models/rf_model.pkl", help="Caminho do modelo")
    parser.add_argument("--processor", default="models/data_processor.pkl", help="Caminho do processor")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_prediction()
    else:
        print("Use --demo para executar demonstração")
        print("Ou importe MoviePredictor para usar em seu código")
