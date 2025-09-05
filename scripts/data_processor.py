"""
Módulo responsável pela preparação e transformação dos dados.
Separação clara entre limpeza, feature engineering e preparação para ML.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
from pathlib import Path

TARGET_COL = "IMDB_Rating"

class DataProcessor:
    """
    Classe responsável por todo o pipeline de preparação dos dados.
    """
    
    def __init__(self):
        self.feature_columns: Optional[List[str]] = None
        self.scalers = {}
        self.encoders = {}
        self.is_fitted = False
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica limpeza básica nos dados.
        """
        from preprocess import preprocess_movies_df
        return preprocess_movies_df(df)
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica feature engineering específico para o modelo.
        """
        df = df.copy()
        
        # 1. Features derivadas de tempo
        if 'Released_Year' in df.columns:
            current_year = 2024
            df['Movie_Age'] = current_year - df['Released_Year']
            df['Is_Recent'] = (df['Movie_Age'] <= 5).astype(int)
            df['Is_Classic'] = (df['Movie_Age'] >= 30).astype(int)
        
        # 2. Features de popularidade
        if 'No_of_Votes' in df.columns:
            df['Log_Votes'] = np.log1p(df['No_of_Votes'].fillna(0))
            df['High_Votes'] = (df['No_of_Votes'] > df['No_of_Votes'].quantile(0.75)).astype(int)
        
        # 3. Features de receita
        if 'Gross' in df.columns:
            filled_gross = pd.to_numeric(df['Gross'], errors="coerce").fillna(0)
            df['Log_Gross'] = np.log1p(filled_gross)
            df['Has_Gross'] = df['Gross'].notna().astype(int)
        
        # 4. Features de rating
        if 'Meta_score' in df.columns:
            df['Has_Meta_Score'] = df['Meta_score'].notna().astype(int)
            filled_meta_score = pd.to_numeric(df['Meta_score'], errors="coerce").fillna(df['Meta_score'].median())
            df['Meta_score_filled'] = filled_meta_score
        
        # 5. Features de duração
        if 'Runtime' in df.columns:
            filled_runtime = pd.to_numeric(df['Runtime'], errors="coerce").fillna(df['Runtime'].median())
            df['Runtime_filled'] = filled_runtime
            df['Is_Long_Movie'] = (df['Runtime_filled'] > 120).astype(int)
            df['Is_Short_Movie'] = (df['Runtime_filled'] < 90).astype(int)
        
        # 6. Features categóricas básicas
        if 'Certificate' in df.columns:
            # Agrupa certificados menos comuns
            cert_counts = df['Certificate'].value_counts()
            common_certs = cert_counts[cert_counts >= 10].index
            df['Certificate_grouped'] = df['Certificate'].where(
                df['Certificate'].isin(common_certs), 'Other'
            )
        
        return df
    
    def select_features(self, df: pd.DataFrame, for_training: bool = True) -> pd.DataFrame:
        """
        Seleciona e prepara features para o modelo.
        
        Args:
            df: DataFrame com features
            for_training: Se True, define as colunas de feature. Se False, usa as já definidas.
        """
        # Features numéricas base
        numeric_base = [
            'Released_Year', 'Runtime_filled', 'Meta_score_filled', 
            'No_of_Votes', 'Gross', 'Movie_Age', 'Log_Votes', 'Log_Gross'
        ]
        
        # Features binárias
        binary_features = [
            'Is_Recent', 'Is_Classic', 'High_Votes', 'Has_Gross', 
            'Has_Meta_Score', 'Is_Long_Movie', 'Is_Short_Movie'
        ]
        
        # Features categóricas (serão encodadas)
        categorical_features = ['Certificate_grouped']
        
        # Filtrar apenas colunas que existem no DataFrame
        available_numeric = [col for col in numeric_base if col in df.columns]
        available_binary = [col for col in binary_features if col in df.columns]
        available_categorical = [col for col in categorical_features if col in df.columns]
        
        if for_training:
            self.feature_columns = available_numeric + available_binary + available_categorical
        
        # Usar as colunas definidas no treinamento
        features_to_use = self.feature_columns or (available_numeric + available_binary + available_categorical)
        
        return df[features_to_use]
    
    def encode_categorical_features(self, df: pd.DataFrame, for_training: bool = True) -> pd.DataFrame:
        """
        Codifica features categóricas.
        """
        df = df.copy()
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if for_training:
                # Criar e ajustar encoder
                encoder = LabelEncoder()
                # Converter tudo para string primeiro para evitar tipos mistos
                df[col] = df[col].astype(str).fillna('Unknown')
                df[col] = encoder.fit_transform(df[col])
                self.encoders[col] = encoder
            else:
                # Usar encoder já treinado
                if col in self.encoders:
                    # Converter para string e preencher ausentes
                    df[col] = df[col].astype(str).fillna('Unknown')
                    # Tratar valores não vistos no treinamento
                    encoder = self.encoders[col]
                    unique_values = set(encoder.classes_)
                    df[col] = df[col].apply(
                        lambda x: x if x in unique_values else 'Unknown'
                    )
                    df[col] = encoder.transform(df[col])
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, for_training: bool = True) -> pd.DataFrame:
        """
        Trata valores ausentes de forma consistente.
        """
        df = df.copy()
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if for_training:
                # Salvar mediana para usar na predição
                median_value = df[col].median()
                self.scalers[f"{col}_median"] = median_value
                df[col] = df[col].fillna(median_value)
            else:
                # Usar mediana do treinamento
                median_value = self.scalers.get(f"{col}_median", df[col].median())
                df[col] = df[col].fillna(median_value)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Aplica todo o pipeline de transformação para dados de treinamento.
        """
        # 1. Limpeza
        df_clean = self.clean_data(df)
        
        # 2. Feature Engineering
        df_features = self.engineer_features(df_clean)
        
        # 3. Seleção de features
        df_selected = self.select_features(df_features, for_training=True)
        
        # 4. Separar target
        if TARGET_COL not in df_clean.columns:
            raise ValueError(f"Target '{TARGET_COL}' não encontrado no DataFrame.")
        
        y = df_clean[TARGET_COL]
        
        # 5. Remover linhas com target ausente
        mask = y.notna()
        df_selected = df_selected[mask]
        y = y[mask]
        
        # 6. Codificar categóricas
        df_encoded = self.encode_categorical_features(df_selected, for_training=True)
        
        # 7. Tratar valores ausentes
        X = self.handle_missing_values(df_encoded, for_training=True)
        
        self.is_fitted = True
        return X, y
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica transformações para dados de predição (usando parâmetros do treinamento).
        """
        if not self.is_fitted:
            raise ValueError("DataProcessor deve ser ajustado antes de transformar novos dados.")
        
        # 1. Limpeza
        df_clean = self.clean_data(df)
        
        # 2. Feature Engineering
        df_features = self.engineer_features(df_clean)
        
        # 3. Seleção de features (usando as definidas no treinamento)
        df_selected = self.select_features(df_features, for_training=False)
        
        # 4. Codificar categóricas
        df_encoded = self.encode_categorical_features(df_selected, for_training=False)
        
        # 5. Tratar valores ausentes
        X = self.handle_missing_values(df_encoded, for_training=False)
        
        return X
    
    def prepare_train_test_split(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        """
        Prepara dados e divide em treino/teste.
        """
        X, y = self.fit_transform(df)
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def save(self, filepath: str):
        """Salva o processor configurado."""
        processor_data = {
            'feature_columns': self.feature_columns,
            'scalers': self.scalers,
            'encoders': self.encoders,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(processor_data, f)
    
    @classmethod
    def load(cls, filepath: str):
        """Carrega um processor salvo."""
        processor = cls()
        
        with open(filepath, 'rb') as f:
            processor_data = pickle.load(f)
        
        processor.feature_columns = processor_data['feature_columns']
        processor.scalers = processor_data['scalers']
        processor.encoders = processor_data['encoders']
        processor.is_fitted = processor_data['is_fitted']
        
        return processor
