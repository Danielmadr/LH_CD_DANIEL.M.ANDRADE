"""
Funções utilitárias para o projeto de predição IMDB.
"""

import pickle
import logging
from pathlib import Path
from typing import Any

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_model(model: Any, filename: str) -> None:
    """
    Salva um modelo treinado em disco.
    
    Args:
        model: Modelo a ser salvo
        filename: Caminho do arquivo onde salvar
    """
    try:
        # Criar diretório se não existir
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filename, "wb") as f:
            pickle.dump(model, f)
        
        logger.info(f"Modelo salvo com sucesso em: {filename}")
        
    except Exception as e:
        logger.error(f"Erro ao salvar modelo em {filename}: {str(e)}")
        raise

def load_model(filename: str) -> Any:
    """
    Carrega um modelo salvo do disco.
    
    Args:
        filename: Caminho do arquivo do modelo
        
    Returns:
        Modelo carregado
    """
    try:
        if not Path(filename).exists():
            raise FileNotFoundError(f"Arquivo do modelo não encontrado: {filename}")
            
        with open(filename, "rb") as f:
            model = pickle.load(f)
        
        logger.info(f"Modelo carregado com sucesso de: {filename}")
        return model
        
    except Exception as e:
        logger.error(f"Erro ao carregar modelo de {filename}: {str(e)}")
        raise
