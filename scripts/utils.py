import pickle
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def save_model(model: Any, filename: str) -> None:
    try:

        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        with open(filename, "wb") as f:
            pickle.dump(model, f)

        logger.info(f"Modelo salvo com sucesso em: {filename}")

    except Exception as e:
        logger.error(f"Erro ao salvar modelo em {filename}: {str(e)}")
        raise


def load_model(filename: str) -> Any:
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
