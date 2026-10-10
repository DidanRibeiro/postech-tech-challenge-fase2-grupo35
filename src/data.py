"""Carregamento e persistência de dados."""
import pandas as pd
from src.config import APPLICATION_RAW, CREDIT_RAW, PROCESSED_FILE

def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    faltantes = [p for p in (APPLICATION_RAW, CREDIT_RAW) if not p.exists()]
    if faltantes:
        nomes = ", ".join(str(p) for p in faltantes)
        raise FileNotFoundError(f"Dados brutos ausentes: {nomes}. Veja data/README.md.")
    return pd.read_csv(APPLICATION_RAW), pd.read_csv(CREDIT_RAW)

def save_processed(df: pd.DataFrame) -> None:
    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_FILE, index=False)

def load_processed() -> pd.DataFrame:
    if not PROCESSED_FILE.exists():
        raise FileNotFoundError("Execute primeiro notebooks/02_preprocessamento.ipynb.")
    return pd.read_csv(PROCESSED_FILE)
