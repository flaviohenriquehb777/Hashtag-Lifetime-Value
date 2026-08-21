from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
from src.data_loader import clean_raw_data, load_raw_data

def run_cleaning(input_path: str, output_path: str):
    """Executa a limpeza de dados e salva o resultado."""
    print(f"Lendo dados de: {input_path}")
    df_raw = load_raw_data(input_path)
    
    print("Iniciando limpeza...")
    df_clean, report = clean_raw_data(df_raw)
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Salvando dados limpos em: {output_path}")
    df_clean.to_csv(output_path, index=False)
    
    print("Limpeza concluída com sucesso!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python src/pipeline_cleaning.py <input_csv> <output_csv>")
        sys.exit(1)
    
    run_cleaning(sys.argv[1], sys.argv[2])
