"""
pipeline_train.py
================
Script para treinamento de modelos com log de experimentos no MLflow (DagsHub).
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.models import build_gemini_style_pipelines
from sklearn.model_selection import train_test_split

def run_training(input_path: str, experiment_name: str = "hashtag-ltv-experiment"):
    """Treina modelos e loga métricas no MLflow."""
    
    # 1. Configuração do MLflow via DagsHub
    repo_owner = "flaviohenriquehb777"
    repo_name = "Hashtag-Lifetime-Value"
    
    print(f"Inicializando DagsHub MLflow para {repo_owner}/{repo_name}...")
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    
    mlflow.set_experiment(experiment_name)
    
    print(f"Lendo dados de: {input_path}")
    df = pd.read_csv(input_path)
    
    # Recriar as colunas temporais EXATAMENTE como o preprocessor espera
    if "data_compra" in df.columns:
        df["data_compra"] = pd.to_datetime(df["data_compra"])
        # As colunas esperadas pelo build_gemini_style_pipelines no models.py são:
        # "dia_semana_compra" e "mes_compra"
        df["dia_semana_compra"] = df["data_compra"].dt.dayofweek
        df["mes_compra"] = df["data_compra"].dt.month
        print("Atributos temporais (mes_compra, dia_semana_compra) recriados.")
    
    target_col = "LTV"
    # O preprocessor no models.py espera essas colunas:
    # Numéricas: "valor_1_compra"
    # Categóricas: "Produto Fonte", "Fonte Campanha", "Sexo", "Formacao", "dia_semana_compra", "mes_compra"
    # Passthrough: "recorrente_1_compra"
    
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    pipelines = build_gemini_style_pipelines()
    
    print(f"Iniciando treinamento de {len(pipelines)} modelos...")
    
    for name, pipe in pipelines.items():
        # Sanitiza o nome para o MLflow
        run_name = name.replace(" ", "_").replace(".", "").replace("(", "").replace(")", "")
        
        with mlflow.start_run(run_name=run_name):
            print(f"Treinando {name}...")
            
            # Treino
            pipe.fit(X_train, y_train)
            
            # Predição
            y_pred = pipe.predict(X_test)
            
            # Métricas
            mae = float(mean_absolute_error(y_test, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2 = float(r2_score(y_test, y_pred))
            
            # Log de parâmetros
            if "Random Forest" in name:
                rf_model = pipe.named_steps["model"]
                mlflow.log_param("n_estimators", rf_model.n_estimators)
            
            # Log de métricas
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R2", r2)
            
            # Log do modelo
            mlflow.sklearn.log_model(pipe, artifact_path="model", registered_model_name=run_name)
            
            print(f"[{name}] R2: {r2:.4f} | RMSE: {rmse:.2f} | MAE: {mae:.2f}")

    print("Treinamento e log de experimentos concluídos!")

if __name__ == "__main__":
    input_file = "data/final/ltv_base_final.csv"
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        
    if not Path(input_file).exists():
        print(f"Erro: Arquivo {input_file} não encontrado.")
        sys.exit(1)
        
    run_training(input_file)
