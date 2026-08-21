# 🎯 Hashtag LTV Prediction

<p align="center">
  <img src="reports/figures/Logo_linkedin_vazada.png" alt="Hashtag Logo" width="200">
</p>

<p align="center">
  <a href="LICENSE.md"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/scikit--learn-1.3+-orange.svg" alt="Scikit-Learn"></a>
  <a href="https://mlflow.org/"><img src="https://img.shields.io/badge/MLOps-MLflow%20%7C%20DVC-red.svg" alt="MLOps"></a>
  <a href="https://github.com/flaviohenriquehb777/Hashtag-Lifetime-Value/actions"><img src="https://github.com/flaviohenriquehb777/Hashtag-Lifetime-Value/actions/workflows/ci.yml/badge.svg" alt="Build Status"></a>
</p>

> Previsão de **Lifetime Value (LTV)** dos clientes da Hashtag Treinamentos, estruturado com padrão **CRISP-DM**, boas práticas de Engenharia de Software e MLOps para Ciência de Dados.

## 📌 Sumário

- [Visão Geral do Modelo](#-visão-geral-do-modelo)
- [Objetivos da Análise](#-objetivos-da-análise)
- [Estrutura do Modelo](#-estrutura-do-modelo)
- [Base de Dados](#-base-de-dados)
- [Metodologia de Análise](#-metodologia-de-análise)
- [Resultados Chave e Apresentação](#-resultados-chave-e-apresentação)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)
- [Instalação e Uso](#-instalação-e-uso)
- [Licença](#-licença)
- [Contato](#-contato)

---

## 📋 Visão Geral do Modelo

Este projeto implementa uma solução de ponta a ponta para prever o **Lifetime Value (LTV)** de clientes. O modelo utiliza dados históricos de compras e perfil demográfico para estimar o valor total que um cliente trará para a empresa ao longo do tempo, permitindo decisões estratégicas de aquisição e retenção.

## 🎯 Objetivos da Análise

1.  **Prever o LTV**: Estimar com precisão o valor futuro dos clientes.
2.  **Identificar Alavancas de Valor**: Entender quais variáveis (produto de entrada, canal, perfil) mais impactam o LTV.
3.  **Otimizar o CAC**: Fornecer uma ferramenta que defina o Custo de Aquisição de Cliente (CAC) máximo recomendado com base no LTV previsto.
4.  **Automação**: Criar um pipeline robusto e testável para deploy contínuo.

## 🏗️ Estrutura do Modelo

O projeto segue a metodologia **CRISP-DM**, com o código organizado de forma modular:

```
├── .github/workflows/       # CI/CD (GitHub Actions)
├── config/                  # Configurações (.yaml)
├── data/                    # Dados (Raw, Processed, Final)
├── docs/                    # Documentação técnica
├── models/                  # Artefatos e calculadoras exportadas
├── notebooks/               # Experimentos numerados (01 a 06)
├── reports/figures/         # Gráficos e assets (incluindo logo)
├── src/                     # Código-fonte (cleaning, models, deploy, etc.)
├── tests/                   # Testes unitários (pytest)
└── requirements.txt         # Dependências
```

## 🗄️ Base de Dados

A base de dados contém informações históricas de clientes, incluindo:
- **Dados Transacionais**: Valor da primeira compra, recorrência, data da compra.
- **Perfil Demográfico**: Sexo, formação, renda.
- **Jornada do Cliente**: Produto fonte e canal de aquisição (campanha).

*Nota: Os dados passam por um processo rigoroso de limpeza e normalização antes da modelagem.*

## 🧪 Metodologia de Análise

1.  **Exploração e Limpeza**: Tratamento de outliers, normalização de strings e engenharia de atributos temporais.
2.  **Benchmarking**: Comparação de múltiplos algoritmos (Dummy, Linear Regression, Polynomial, Random Forest).
3.  **Validação Cruzada**: Uso de K-Fold (5 folds) para garantir a estabilidade das métricas.
4.  **Seleção de Modelo**: Foco em interpretabilidade e performance, resultando na escolha da Regressão Linear para o deploy da calculadora.
5.  **Simulação de Negócio**: Tradução do erro estatístico em impacto financeiro.

## 📈 Resultados Chave e Apresentação

- **Performance**: O modelo de Regressão Linear apresentou um $R^2$ sólido, permitindo uma estimativa confiável para o LTV.
- **Artefato de Deploy**: Uma calculadora interativa em Excel foi gerada automaticamente, permitindo que o time de marketing simule o LTV e o CAC máximo em tempo real.
- **Insights**: Identificação de que o "Produto Fonte" e a "Recorrência na 1ª Compra" são os principais preditores de valor a longo prazo.

## 🛠️ Tecnologias Utilizadas

- **Linguagem**: Python 3.10+
- **Data Science**: Pandas, NumPy, Scikit-Learn, SciPy
- **Visualização**: Matplotlib, Seaborn, Plotly
- **MLOps**: DVC, MLflow
- **Qualidade**: Pytest, Black, Flake8, GitHub Actions
- **Deploy**: Openpyxl, PyWin32 (automação Excel)

## 🚀 Instalação e Uso

### 1. Clonar o Repositório
```bash
git clone https://github.com/flaviohenriquehb/Hashtag-Lifetime-Value.git
cd Hashtag-Lifetime-Value
```

### 2. Configurar Ambiente
```bash
python -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\activate no Windows
pip install -r requirements.txt
```

### 3. Executar Testes
```bash
pytest tests/
```

## 📄 Licença

Este projeto está sob a licença MIT - veja o arquivo [LICENSE.md](LICENSE.md) para detalhes.

## ✉️ Contato

**Nome**: Flávio Henrique Barbosa  
**LinkedIn**: [linkedin.com/in/flávio-henrique-barbosa-38465938](https://www.linkedin.com/in/fl%C3%A1vio-henrique-barbosa-38465938)  
**Email**: flaviohenriquehb777@outlook.com
