# 🎬 Projeto de Predição de Notas IMDB

Este projeto é uma aplicação completa de Machine Learning para predição de notas IMDB de filmes, desenvolvida com **Streamlit** e incluindo análise exploratória de dados (EDA) e três modelos de ML treinados.

![Tela Principal da Aplicação](docs/screenshots/main_page.png)
_[Placeholder: Screenshot da página principal da aplicação]_

## 🚀 Funcionalidades

### 📊 Análise Exploratória de Dados (EDA)

- Visualizações interativas dos dados
- Estatísticas descritivas
- Análise de correlações
- Distribuições de variáveis importantes
- Insights sobre o dataset IMDB

![Página de EDA](docs/screenshots/eda_page.png)
_[Placeholder: Screenshot da página de análise EDA]_

### 🤖 Predição de Notas IMDB

- **3 Modelos disponíveis:**
  - RandomForest
  - XGBoost
  - XGBoost com Optuna (melhor performance)
- **Duas formas de entrada:**
  - Formulário interativo
  - Upload de dados em JSON
- **Métricas de avaliação** detalhadas para cada modelo
- **Features importantes** de cada modelo
- **Interpretação automática** das predições

![Página de Predição](docs/screenshots/prediction_page.png)
_[Placeholder: Screenshot da página de predição]_

### 🗄️ Banco de Dados SQLite

- Armazenamento local dos dados processados
- Scripts para inicialização e população do banco
- Consultas otimizadas para análise

## 📈 Performance dos Modelos

![Comparação de Modelos](docs/screenshots/model_comparison.png)
_[Placeholder: Gráfico comparativo da performance dos modelos]_

### 📊 RandomForest

- **RMSE:** 0.1937
- **MAE:** 0.1534
- **R²:** 0.5536
- **Top Features:**
  1. Log_Votes: 0.2643
  2. No_of_Votes: 0.2539
  3. Meta_score: 0.0933
  4. Released_Year: 0.0612
  5. Movie_Age: 0.0610

### 📊 XGBoost

- **RMSE:** 0.1802
- **MAE:** 0.1453
- **R²:** 0.6136
- **Top Features:**
  1. Genre_Drama,Horror,Mystery: 0.1110
  2. No_of_Votes: 0.0838
  3. Genre_Drama,Thriller,War: 0.0727
  4. Genre_Action,Sci-Fi,Thriller: 0.0641
  5. Genre_Action,Sci-Fi: 0.0613

### 📊 XGBoost (Optuna) - **RECOMENDADO**

- **RMSE:** 0.0883
- **MAE:** 0.0697
- **R²:** 0.9073
- **Top Features:**
  1. High_Votes: 0.2094
  2. No_of_Votes: 0.0708
  3. Log_Votes: 0.0546
  4. Decade: 0.0535
  5. Is_Classic: 0.0487

![Feature Importance](docs/screenshots/feature_importance.png)
_[Placeholder: Gráfico de importância das features do melhor modelo]_

## 🛠️ Instalação e Execução

### Pré-requisitos

- Python 3.8+
- pip

### Configuração do Ambiente

1. **Clone o repositório:**

```bash
git clone <repository-url>
cd LH_CD_DANIEL.M.ANDRADE
```

2. **Crie um ambiente virtual:**

```bash
python -m venv .venv
```

3. **Ative o ambiente virtual:**

- Windows: `.venv\Scripts\activate`
- Linux/Mac: `source .venv/bin/activate`

4. **Instale as dependências:**

```bash
pip install streamlit pandas numpy scikit-learn xgboost optuna sqlite3
```

### Executando a Aplicação

```bash
streamlit run app/main.py
```

A aplicação estará disponível em: `http://localhost:8501`

![Aplicação Rodando](docs/screenshots/app_running.png)
_[Placeholder: Screenshot da aplicação rodando no navegador]_

### Inicializando o Banco de Dados (Opcional)

Se desejar recriar o banco de dados:

```bash
python scripts/init_db.py --csv data/raw/desafio_indicium_imdb.csv --db data/production.db
```

## 📁 Estrutura do Projeto

```
├── app/
│   ├── main.py                 # Aplicação principal Streamlit
│   └── pages/
│       ├── EDA_Analysis.py     # Página de análise exploratória
│       └── Prediction.py       # Página de predição
├── data/
│   ├── production.db           # Banco de dados SQLite
│   └── raw/
│       └── desafio_indicium_imdb.csv  # Dataset original
├── docs/
│   ├── refatoracao_ml_pipeline.md      # Documentação do pipeline
│   ├── relatorio_analise_estatistica.md # Relatório estatístico
│   ├── relatorio_engenharia_features.md # Documentação de features
│   └── screenshots/            # Screenshots da aplicação (placeholders)
├── models/
│   ├── rf_model.pkl           # Modelo RandomForest treinado
│   ├── xgb_model.pkl          # Modelo XGBoost treinado
│   └── xgb_optuna_model.pkl   # Modelo XGBoost otimizado
├── scripts/
│   ├── train_rf.py            # Script treinamento RandomForest
│   ├── train_xgb.py           # Script treinamento XGBoost
│   ├── train_xgb_optuna.py    # Script treinamento XGBoost + Optuna
│   ├── optuna_tuner.py        # Otimização de hiperparâmetros
│   ├── evaluate_models.py     # Avaliação e comparação de modelos
│   ├── predict.py             # Script de predição standalone
│   ├── preprocess.py          # Pré-processamento básico
│   ├── preprocessing.py       # Feature engineering avançado
│   ├── init_db.py             # Inicialização do banco SQLite
│   └── utils.py               # Funções utilitárias
├── temp/                      # Arquivos temporários e testes
├── test_model.ipynb          # Notebook de experimentação
└── README.MD                 # Este arquivo
```

## 🎯 Como Usar a Predição

![Formulário de Predição](docs/screenshots/prediction_form.png)
_[Placeholder: Screenshot do formulário de predição preenchido]_

### Via Formulário

1. Acesse a página "Predição de Notas"
2. Selecione o modelo desejado
3. Preencha os campos do formulário:
   - **Título do Filme**
   - **Ano de Lançamento**
   - **Certificação** (G, PG, PG-13, R, etc.)
   - **Duração** (em minutos)
   - **Gênero** (ex: Drama, Action,Adventure)
   - **Meta Score** (0-100)
   - **Número de Votos**
   - **Diretor e Atores**
   - **Sinopse**
4. Clique em "Prever Nota IMDB"

![Resultado da Predição](docs/screenshots/prediction_result.png)
_[Placeholder: Screenshot do resultado da predição com interpretação]_

### Via JSON

1. Acesse a aba "JSON"
2. Use o exemplo fornecido ou cole seu próprio JSON
3. Clique em "Prever via JSON"

![Predição via JSON](docs/screenshots/json_prediction.png)
_[Placeholder: Screenshot da predição via JSON]_

**Exemplo de JSON:**

```json
{
  "Series_Title": "The Shawshank Redemption",
  "Released_Year": 1994,
  "Certificate": "A",
  "Runtime": "142 min",
  "Genre": "Drama",
  "Overview": "Two imprisoned men bond over a number of years.",
  "Meta_score": 80.0,
  "Director": "Frank Darabont",
  "Star1": "Tim Robbins",
  "Star2": "Morgan Freeman",
  "Star3": "Bob Gunton",
  "Star4": "William Sadler",
  "No_of_Votes": 2343110,
  "Gross": "28,341,469"
}
```

## 📊 Interpretação dos Resultados

- **8.0+ :** 🌟 Excelente! Potencial clássico
- **7.0-7.9:** 👍 Muito bom! Vale a pena assistir
- **6.0-6.9:** 👌 Bom filme, mas pode não agradar a todos
- **< 6.0:** 👎 Nota baixa, possíveis problemas de qualidade

## 🔧 Features Importantes para Predição

- **Número de Votos:** Mais votos = melhor predição
- **Meta Score:** Nota da crítica especializada
- **Gênero:** Impacta significativamente na nota
- **Ano de Lançamento:** Filmes clássicos têm padrões diferentes
- **Duração:** Filmes muito curtos/longos podem ter notas diferentes

## 🛠️ Pipeline de Machine Learning

### 1. Pré-processamento

- Limpeza de dados ausentes
- Normalização de formatos
- Tratamento de outliers

### 2. Feature Engineering

- Criação de features derivadas (Movie_Age, Is_Classic, etc.)
- Transformações logarítmicas
- Encoding de variáveis categóricas
- Features de frequência para atores/diretores

### 3. Treinamento de Modelos

- **RandomForest:** Baseline robusto
- **XGBoost:** Modelo de gradient boosting
- **XGBoost + Optuna:** Otimização automática de hiperparâmetros

### 4. Avaliação

- Validação cruzada
- Métricas: RMSE, MAE, R²
- Análise de feature importance

## 🧪 Experimentos e Notebooks

O projeto inclui notebooks Jupyter para experimentação:

- `test_model.ipynb`: Experimentos com diferentes modelos e configurações
- Análise exploratória detalhada
- Testes de diferentes estratégias de feature engineering

## 📊 Scripts Standalone

### Treinamento de Modelos

```bash
# Treinar RandomForest
python scripts/train_rf.py

# Treinar XGBoost
python scripts/train_xgb.py

# Treinar XGBoost com Optuna
python scripts/train_xgb_optuna.py
```

### Avaliação de Modelos

```bash
python scripts/evaluate_models.py
```

### Predição via CLI

```bash
python scripts/predict.py models/xgb_optuna_model.pkl
```

## 🗄️ Banco de Dados

O projeto utiliza SQLite para armazenamento:

- **Tabela:** `movies`
- **Localização:** `data/production.db`
- **Estrutura:** Dados processados e prontos para ML

### Comandos úteis:

```bash
# Recriar banco do zero
python scripts/init_db.py --csv data/raw/desafio_indicium_imdb.csv --db data/production.db
```

## 📋 Dependências Completas

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=1.7.0
optuna>=3.0.0
sqlite3 (built-in)
```

## 🤝 Contribuição

Este projeto foi desenvolvido como parte de um desafio de Data Science. Sugestões e melhorias são bem-vindas!

### Como Contribuir:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📈 Próximos Passos

- [ ] Implementar mais modelos (CatBoost, LightGBM)
- [ ] Adicionar explicabilidade com SHAP
- [ ] Deploy em produção (Heroku/Streamlit Cloud)
- [ ] API REST para predições
- [ ] Interface mobile-friendly
- [ ] Cache de predições
- [ ] Monitoramento de drift de dados

## 🏆 Resultados Alcançados

- ✅ **R² = 0.9073** com XGBoost + Optuna
- ✅ **Interface intuitiva** para usuários não-técnicos
- ✅ **Pipeline completo** de ML
- ✅ **Documentação abrangente**
- ✅ **Código modular** e reutilizável

![Arquitetura do Sistema](docs/screenshots/system_architecture.png)
_[Placeholder: Diagrama da arquitetura do sistema]_

## 📞 Contato

- **Autor:** Daniel M. Andrade
- **Email:** [seu-email@exemplo.com]
- **LinkedIn:** [seu-linkedin]
- **GitHub:** [@Danielmadr](https://github.com/Danielmadr)

## 📄 Licença

Este projeto é de uso educacional e demonstrativo.

---

**⭐ Se este projeto foi útil para você, não esqueça de dar uma estrela!**
