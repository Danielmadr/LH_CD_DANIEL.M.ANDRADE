# Refatoração: Separação da Lógica de Preparação dos Dados e Treinamento

## 📋 Resumo da Refatoração

Este documento descreve a refatoração realizada para **separar claramente a lógica de preparação dos dados do treinamento do modelo**, criando uma arquitetura mais modular, maintível e reutilizável.

## ❌ Problemas Identificados na Estrutura Original

### 1. **Mistura de Responsabilidades**

- O script `train_model.py` misturava preparação de dados, feature engineering e treinamento
- Lógica de seleção de features duplicada entre treinamento e predição
- Difícil manutenção e teste de componentes individuais

### 2. **Inconsistências**

- Transformações diferentes no treino vs predição
- Valores ausentes tratados de formas distintas
- Sem garantia de que as mesmas features eram usadas

### 3. **Baixa Reutilização**

- Código de preprocessamento não reutilizável
- Necessidade de replicar lógica para novas predições
- Difícil escalar para novos datasets ou modelos

## ✅ Nova Arquitetura

### 🏗️ Princípios da Refatoração

1. **Single Responsibility Principle**: Cada classe tem uma responsabilidade específica
2. **DRY (Don't Repeat Yourself)**: Transformações reutilizáveis
3. **Consistency**: Mesmas transformações no treino e predição
4. **Modularity**: Componentes independentes e testáveis

### 📁 Estrutura de Arquivos

```
scripts/
├── data_processor.py      # 🔧 Preparação e transformação dos dados
├── train_model_v2.py      # 🤖 Treinamento e avaliação do modelo
├── predict.py             # 🎯 Interface para predições
└── preprocess.py          # 🧹 Limpeza básica (reutilizada)

models/
├── rf_model.pkl           # 💾 Modelo treinado
└── data_processor.pkl     # ⚙️ Processor configurado
```

## 🔧 Componentes da Nova Estrutura

### 1. **DataProcessor** (`data_processor.py`)

**Responsabilidade**: Toda a preparação e transformação dos dados

#### Features Principais:

- **Limpeza de dados**: Reutiliza `preprocess.py` existente
- **Feature Engineering**: Cria features derivadas automaticamente
- **Codificação categórica**: Label encoding consistente
- **Tratamento de valores ausentes**: Estratégias salvas e reutilizáveis
- **Serialização**: Salva configurações para reutilização

#### Novas Features Criadas:

```python
# Features temporais
Movie_Age = current_year - Released_Year
Is_Recent = Movie_Age <= 5
Is_Classic = Movie_Age >= 30

# Features de popularidade
Log_Votes = log(No_of_Votes + 1)
High_Votes = No_of_Votes > quantile(0.75)

# Features de receita
Log_Gross = log(Gross + 1)
Has_Gross = indicator se tem informação de receita

# Features de rating
Has_Meta_Score = indicator se tem Meta Score
Meta_score_filled = Meta Score com valores imputados

# Features de duração
Runtime_filled = Runtime com valores imputados
Is_Long_Movie = Runtime > 120 min
Is_Short_Movie = Runtime < 90 min
```

#### Uso:

```python
# Treinamento
processor = DataProcessor()
X, y = processor.fit_transform(df)
processor.save('models/data_processor.pkl')

# Predição
processor = DataProcessor.load('models/data_processor.pkl')
X_new = processor.transform(df_new)
```

### 2. **ModelTrainer** (`train_model_v2.py`)

**Responsabilidade**: Treinamento, avaliação e serialização do modelo

#### Features Principais:

- **Configuração flexível**: Parâmetros do modelo personalizáveis
- **Treinamento otimizado**: RandomForest com parâmetros melhorados
- **Avaliação completa**: Múltiplas métricas e feature importance
- **Serialização**: Salva modelo com metadados

#### Melhorias no Modelo:

```python
# Parâmetros otimizados
{
    'n_estimators': 300,      # Mais árvores
    'max_depth': 15,          # Controle de overfitting
    'min_samples_split': 5,   # Regularização
    'min_samples_leaf': 2,    # Regularização
    'random_state': 42,       # Reprodutibilidade
    'n_jobs': -1             # Paralelização
}
```

#### Uso:

```python
trainer = ModelTrainer()
trainer.train(X_train, y_train)
results = trainer.evaluate(X_test, y_test)
trainer.save_model('models/rf_model.pkl')
```

### 3. **MoviePredictor** (`predict.py`)

**Responsabilidade**: Interface simples para predições

#### Features Principais:

- **Carregamento automático**: Modelo + processor
- **Predição individual**: Para um filme
- **Predição em lote**: Para múltiplos filmes
- **Explicabilidade**: Mostra features mais importantes

#### Uso:

```python
predictor = MoviePredictor('models/rf_model.pkl', 'models/data_processor.pkl')

# Predição individual
rating = predictor.predict_single(movie_data)

# Predição em lote
ratings = predictor.predict_batch(movies_df)

# Com explicação
rating = predictor.explain_prediction(movie_data, top_features=5)
```

## 📊 Resultados da Refatoração

### 🎯 Métricas do Modelo Melhorado

```
AVALIAÇÃO DO MODELO
==================================================
RMSE: 0.1952  (melhor que antes: ~0.22)
MAE:  0.1516
MSE:  0.0381
R²:   0.4197

TOP FEATURES MAIS IMPORTANTES
==============================
1. No_of_Votes         : 0.2379
2. Log_Votes           : 0.2326  ← Nova feature
3. Meta_score_filled   : 0.1447
4. Released_Year       : 0.0841
5. Log_Gross           : 0.0816  ← Nova feature
```

### 🚀 Benefícios Alcançados

#### 1. **Melhor Performance**

- RMSE reduzido de ~0.22 para 0.1952
- Novas features melhoraram poder preditivo
- Feature engineering mais sofisticado

#### 2. **Maior Consistência**

- Mesmas transformações no treino e predição
- Tratamento consistente de valores ausentes
- Features idênticas em todos os usos

#### 3. **Facilidade de Manutenção**

- Código modular e testável
- Responsabilidades bem definidas
- Fácil adicionar novas features

#### 4. **Reutilização**

- Processor salvo e reutilizável
- Interface simples para predições
- Componentes independentes

## 🔄 Comparação: Antes vs Depois

### ❌ Estrutura Anterior

```python
# train_model.py - TUDO JUNTO
def load_for_training(source):
    df = load_data(source)
    # Limpeza inline
    # Feature selection inline
    # Split inline
    return X_train, X_test, y_train, y_test

def train_and_save(source):
    X_train, X_test, y_train, y_test = load_for_training(source)
    model = RandomForestRegressor()  # Parâmetros básicos
    model.fit(X_train, y_train)
    # Avaliação básica
    pickle.dump(model, f)
```

**Problemas**:

- Mistura preparação + treinamento
- Transformações não salvas
- Difícil reutilizar
- Features básicas apenas

### ✅ Nova Estrutura

```python
# Preparação (data_processor.py)
processor = DataProcessor()
X, y = processor.fit_transform(df)  # Feature engineering avançado
processor.save('data_processor.pkl')

# Treinamento (train_model_v2.py)
trainer = ModelTrainer(optimized_params)
trainer.train(X_train, y_train)
results = trainer.evaluate(X_test, y_test)  # Avaliação completa
trainer.save_model('rf_model.pkl')

# Predição (predict.py)
predictor = MoviePredictor('rf_model.pkl', 'data_processor.pkl')
rating = predictor.predict_single(movie_data)  # Automático e consistente
```

**Vantagens**:

- Responsabilidades separadas
- Componentes reutilizáveis
- Features avançadas
- Interface simples

## 🧪 Como Testar a Nova Estrutura

### 1. **Treinar Novo Modelo**

```bash
python scripts/train_model_v2.py --db data/production.db
```

### 2. **Fazer Predições**

```bash
python scripts/predict.py --demo
```

### 3. **Usar no Notebook**

```python
# Ver: demo_nova_estrutura.ipynb
predictor = MoviePredictor('models/rf_model.pkl', 'models/data_processor.pkl')
rating = predictor.predict_single(movie_data)
```

## 🔮 Próximos Passos Sugeridos

### 1. **Feature Engineering Avançado**

- Análise de sentimento no Overview
- Embeddings de texto para títulos/descrições
- Features de rede (colaborações diretor-ator)

### 2. **Modelos Mais Sofisticados**

- XGBoost ou LightGBM
- Redes neurais para features textuais
- Ensemble de modelos

### 3. **Pipeline de ML Completo**

- Validação cruzada
- Hyperparameter tuning
- Monitoramento de modelo em produção

### 4. **API de Produção**

```python
# FastAPI endpoint
@app.post("/predict")
def predict_movie_rating(movie: MovieData):
    return predictor.predict_single(movie.dict())
```

## 📚 Conclusão

A refatoração criou uma arquitetura **robusta, escalável e maintível** que:

- ✅ **Separa claramente** preparação de dados e treinamento
- ✅ **Melhora a performance** do modelo (RMSE de 0.22 → 0.19)
- ✅ **Facilita manutenção** com código modular
- ✅ **Aumenta reutilização** com componentes serializáveis
- ✅ **Garante consistência** entre treino e predição
- ✅ **Simplifica uso** com interfaces claras

A nova estrutura está pronta para **produção** e **escalonamento**, seguindo as melhores práticas de MLOps e engenharia de software.
