import streamlit as st
import pandas as pd
import json
import sys
import os
import numpy as np
from pathlib import Path
from datetime import datetime

# Adicionar o diretório raiz ao path para importar os módulos
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from scripts.utils import load_model

def preprocess_movies_df(df: pd.DataFrame) -> pd.DataFrame:
    """Função de pré-processamento básico dos dados do filme."""
    df = df.copy()
    
    # 1. Series_Title: remover espaços extras
    if 'Series_Title' in df.columns:
        df['Series_Title'] = df['Series_Title'].astype(str).str.strip()

    # 2. Released_Year: converter para inteiro, tratar ausentes
    if 'Released_Year' in df.columns:
        df['Released_Year'] = pd.to_numeric(df['Released_Year'], errors='coerce').astype('Int64')

    # 3. Certificate: padronizar valores, preencher ausentes
    if 'Certificate' in df.columns:
        df['Certificate'] = df['Certificate'].fillna('Unknown').astype(str).str.strip()

    # 4. Runtime: extrair minutos, converter para inteiro
    if 'Runtime' in df.columns:
        df['Runtime'] = df['Runtime'].astype(str).str.extract(r'(\d+)').astype(float)

    # 5. Genre: remover espaços extras após vírgula
    if 'Genre' in df.columns:
        df['Genre'] = df['Genre'].astype(str).str.replace(', ', ',', regex=False)

    # 6. IMDB_Rating: converter para float
    if 'IMDB_Rating' in df.columns:
        df['IMDB_Rating'] = pd.to_numeric(df['IMDB_Rating'], errors='coerce')

    # 7. Overview: remover quebras de linha e espaços extras
    if 'Overview' in df.columns:
        df['Overview'] = df['Overview'].astype(str).str.replace('\n', ' ', regex=False).str.strip()

    # 8. Meta_score: converter para float
    if 'Meta_score' in df.columns:
        df['Meta_score'] = pd.to_numeric(df['Meta_score'], errors='coerce')

    # 9. Director: remover espaços extras
    if 'Director' in df.columns:
        df['Director'] = df['Director'].astype(str).str.strip()

    # 10-13. Star1-4: remover espaços extras
    for star in ['Star1', 'Star2', 'Star3', 'Star4']:
        if star in df.columns:
            df[star] = df[star].astype(str).str.strip()

    # 14. No_of_Votes: remover vírgulas, converter para inteiro
    if 'No_of_Votes' in df.columns:
        df['No_of_Votes'] = (
            df['No_of_Votes']
            .astype(str)
            .str.replace(',', '', regex=False)
        )
        df['No_of_Votes'] = pd.to_numeric(df['No_of_Votes'], errors='coerce').astype('Int64')

    # 15. Gross: remover vírgulas, converter para float
    if 'Gross' in df.columns:
        df['Gross'] = (
            df['Gross']
            .astype(str)
            .str.replace(',', '', regex=False)
        )
        df['Gross'] = pd.to_numeric(df['Gross'], errors='coerce')

    # Remover duplicatas
    df = df.drop_duplicates()

    # Padronizar valores ausentes para None
    df = df.replace({np.nan: None})

    return df

def basic_clean(df):
    """Função de limpeza e feature engineering dos dados."""
    COLS_TO_DROP = ["id", "Series_Title", "Unnamed: 0"]

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    for col in COLS_TO_DROP:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    df = df.dropna()
    df = preprocess_movies_df(df)

    # Features derivadas
    if 'Released_Year' in df.columns:
        current_year = datetime.now().year
        # Convert to numeric first to avoid object dtype fillna warning
        df['Released_Year'] = pd.to_numeric(df['Released_Year'], errors='coerce')
        df['Released_Year'] = df['Released_Year'].fillna(df['Released_Year'].median())
        df['Movie_Age'] = current_year - df['Released_Year']
        df['Is_Recent'] = (df['Movie_Age'] <= 5).astype(int)
        df['Is_Classic'] = (df['Movie_Age'] >= 30).astype(int)

    if 'No_of_Votes' in df.columns:
        df['Log_Votes'] = np.log1p(df['No_of_Votes'].fillna(0))
        df['High_Votes'] = (df['No_of_Votes'] > df['No_of_Votes'].quantile(0.75)).astype(int)

    if 'Gross' in df.columns:
        filled_gross = pd.to_numeric(df['Gross'], errors="coerce").fillna(0)
        df['Log_Gross'] = np.log1p(filled_gross)
        df['Has_Gross'] = df['Gross'].notna().astype(int)
        df['Gross'] = filled_gross

    if 'Meta_score' in df.columns:
        df['Has_Meta_Score'] = df['Meta_score'].notna().astype(int)
        filled_meta_score = pd.to_numeric(df['Meta_score'], errors="coerce")
        meta_median = filled_meta_score.median()
        df['Meta_score'] = filled_meta_score.fillna(meta_median)

    if 'Runtime' in df.columns:
        filled_runtime = pd.to_numeric(df['Runtime'], errors="coerce")
        runtime_median = filled_runtime.median()
        df['Runtime_filled'] = filled_runtime.fillna(runtime_median)
        df['Is_Long_Movie'] = (df['Runtime_filled'] > 120).astype(int)
        df['Is_Short_Movie'] = (df['Runtime_filled'] < 90).astype(int)
        df['Runtime'] = filled_runtime.fillna(runtime_median)

    if "Genre" in df.columns:
        df['Genre'] = df['Genre'].fillna('Unknown')
        df["Main_Genre"] = df["Genre"].astype(str).str.split(",").str[0]
        genre_dummies = pd.get_dummies(df["Genre"].astype(str), prefix="Genre")
        df = pd.concat([df, genre_dummies], axis=1)

    for col in ["Director", "Star1", "Star2", "Star3", "Star4"]:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
            freq = df[col].value_counts()
            df[f"{col}_Freq"] = df[col].map(freq).fillna(0)

    if "Released_Year" in df.columns:
        df["Decade"] = (df["Released_Year"] // 10) * 10

    for col in ["Certificate", "Overview", "Genre", "Director", "Star1", "Star2", "Star3", "Star4"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    for col in df.columns:
        if col != 'IMDB_Rating' and df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'IMDB_Rating' in df.columns:
        df = df.dropna(subset=['IMDB_Rating'])

    df = df.fillna(0)
    df = df.infer_objects(copy=False)
    return df

# Informações dos modelos
MODEL_INFO = {
    "RandomForest": {
        "path": "models/rf_model.pkl",
        "description": "📊 RandomForest - Avaliação",
        "metrics": {
            "RMSE": 0.1937,
            "MAE": 0.1534,
            "R²": 0.5536
        },
        "top_features": [
            "Log_Votes: 0.2643",
            "No_of_Votes: 0.2539", 
            "Meta_score: 0.0933",
            "Released_Year: 0.0612",
            "Movie_Age: 0.0610"
        ]
    },
    "XGBoost": {
        "path": "models/xgb_model.pkl",
        "description": "📊 XGBoost - Avaliação",
        "metrics": {
            "RMSE": 0.1802,
            "MAE": 0.1453,
            "R²": 0.6136
        },
        "top_features": [
            "Genre_Drama,Horror,Mystery: 0.1110",
            "No_of_Votes: 0.0838",
            "Genre_Drama,Thriller,War: 0.0727",
            "Genre_Action,Sci-Fi,Thriller: 0.0641",
            "Genre_Action,Sci-Fi: 0.0613"
        ]
    },
    "XGBoost (Optuna)": {
        "path": "models/xgb_optuna_model.pkl",
        "description": "📊 XGBoost (Optuna) - Avaliação",
        "metrics": {
            "RMSE": 0.0883,
            "MAE": 0.0697,
            "R²": 0.9073
        },
        "top_features": [
            "High_Votes: 0.2094",
            "No_of_Votes: 0.0708",
            "Log_Votes: 0.0546",
            "Decade: 0.0535",
            "Is_Classic: 0.0487"
        ]
    }
}

def check_model_availability():
    """Verifica quais modelos estão disponíveis."""
    available_models = {}
    for model_name, model_info in MODEL_INFO.items():
        model_path = os.path.join(root_dir, model_info["path"])
        if os.path.exists(model_path):
            available_models[model_name] = model_info
            available_models[model_name]["full_path"] = model_path
    return available_models

def predict_movie_rating(model_path, input_data, reference_csv=None):
    """Função para fazer predição da nota do filme."""
    try:
        # Carregar modelo
        model = load_model(model_path)
        
        # Converter para DataFrame
        if isinstance(input_data, str):
            input_json = json.loads(input_data)
        else:
            input_json = input_data
            
        df_input = pd.DataFrame([input_json])
        
        # Carregar dataset de referência
        if reference_csv is None:
            reference_csv = os.path.join(root_dir, "data/raw/desafio_indicium_imdb.csv")
        
        df_ref = pd.read_csv(reference_csv)
        df_ref = basic_clean(df_ref)
        train_cols = df_ref.drop(columns=["IMDB_Rating"]).columns
        
        # Pré-processar entrada
        df_input_clean = basic_clean(df_input)
        
        # Garantir mesmas colunas que treino
        missing_cols = [col for col in train_cols if col not in df_input_clean.columns]
        if missing_cols:
            df_missing = pd.DataFrame(0, index=df_input_clean.index, columns=missing_cols)
            df_input_clean = pd.concat([df_input_clean, df_missing], axis=1)
        
        df_input_clean = df_input_clean[train_cols]
        
        # Fazer predição
        pred = model.predict(df_input_clean.astype("float32"))[0]
        return pred, True, "Predição realizada com sucesso!"
        
    except Exception as e:
        return None, False, f"Erro na predição: {str(e)}"

def display_model_info(model_name, model_info):
    """Exibe informações do modelo."""
    st.subheader(model_info["description"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Métricas:**")
        for metric, value in model_info["metrics"].items():
            st.write(f"   {metric} = {value:.4f}")
    
    with col2:
        st.write("**Top 5 Features Importantes:**")
        for i, feature in enumerate(model_info["top_features"], 1):
            st.write(f"   {i}. {feature}")

def page_prediction():
    """Página principal de predição."""
    st.title("🎬 Predição de Notas IMDB")
    st.write("Utilize os modelos treinados para prever a nota IMDB de um filme.")
    
    # Verificar modelos disponíveis
    available_models = check_model_availability()
    
    if not available_models:
        st.error("❌ Nenhum modelo encontrado! Verifique se os modelos estão na pasta 'models/'.")
        return
    
    # Seleção do modelo
    st.subheader("🤖 Seleção do Modelo")
    selected_model = st.selectbox(
        "Escolha o modelo para predição:",
        list(available_models.keys())
    )
    
    # Exibir informações do modelo selecionado
    if selected_model:
        with st.expander("📊 Informações do Modelo Selecionado", expanded=True):
            display_model_info(selected_model, available_models[selected_model])
            
            # Verificar se o modelo carrega corretamente
            try:
                model = load_model(available_models[selected_model]["full_path"])
                st.success("✅ Modelo carregado com sucesso!")
            except Exception as e:
                st.error(f"❌ Erro ao carregar modelo: {str(e)}")
                return
    
    st.divider()
    
    # Abas para diferentes tipos de entrada
    tab1, tab2 = st.tabs(["📝 Formulário", "📄 JSON"])
    
    with tab1:
        st.subheader("📝 Entrada via Formulário")
        
        # Criar formulário
        with st.form("movie_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                series_title = st.text_input("Título do Filme", value="", help="Nome do filme")
                released_year = st.number_input("Ano de Lançamento", min_value=1900, max_value=2025, value=2020)
                certificate = st.selectbox("Certificação", ["G", "PG", "PG-13", "R", "NC-17", "A", "U", "UA", "Unknown"])
                runtime = st.number_input("Duração (minutos)", min_value=1, max_value=500, value=120)
                
            with col2:
                genre = st.text_input("Gênero", value="Drama", help="Ex: Drama, Action,Adventure, Comedy")
                meta_score = st.number_input("Meta Score", min_value=0.0, max_value=100.0, value=70.0, step=0.1)
                no_of_votes = st.number_input("Número de Votos", min_value=1, value=100000)
                gross = st.text_input("Receita Bruta", value="", help="Ex: 100,000,000 (pode deixar vazio)")
            
            director = st.text_input("Diretor", value="")
            
            col3, col4, col5, col6 = st.columns(4)
            with col3:
                star1 = st.text_input("Ator Principal 1", value="")
            with col4:
                star2 = st.text_input("Ator Principal 2", value="")
            with col5:
                star3 = st.text_input("Ator Principal 3", value="")
            with col6:
                star4 = st.text_input("Ator Principal 4", value="")
            
            overview = st.text_area("Sinopse", value="", help="Descrição do filme")
            
            submitted = st.form_submit_button("🎯 Prever Nota IMDB")
            
            if submitted:
                # Criar dicionário com dados do formulário
                movie_data = {
                    'Series_Title': series_title,
                    'Released_Year': released_year,
                    'Certificate': certificate,
                    'Runtime': f'{runtime} min',
                    'Genre': genre,
                    'Overview': overview,
                    'Meta_score': meta_score,
                    'Director': director,
                    'Star1': star1,
                    'Star2': star2,
                    'Star3': star3,
                    'Star4': star4,
                    'No_of_Votes': no_of_votes,
                    'Gross': gross if gross else ""
                }
                
                # Fazer predição
                prediction, success, message = predict_movie_rating(
                    available_models[selected_model]["full_path"], 
                    movie_data
                )
                
                if success:
                    st.success(message)
                    st.balloons()
                    
                    # Exibir resultado
                    col_pred1, col_pred2, col_pred3 = st.columns([1, 2, 1])
                    with col_pred2:
                        st.metric(
                            label="🎬 Nota IMDB Prevista",
                            value=f"{prediction:.2f}",
                            delta=f"{prediction - 5.0:.2f}" if prediction >= 5.0 else f"{prediction - 5.0:.2f}"
                        )
                        
                        # Interpretação da nota
                        if prediction >= 8.0:
                            st.success("🌟 Excelente! Este filme pode ser um clássico!")
                        elif prediction >= 7.0:
                            st.info("👍 Muito bom! Vale a pena assistir!")
                        elif prediction >= 6.0:
                            st.warning("👌 Bom filme, mas pode não agradar a todos.")
                        else:
                            st.error("👎 Nota baixa. Pode ter problemas de qualidade.")
                else:
                    st.error(message)
    
    with tab2:
        st.subheader("📄 Entrada via JSON")
        st.write("Cole o JSON com os dados do filme no formato abaixo:")
        
        # Exemplo de JSON
        example_json = {
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
        
        with st.expander("📋 Exemplo de JSON"):
            st.json(example_json)
        
        json_input = st.text_area(
            "JSON do Filme:",
            value="",
            height=300,
            help="Cole aqui o JSON com os dados do filme"
        )
        
        col_json1, col_json2, col_json3 = st.columns([1, 1, 1])
        
        with col_json1:
            if st.button("📋 Usar Exemplo"):
                st.session_state.json_example = json.dumps(example_json, indent=2)
        
        with col_json2:
            if st.button("🎯 Prever via JSON"):
                if json_input.strip():
                    try:
                        # Validar JSON
                        json_data = json.loads(json_input)
                        
                        # Fazer predição
                        prediction, success, message = predict_movie_rating(
                            available_models[selected_model]["full_path"], 
                            json_data
                        )
                        
                        if success:
                            st.success(message)
                            st.balloons()
                            
                            # Exibir resultado
                            col_pred1, col_pred2, col_pred3 = st.columns([1, 2, 1])
                            with col_pred2:
                                st.metric(
                                    label="🎬 Nota IMDB Prevista",
                                    value=f"{prediction:.2f}",
                                    delta=f"{prediction - 5.0:.2f}" if prediction >= 5.0 else f"{prediction - 5.0:.2f}"
                                )
                                
                                # Interpretação da nota
                                if prediction >= 8.0:
                                    st.success("🌟 Excelente! Este filme pode ser um clássico!")
                                elif prediction >= 7.0:
                                    st.info("👍 Muito bom! Vale a pena assistir!")
                                elif prediction >= 6.0:
                                    st.warning("👌 Bom filme, mas pode não agradar a todos.")
                                else:
                                    st.error("👎 Nota baixa. Pode ter problemas de qualidade.")
                        else:
                            st.error(message)
                            
                    except json.JSONDecodeError as e:
                        st.error(f"❌ JSON inválido: {str(e)}")
                else:
                    st.warning("⚠️ Por favor, insira um JSON válido.")
        
        # Se há exemplo na sessão, mostrar
        if hasattr(st.session_state, 'json_example'):
            json_input = st.text_area(
                "JSON do Filme:",
                value=st.session_state.json_example,
                height=300,
                key="json_with_example"
            )
    
    # Informações adicionais
    st.divider()
    st.subheader("ℹ️ Informações Importantes")
    
    with st.expander("📝 Campos Obrigatórios e Dicas"):
        st.write("""
        **Campos mais importantes para a predição:**
        - **Número de Votos**: Quanto maior, melhor a predição
        - **Meta Score**: Nota da crítica especializada (0-100)
        - **Gênero**: Impacta significativamente na nota
        - **Ano de Lançamento**: Filmes mais antigos podem ter padrões diferentes
        - **Duração**: Filmes muito curtos ou muito longos podem ter notas diferentes
        
        **Formato dos campos:**
        - **Duração**: Número em minutos (ex: 120)
        - **Gênero**: Use vírgulas para separar múltiplos gêneros (ex: "Action,Adventure,Sci-Fi")
        - **Receita Bruta**: Pode incluir vírgulas (ex: "100,000,000") ou deixar vazio
        - **Certificação**: G, PG, PG-13, R, NC-17, A, U, UA, ou Unknown
        """)
    
    with st.expander("🤖 Sobre os Modelos"):
        st.write("""
        **RandomForest**: Modelo baseado em múltiplas árvores de decisão. Boa interpretabilidade.
        
        **XGBoost**: Gradient Boosting otimizado. Melhor performance que RandomForest.
        
        **XGBoost (Optuna)**: XGBoost com hiperparâmetros otimizados via Optuna. Melhor modelo disponível.
        
        **Recomendação**: Use o XGBoost (Optuna) para melhores resultados.
        """)

if __name__ == "__main__":
    page_prediction()
