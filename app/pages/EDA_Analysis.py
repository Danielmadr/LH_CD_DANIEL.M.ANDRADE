
import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import re
import sys
from pathlib import Path
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Adicionar pasta raiz ao path para importar config
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from config import DATABASE_PATH


def carregar_dados_db(db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM movies", conn)
    conn.close()
    return df

def plot_correlation_matrix(df: pd.DataFrame):
    # Selecionar apenas colunas numéricas relevantes
    colunas_relevantes = [
        'Released_Year', 'Runtime', 'IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Gross'
    ]
    df_corr = df[colunas_relevantes].copy()
    corr = df_corr.corr()
    # Melhorar nomes das colunas para o gráfico
    corr.index.name = 'Variável'
    corr.columns.name = 'Variável'
    fig = px.imshow(
        corr,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        title="Matriz de Correlação",
        labels=dict(color="Correlação"),
        x=corr.columns,
        y=corr.index
    )
    fig.update_layout(
        title_x=0.5,
        width=800,
        height=600
    )
    st.plotly_chart(fig, width='stretch')

    # Extrair correlações mais altas (excluindo diagonal)
    st.subheader("Principais Correlações")
    corr_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            var1 = corr.columns[i]
            var2 = corr.columns[j]
            corr_val = corr.iloc[i, j]
            if not pd.isna(corr_val):
                corr_pairs.append({
                    'Variável 1': var1,
                    'Variável 2': var2,
                    'Força': abs(corr_val)
                })
    if corr_pairs:
        corr_df = pd.DataFrame(corr_pairs).sort_values('Força', ascending=False).reset_index(drop=True)
        st.dataframe(corr_df.head(10))
    return fig

def plot_top_recommendations(df: pd.DataFrame):
    # Critérios: Alta nota IMDB (>=8.0) + Muitos votos (acima do percentil 70) + Gêneros populares
    st.markdown(
        """
        **Critérios para recomendação:**
        - Nota IMDB ≥ 8.0
        - Número de votos acima do percentil 70
        - Gêneros mais frequentes entre os melhores
        
        """
    )
    high_rated = df[df['IMDB_Rating'] >= 8.0]
    popular = high_rated[high_rated['No_of_Votes'] >= high_rated['No_of_Votes'].quantile(0.7)]

    # Gráfico 1: Distribuição de Rating vs Votos (apenas filmes populares)
    fig1 = px.scatter(
        popular,
        x='No_of_Votes',
        y='IMDB_Rating',
        color='Genre',
        size='IMDB_Rating',
        hover_data=['Series_Title', 'Released_Year'],
        title='Rating vs Popularidade (Filmes Populares)',
        subtitle= "Nota: Um filme é considerado popular nesta análise se possui nota IMDB igual ou superior a 8.0 e está entre os 30% com maior número de votos.",
        labels={'No_of_Votes': 'Número de Votos', 'IMDB_Rating': 'Rating IMDB'}
    )
    fig1.update_xaxes(type='log')
    st.plotly_chart(fig1, width='stretch')

    # Gráfico 2: Distribuição por gênero dos filmes bem avaliados
    genre_counts = popular['Genre'].value_counts().head(10)
    fig2 = px.bar(
        x=genre_counts.values,
        y=genre_counts.index,
        orientation='h',
        title='Gêneros Mais Populares (Rating ≥ 8.0)',
        labels={'x': 'Número de Filmes', 'y': 'Gênero'}
    )
    fig2.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig2, width='stretch')

    # Top 10 filmes recomendados (desempate por votos)
    top_recommendations = popular.sort_values(
        ['IMDB_Rating', 'No_of_Votes'], ascending=[False, False]
    ).head(10)[['Series_Title', 'IMDB_Rating', 'No_of_Votes', 'Genre', 'Released_Year']]

    st.subheader("🏆 Top 10 Filmes Recomendados")
    st.dataframe(top_recommendations)

    return top_recommendations

def analise_fatores_faturamento(df: pd.DataFrame):
    """
    Análise completa dos fatores que influenciam o faturamento dos filmes
    """    
    # Filtrar apenas filmes com dados de faturamento
    df_revenue = df.dropna(subset=['Gross']).copy()
    
    if len(df_revenue) == 0:
        st.warning("Não há dados suficientes de faturamento para análise.")
        return
    
    st.markdown(f"**Dados disponíveis:** {len(df_revenue):,} filmes com informações de faturamento")
    
    # Criar categorias de faturamento
    df_revenue['Faturamento_Categoria'] = pd.cut(
        df_revenue['Gross'], 
        bins=[0, 50_000_000, 200_000_000, float('inf')],
        labels=['Baixo (<$50M)', 'Médio ($50M-$200M)', 'Alto (>$200M)']
    )
    
    # 1. CORRELAÇÃO COM FATURAMENTO
    st.subheader("🔍 Correlação das Variáveis com Faturamento")
    
    # Calcular correlações
    colunas_numericas = ['Released_Year', 'Runtime', 'IMDB_Rating', 'Meta_score', 'No_of_Votes']
    correlacoes = []
    
    for col in colunas_numericas:
        if col in df_revenue.columns:
            corr = df_revenue[col].corr(df_revenue['Gross'])
            if not pd.isna(corr):
                correlacoes.append({'Variável': col, 'Correlação': corr, 'Abs_Correlação': abs(corr)})
    
    if correlacoes:
        corr_df = pd.DataFrame(correlacoes).sort_values('Abs_Correlação', ascending=False)
        
        # Gráfico de barras das correlações
        fig_corr = px.bar(
            corr_df, 
            x='Correlação', 
            y='Variável',
            orientation='h',
            title='Correlação das Variáveis com Faturamento',
            color='Correlação',
            color_continuous_scale='RdBu_r',
            text='Correlação'
        )
        fig_corr.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig_corr.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_corr, width='stretch')
        
        # Tabela de correlações
        st.dataframe(corr_df[['Variável', 'Correlação']].round(3))
    
    # 2. ANÁLISE POR RATING IMDB
    st.subheader("⭐ Faturamento por Faixa de Rating IMDB")
    
    # Criar faixas de rating
    df_revenue['Rating_Faixa'] = pd.cut(
        df_revenue['IMDB_Rating'], 
        bins=[0, 6.5, 7.5, 8.5, 10],
        labels=['Baixo (≤6.5)', 'Médio (6.5-7.5)', 'Alto (7.5-8.5)', 'Excelente (>8.5)']
    )
    
    rating_stats = df_revenue.groupby('Rating_Faixa', observed=True)['Gross'].agg([
        'count', 'mean', 'median', 'std'
    ]).round(2)
    rating_stats.columns = ['Quantidade', 'Média', 'Mediana', 'Desvio Padrão']
    
    # Box plot por rating
    fig_box = px.box(
        df_revenue, 
        x='Rating_Faixa', 
        y='Gross',
        title='Distribuição de Faturamento por Faixa de Rating IMDB'
    )
    fig_box.update_yaxes(title='Faturamento (US$)')
    st.plotly_chart(fig_box, width='stretch')
    
    st.dataframe(rating_stats)
    
    # 3. ANÁLISE POR GÊNERO
    st.subheader("🎬 Faturamento por Gênero")
    
    # Processar gêneros (pegar apenas o primeiro gênero)
    df_revenue['Genero_Principal'] = df_revenue['Genre'].str.split(',').str[0]
    genero_stats = df_revenue.groupby('Genero_Principal')['Gross'].agg([
        'count', 'mean', 'median'
    ]).sort_values('mean', ascending=False).head(10)
    genero_stats.columns = ['Quantidade', 'Faturamento_Médio', 'Faturamento_Mediano']
    
    # Filtrar gêneros com pelo menos 5 filmes
    genero_stats_filtered = genero_stats[genero_stats['Quantidade'] >= 5]
    
    fig_genero = px.bar(
        x=genero_stats_filtered.index,
        y=genero_stats_filtered['Faturamento_Médio'],
        title='Faturamento Médio por Gênero (min. 5 filmes)',
        labels={'x': 'Gênero', 'y': 'Faturamento Médio (US$)'}
    )
    fig_genero.update_xaxes(tickangle=45)
    st.plotly_chart(fig_genero, width='stretch')
    
    st.dataframe(genero_stats_filtered)
    
    # 4. ANÁLISE POR ANO DE LANÇAMENTO
    st.subheader("📅 Evolução do Faturamento por Década")
    
    # Criar décadas
    df_revenue['Decada'] = (df_revenue['Released_Year'] // 10) * 10
    decada_stats = df_revenue.groupby('Decada')['Gross'].agg([
        'count', 'mean', 'median'
    ]).dropna()
    
    fig_decada = px.line(
        x=decada_stats.index,
        y=decada_stats['mean'],
        title='Evolução do Faturamento Médio por Década',
        labels={'x': 'Década', 'y': 'Faturamento Médio (US$)'},
        markers=True
    )
    st.plotly_chart(fig_decada, width='stretch')
    return df_revenue

def analise_overview_insights(df: pd.DataFrame):
    """
    Análise completa da coluna Overview para extrair insights e inferir gêneros
    """    
    # Filtrar dados válidos de Overview
    df_overview = df.dropna(subset=['Overview']).copy()
    df_overview = df_overview[df_overview['Overview'].str.len() > 10]  # Filtrar descrições muito curtas
    
    if len(df_overview) == 0:
        st.warning("Não há dados suficientes de Overview para análise.")
        return
    
    st.markdown(f"**Dados disponíveis:** {len(df_overview):,} filmes com descrições válidas")
    
    # ========== ANÁLISES BÁSICAS DE TEXTO ==========
    st.subheader("📝 Características das Descrições")
    
    # Estatísticas básicas
    df_overview['overview_length'] = df_overview['Overview'].str.len()
    df_overview['overview_words'] = df_overview['Overview'].str.split().str.len()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Comprimento médio", 
            f"{df_overview['overview_length'].mean():.0f} chars"
        )
    
    with col2:
        st.metric(
            "Palavras médias", 
            f"{df_overview['overview_words'].mean():.0f} palavras"
        )
    
    with col3:
        st.metric(
            "Descrição mais longa", 
            f"{df_overview['overview_length'].max()} chars"
        )
    
    with col4:
        st.metric(
            "Descrição mais curta", 
            f"{df_overview['overview_length'].min()} chars"
        )
    
    # Distribuição do comprimento das descrições
    fig_length = px.histogram(
        df_overview,
        x='overview_length',
        nbins=50,
        title='Distribuição do Comprimento das Descrições',
        labels={'overview_length': 'Comprimento (caracteres)', 'count': 'Frequência'}
    )
    st.plotly_chart(fig_length, width='stretch')
    
    # ========== ANÁLISE DE PALAVRAS-CHAVE ==========
    # Definir stop words básicas
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
        'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 
        'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'his', 'her', 'its', 
        'their', 'he', 'she', 'it', 'they', 'him', 'them', 'this', 'that', 'these', 'those', 
        'when', 'where', 'why', 'how', 'what', 'who', 'which', 'whom', 'whose', 'if', 'because', 
        'while', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 
        'over', 'under', 'again', 'further', 'then', 'once', 'one', 'two', 'three', 'as', 'from',
        'into', 'through', 'between', 'about', 'against', 'upon', 'within', 'without', 'around',
        'becomes', 'become', 'finds', 'find', 'gets', 'get', 'goes', 'go', 'comes', 'come',
        'takes', 'take', 'makes', 'make', 'gives', 'give', 'tries', 'try', 'tells', 'tell'
    }
    st.subheader("🔍 Palavras-Chave Mais Frequentes")
    
    # Função para limpar e extrair palavras
    def extract_keywords(text):
        # Converter para minúsculo e remover pontuação
        text = re.sub(r'[^\w\s]', ' ', text.lower())
    
        words = text.split()
        # Filtrar palavras com mais de 2 caracteres que não são stop words
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        return keywords
    
    # Extrair todas as palavras-chave
    all_keywords = []
    for overview in df_overview['Overview'].dropna():
        all_keywords.extend(extract_keywords(str(overview)))
    
    # Contar frequências
    keyword_counts = Counter(all_keywords)
    top_keywords = keyword_counts.most_common(20)
    
    if top_keywords:
        keywords_df = pd.DataFrame(top_keywords, columns=['Palavra', 'Frequência'])
        
        fig_keywords = px.bar(
            keywords_df,
            x='Frequência',
            y='Palavra',
            orientation='h',
            title='Top 20 Palavras-Chave Mais Frequentes',
            labels={'Frequência': 'Número de Ocorrências', 'Palavra': 'Palavra-Chave'}
        )
        fig_keywords.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_keywords, width='stretch')
        
        st.dataframe(keywords_df.head(15))
    
    # ========== ANÁLISE POR GÊNERO ==========
    st.subheader("🎭 Palavras-Chave por Gênero")
    
    # Processar gêneros
    df_overview['Genero_Principal'] = df_overview['Genre'].str.split(',').str[0]
    
    # Analisar palavras-chave por gênero (top 5 gêneros mais comuns)
    top_genres = df_overview['Genero_Principal'].value_counts().head(5).index.tolist()
    
    genre_keywords = {}
    for genre in top_genres:
        genre_overviews = df_overview[df_overview['Genero_Principal'] == genre]['Overview']
        genre_words = []
        for overview in genre_overviews.dropna():
            genre_words.extend(extract_keywords(str(overview)))
        
        # Palavras mais comuns neste gênero
        genre_keyword_counts = Counter(genre_words)
        genre_keywords[genre] = genre_keyword_counts.most_common(10)
    
    # Criar tabs para cada gênero
    if len(top_genres) > 0:
        genre_tabs = st.tabs([f"📽️ {genre}" for genre in top_genres])
        
        for i, genre in enumerate(top_genres):
            with genre_tabs[i]:
                if genre in genre_keywords and genre_keywords[genre]:
                    genre_df = pd.DataFrame(genre_keywords[genre], columns=['Palavra', 'Frequência'])
                    
                    fig_genre = px.bar(
                        genre_df,
                        x='Frequência',
                        y='Palavra',
                        orientation='h',
                        title=f'Palavras-Chave Mais Comuns - {genre}',
                        color='Frequência',
                        color_continuous_scale='viridis'
                    )
                    fig_genre.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_genre, width='stretch')
                    
                    st.dataframe(genre_df)
    
    # ========== NUVEM DE PALAVRAS POR GÊNERO ==========
    st.subheader("☁️ Nuvem de Palavras por Gênero")
    st.markdown("Visualização das palavras mais frequentes em forma de nuvem para cada gênero de filme.")
    
    # Criar nuvens de palavras para os top 5 gêneros
    top_5_genres = df_overview['Genero_Principal'].value_counts().head(5).index.tolist()
    
    # Garantir que Crime e Biography estejam incluídos se existirem no dataset
    genres_of_interest = ['Crime', 'Biography']
    for genre in genres_of_interest:
        if genre in df_overview['Genero_Principal'].values and genre not in top_5_genres:
            top_5_genres.append(genre)
    
    # Limitar a um máximo de 5 gêneros para melhor visualização
    display_genres = top_5_genres[:5]
    
    if len(display_genres) > 0:
        # Criar tabs para cada gênero (similar ao gráfico de palavras-chave)
        wordcloud_tabs = st.tabs([f"☁️ {genre}" for genre in display_genres])
        
        for i, genre in enumerate(display_genres):
            with wordcloud_tabs[i]:
                # Combinar todas as descrições do gênero
                genre_overviews = df_overview[df_overview['Genero_Principal'] == genre]['Overview']
                genre_text = ' '.join(genre_overviews.dropna().astype(str))
                
                # Limpar o texto
                genre_text_clean = re.sub(r'[^\w\s]', ' ', genre_text.lower())
                
                if len(genre_text_clean.strip()) > 0:
                    try:
                        # Mostrar estatísticas do gênero
                        num_filmes = len(genre_overviews)
                        avg_length = genre_overviews.str.len().mean()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Filmes analisados", f"{num_filmes:,}")
                        with col2:
                            st.metric("Comprimento médio descrição", f"{avg_length:.0f} chars")
                        
                        # Criar a nuvem de palavras
                        wordcloud = WordCloud(
                            width=800, 
                            height=500, 
                            background_color='white',
                            stopwords=stop_words,
                            max_words=75,
                            colormap='viridis',
                            min_font_size=12,
                            relative_scaling=0.5,
                            max_font_size=100
                        ).generate(genre_text_clean)
                        
                        # Criar figura matplotlib com tamanho maior
                        fig, ax = plt.subplots(figsize=(12, 8))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        ax.set_title(f'Nuvem de Palavras - {genre}', fontsize=16, fontweight='bold', pad=20)
                        
                        # Exibir no Streamlit
                        st.pyplot(fig, width='stretch')
                        plt.close(fig)  # Limpar a figura para evitar warnings
                        
                    except Exception as e:
                        st.warning(f"Não foi possível gerar nuvem de palavras para {genre}: {str(e)}")
                else:
                    st.warning(f"Texto insuficiente para gerar nuvem de palavras para {genre}")
    
    return df_overview

def page_analysis():
    st.title("Análise Exploratória de Dados (EDA)")
    
    db_path = str(DATABASE_PATH)
    try:
        df = carregar_dados_db(db_path)
    except Exception as e:
        st.error(f"Erro ao carregar dados do banco: {e}")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["🔗 Correlações Gerais", "🏆 Recomendações", "🎯 Análise de Faturamento", "📊 Análise das Visões Gerais dos Filmes"])

    with tab1:
        st.subheader("Matriz de Correlação entre Variáveis Numéricas")
        st.markdown("Veja como as principais variáveis numéricas dos filmes se relacionam entre si. A matriz de correlação abaixo destaca as relações mais fortes, tanto positivas quanto negativas, que podem influenciar o desempenho e a recepção dos filmes.")
        plot_correlation_matrix(df)
    
    with tab2:
        st.subheader("Top 10 Filmes Recomendados (Alta Nota e Popularidade)")
        st.markdown("Confira os filmes mais recomendados, considerando tanto a avaliação do público quanto a popularidade. Os gráficos a seguir mostram a distribuição dos filmes populares e os gêneros mais frequentes entre eles.")
        plot_top_recommendations(df)
    
    with tab3:
        st.subheader("Análise de Faturamento")
        st.markdown("Nesta seção, vamos explorar os fatores que influenciam o faturamento dos filmes.")
        analise_fatores_faturamento(df)


    with tab4:
        st.subheader("Análise das Visões Gerais dos Filmes")
        st.markdown("Nesta seção, vamos explorar as visões gerais dos filmes, incluindo aspectos como direção, elenco e produção.")
        analise_overview_insights(df)

if __name__ == "__main__":
    page_analysis()