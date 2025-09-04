
import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px


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
    st.plotly_chart(fig, use_container_width=True)

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
    st.plotly_chart(fig1, use_container_width=True)

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
    st.plotly_chart(fig2, use_container_width=True)

    # Top 10 filmes recomendados (desempate por votos)
    top_recommendations = popular.sort_values(
        ['IMDB_Rating', 'No_of_Votes'], ascending=[False, False]
    ).head(10)[['Series_Title', 'IMDB_Rating', 'No_of_Votes', 'Genre', 'Released_Year']]

    st.subheader("🏆 Top 10 Filmes Recomendados")
    st.dataframe(top_recommendations)

    return top_recommendations

def page_analysis():
    st.title("Análise Exploratória de Dados (EDA)")
    
    db_path = "data/production.db"
    try:
        df = carregar_dados_db(db_path)
    except Exception as e:
        st.error(f"Erro ao carregar dados do banco: {e}")
        return

    # Criar abas para diferentes tipos de análise
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
    # Nova análise focada em faturamento
    # analise_fatores_faturamento(df)

    with tab4:
        st.subheader("Análise das Visões Gerais dos Filmes")
        st.markdown("Nesta seção, vamos explorar as visões gerais dos filmes, incluindo aspectos como direção, elenco e produção.")

if __name__ == "__main__":
    page_analysis()