

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

def page_analysis():
    st.title("Análise Exploratória de Dados (EDA)")
    st.write("Aqui você pode explorar os dados dos filmes.")

    db_path = "data/production.db"
    try:
        df = carregar_dados_db(db_path)
    except Exception as e:
        st.error(f"Erro ao carregar dados do banco: {e}")
        return

    st.subheader("Matriz de Correlação (colunas numéricas)")
    plot_correlation_matrix(df)
    

if __name__ == "__main__":
    page_analysis()