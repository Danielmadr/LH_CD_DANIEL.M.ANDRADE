import streamlit as st
from EDA_Analysis import page_analysis
from Prediction import page_prediction


def main():
    st.set_page_config(
        page_title="IMDB Rating Predictor",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.com/Danielmadr/LH_CD_DANIEL.M.ANDRADE",
            "Report a bug": "https://github.com/Danielmadr/LH_CD_DANIEL.M.ANDRADE/issues",
            "About": "# Sistema de Predição de Notas IMDB\n\nDesenvolvido por Daniel M. Andrade\n\nUtiliza Machine Learning para prever notas de filmes no IMDB.",
        },
    )
    st.sidebar.title("Navegação")
    page = st.sidebar.radio("Ir para:", ["Análise EDA", "Predição de Notas"], index=0)

    if page == "Análise EDA":
        page_analysis()
    elif page == "Predição de Notas":
        page_prediction()


if __name__ == "__main__":
    main()
