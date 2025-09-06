import streamlit as st
from pages.EDA_Analysis import page_analysis
from pages.Prediction import page_prediction


def home():
    """Função principal da aplicação."""
    st.sidebar.title("Navegação")
    page = st.sidebar.radio("Ir para:", ["Análise EDA", "Predição de Notas"], index=0)

    if page == "Análise EDA":
        page_analysis()
    elif page == "Predição de Notas":
        page_prediction()


if __name__ == "__main__":
    home()
