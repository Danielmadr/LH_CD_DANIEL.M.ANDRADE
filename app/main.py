import streamlit as st
from pages.EDA_Analysis import page_analysis


def home():
    """Função principal da aplicação."""
    st.sidebar.title("Navegação")
    page = st.sidebar.radio("Ir para:", ["Análise EDA"], index=0)

    if page == "Análise EDA":
        page_analysis()


if __name__ == "__main__":
    home()
