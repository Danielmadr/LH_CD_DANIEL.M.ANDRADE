# Relatório de Análise Estatística dos Dados IMDB

## 1. Tratamento dos Dados

Antes de iniciar as análises estatísticas, foi realizado um processo de preprocessamento nos dados brutos do arquivo CSV, visando garantir a qualidade e a padronização das informações inseridas no banco de dados. As principais etapas de tratamento foram:

- **Remoção de espaços extras** em colunas textuais, como títulos, nomes de diretores e atores.
- **Conversão de tipos**: colunas numéricas como ano de lançamento, duração, notas, número de votos e faturamento foram convertidas para os tipos adequados (inteiro ou float), facilitando cálculos e comparações.
- **Padronização de valores ausentes**: valores faltantes foram convertidos para `NULL` no banco de dados, seguindo o padrão de bancos relacionais.
- **Extração e limpeza de informações**: por exemplo, a coluna de duração foi convertida de texto (ex: "175 min") para número de minutos, e valores monetários tiveram vírgulas removidas para correta conversão numérica.
- **Padronização de categorias**: colunas como classificação indicativa e gêneros foram padronizadas para facilitar agrupamentos e filtros.
- **Remoção de duplicatas**: registros duplicados foram eliminados para evitar distorções nas análises.

Esses procedimentos garantiram que os dados estivessem prontos para análises exploratórias e correlacionais, minimizando ruídos e inconsistências.

## 2. Matriz de Correlação das Variáveis Numéricas

Para identificar possíveis relações entre as variáveis numéricas do conjunto de dados, foi gerada uma matriz de correlação utilizando as colunas relevantes: `Released_Year`, `Runtime`, `IMDB_Rating`, `Meta_score`, `No_of_Votes` e `Gross`. A matriz de correlação permite visualizar o grau de associação linear entre essas variáveis, auxiliando na identificação de padrões e possíveis insights para análises futuras.

A matriz foi construída utilizando a biblioteca Plotly Express, que possibilita uma visualização interativa dos coeficientes de correlação. Valores próximos de 1 indicam forte correlação positiva, valores próximos de -1 indicam forte correlação negativa e valores próximos de 0 indicam ausência de correlação linear significativa.

> **Nota:** Colunas como `id` e campos textuais foram desconsiderados nesta análise, pois não contribuem para a avaliação de correlação entre variáveis quantitativas.

---

## 3. Análise das Principais Correlações

A matriz de correlação revela algumas relações interessantes entre as variáveis do conjunto de dados de filmes:

- **No_of_Votes x Gross (0.59):** Existe uma forte correlação positiva entre o número de votos e a Faturamento. Isso sugere que filmes mais populares (com mais votos) tendem a arrecadar mais, o que é esperado, pois maior engajamento do público geralmente reflete em maior receita.

- **IMDB_Rating x No_of_Votes (0.48):** Filmes com mais votos tendem a ter notas mais altas no IMDB. Isso pode indicar que filmes bem avaliados atraem mais espectadores, ou que filmes populares recebem mais avaliações positivas.

- **Released_Year x Meta_score (-0.34):** Filmes mais recentes parecem ter uma leve tendência a receber notas mais altas dos críticos (Meta_score). Isso pode refletir mudanças nos critérios de avaliação ou evolução na produção cinematográfica.

- **IMDB_Rating x Meta_score (0.27):** Existe uma correlação positiva entre a nota do IMDB e a dos críticos, mas não é tão forte. Isso sugere que público e crítica nem sempre concordam plenamente sobre a qualidade dos filmes.

- **Runtime x IMDB_Rating (0.24):** Filmes mais longos tendem a ter avaliações um pouco melhores no IMDB, talvez porque roteiros mais desenvolvidos agradem mais o público.

- **Released_Year x Gross (0.23):** Filmes mais recentes tendem a ter um Faturamento um pouco maior, o que pode ser explicado por inflação, maior alcance de distribuição ou crescimento do mercado cinematográfico.

- **IMDB_Rating x Gross (0.10):** Apesar de esperado, a correlação entre nota do IMDB e Faturamento (**0.10**) é fraca, indicando que **sucesso financeiro nem sempre significa qualidade percebida**.