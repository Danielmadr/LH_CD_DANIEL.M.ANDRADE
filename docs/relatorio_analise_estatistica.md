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

---

*Continue o relatório a partir deste ponto, detalhando as análises estatísticas realizadas.*
