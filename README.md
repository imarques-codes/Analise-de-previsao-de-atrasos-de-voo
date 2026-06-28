# Análise de Previsão de Atrasos de Voos

## Sobre o projeto

Este projeto apresenta uma análise de dados e modelagem preditiva para estimar atrasos em voos, utilizando variáveis operacionais como companhia aérea, tipo de aeronave, origem, horário de chegada, tipo de voo, feriado e dia da semana.

O notebook foi desenvolvido em Python no Google Colab e segue um fluxo completo de projeto de Ciência de Dados, passando por exploração dos dados, visualização, engenharia de atributos, codificação de variáveis, limpeza da base, treinamento de modelos, validação cruzada, seleção de features, otimização de hiperparâmetros e salvamento do modelo final.

---

## Objetivo

O objetivo principal é construir um modelo de Machine Learning capaz de prever o atraso de voos em minutos a partir das características disponíveis na base de dados.

O projeto busca responder perguntas como:

- Quais companhias aéreas apresentam maior atraso médio?
- Voos em feriados possuem atrasos maiores?
- O tipo de voo influencia o atraso?
- Determinados tipos de aeronave aparecem com maior frequência?
- Quais variáveis são mais importantes para prever atrasos?
- Um modelo de regressão consegue estimar o atraso de um voo com desempenho superior a um modelo baseline?

---

## Problema de negócio

Atrasos em voos impactam passageiros, companhias aéreas, aeroportos e toda a operação logística envolvida no transporte aéreo.

Com uma análise orientada por dados, é possível identificar padrões que ajudam a entender fatores associados aos atrasos e construir modelos preditivos que apoiem decisões operacionais, planejamento de rotas, gestão de frota e melhoria da experiência do cliente.

---

## Tecnologias utilizadas

- Python
- Google Colab
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Yellowbrick
- Pickle

---

## Como funciona o projeto

O notebook segue as seguintes etapas:

### 1. Importação das bibliotecas

São importadas bibliotecas para manipulação de dados, visualização, modelagem preditiva e avaliação dos modelos.

Principais bibliotecas utilizadas:

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
```

---

### 2. Leitura dos dados

O projeto utiliza o arquivo:

```text
flights.csv
```

No notebook original, a leitura está configurada para o ambiente do Google Colab:

```python
dados = pd.read_csv('/content/flights.csv')
```

Para executar localmente, recomenda-se colocar o arquivo na pasta `data/` e ajustar o caminho para:

```python
dados = pd.read_csv('data/flights.csv')
```

---

### 3. Entendimento inicial da base

São realizadas verificações iniciais para compreender a estrutura do dataset:

- Visualização das primeiras linhas;
- Visualização das últimas linhas;
- Dimensão da base;
- Estatísticas descritivas de variáveis numéricas;
- Estatísticas de variáveis categóricas;
- Informações gerais da base com `info()`.

---

### 4. Visualização dos dados

O projeto utiliza gráficos para analisar padrões relacionados aos atrasos dos voos.

São criadas visualizações como:

- Atraso médio por companhia aérea;
- Quantidade de voos por companhia aérea;
- Atraso médio por tipo de voo;
- Quantidade de voos por tipo de voo;
- Relação entre feriados e atrasos médios;
- Quantidade de voos por tipo de aeronave;
- Histogramas de horários de chegada e saída;
- Boxplot e histograma da variável de atraso.

---

### 5. Feature Engineering

São criadas novas variáveis a partir das colunas existentes.

O notebook converte informações de ano e dia do ano em uma coluna de data:

```python
dados['date'] = pd.to_datetime(
    dados['year'].astype(str) + '-' + (dados['day'] + 1).astype(str),
    format='%Y-%j'
)
```

Também são criadas variáveis como:

- `is_weekend`: identifica se o voo ocorreu em final de semana;
- `day_name`: adiciona o nome do dia da semana.

Essas variáveis ajudam o modelo a capturar padrões temporais relacionados aos atrasos.

---

### 6. Feature Encoding

As variáveis categóricas são transformadas em variáveis numéricas para permitir o treinamento dos modelos de Machine Learning.

O projeto utiliza:

- Substituição de categorias binárias;
- One-Hot Encoding com `pd.get_dummies()`.

Variáveis categóricas codificadas:

```python
variaveis_categoricas = ['airline', 'aircraft_type', 'origin', 'day_name']
```

---

### 7. Limpeza dos dados

Algumas colunas são removidas por não serem necessárias ao modelo ou por apresentarem alta correlação com outras variáveis.

A coluna `departure_time`, por exemplo, apresenta alta correlação com `arrival_time`, sendo removida para reduzir redundância.

Colunas removidas:

```python
['flight_id', 'departure_time', 'day', 'year', 'date']
```

---

## Modelagem preditiva

### Modelo baseline — DummyRegressor

O primeiro modelo utilizado é o `DummyRegressor`, que serve como baseline para comparação.

Resultado observado no notebook:

```text
RMSE: 23.2241
MAE: 18.6127
R²: -0.0000
```

Esse modelo funciona como referência mínima, pois realiza previsões simples sem aprender padrões relevantes dos dados.

---

### Modelo RandomForestRegressor

Em seguida, é treinado um modelo `RandomForestRegressor` com profundidade máxima definida.

Resultado observado no notebook:

```text
RMSE: 13.7301
MAE: 11.0138
R²: 0.6505
```

O modelo apresentou desempenho superior ao baseline, indicando que conseguiu capturar padrões relevantes nos dados.

---

### Validação cruzada

O projeto aplica validação cruzada com `KFold`, avaliando o modelo com as métricas:

- MAE;
- RMSE;
- R².

Resultado médio observado:

```text
MAE médio: -11.151
RMSE médio: -13.883
R² médio: 0.638
```

A validação cruzada ajuda a verificar a estabilidade do modelo em diferentes divisões da base.

---

## Seleção de features

O projeto utiliza a importância das variáveis do `RandomForestRegressor` para identificar os principais fatores associados à previsão de atrasos.

Entre as variáveis com maior importância no notebook estão:

- `airline_BZ`;
- `is_holiday`;
- `aircraft_type_Airbus A320`;
- `aircraft_type_Airbus A330`;
- `aircraft_type_Embraer E175`;
- `arrival_time`.

Essa etapa contribui para entender quais características possuem maior impacto na previsão do atraso.

---

## Otimização de hiperparâmetros

O projeto utiliza `GridSearchCV` para buscar a melhor combinação de hiperparâmetros do modelo Random Forest.

Parâmetros testados:

```python
param_grid = {
    'max_depth': [5, 10, 15],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [2, 4, 6],
    'n_estimators': [100, 150, 200]
}
```

Melhores parâmetros encontrados no notebook:

```text
max_depth: 10
min_samples_leaf: 3
min_samples_split: 2
n_estimators: 200
```

Resultado final do modelo otimizado:

```text
RMSE: 13.2308
MAE: 10.6296
R²: 0.6754
```

---

## Salvamento do modelo

Ao final do projeto, o melhor modelo é salvo em arquivo `.pkl` com Pickle:

```python
with open('model_producao.pkl', 'wb') as file:
    pickle.dump(model_grid.best_estimator_, file)
```

Esse arquivo pode ser utilizado futuramente para deploy ou integração com uma aplicação.

---

## Estrutura sugerida do repositório

```text
flight-delay-prediction-analysis/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── flight_delay_prediction_analysis.ipynb
│
├── data/
│   └── flights.csv
│
├── models/
│   └── model_producao.pkl
│
└── img/
```

---

## Como executar no Google Colab

1. Abra o notebook no Google Colab.
2. Faça upload do arquivo `flights.csv`.
3. Confirme que o arquivo está disponível no caminho:

```text
/content/flights.csv
```

4. Execute as células na ordem.
5. Analise os gráficos, métricas e resultados do modelo.

Depois que o projeto estiver publicado no GitHub, você poderá abrir o notebook no Colab usando um link neste formato:

```text
https://colab.research.google.com/github/imarques-codes/flight-delay-prediction-analysis/blob/main/notebooks/flight_delay_prediction_analysis.ipynb
```

---

## Como executar localmente

Clone o repositório:

```bash
git clone https://github.com/imarques-codes/flight-delay-prediction-analysis.git
```

Acesse a pasta do projeto:

```bash
cd flight-delay-prediction-analysis
```

Crie um ambiente virtual:

```bash
python -m venv .venv
```

Ative o ambiente virtual:

```bash
# Windows
.venv\Scripts\activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Abra o notebook:

```bash
jupyter notebook notebooks/flight_delay_prediction_analysis.ipynb
```

---

## Base de dados

O projeto utiliza o arquivo:

```text
flights.csv
```

A base contém informações de voos, como:

- ID do voo;
- Companhia aérea;
- Tipo de aeronave;
- Tipo de voo;
- Origem;
- Horário de chegada;
- Horário de saída;
- Dia;
- Ano;
- Indicador de feriado;
- Atraso em minutos.

Caso a base não esteja disponível no repositório, coloque o arquivo dentro da pasta `data/` e ajuste o caminho no notebook.

---

## Resultados do projeto

O modelo final otimizado apresentou os seguintes resultados:

```text
RMSE: 13.2308
MAE: 10.6296
R²: 0.6754
```

Esses resultados indicam que o modelo conseguiu explicar aproximadamente 67% da variação dos atrasos nos dados de teste, superando o modelo baseline.

---

## Principais aprendizados

Este projeto fortalece conhecimentos práticos em:

- Análise exploratória de dados;
- Visualização de dados com Matplotlib e Seaborn;
- Engenharia de atributos;
- Encoding de variáveis categóricas;
- Treinamento de modelos de regressão;
- Criação de modelo baseline;
- Avaliação com RMSE, MAE e R²;
- Validação cruzada;
- Seleção de features;
- Otimização de hiperparâmetros com GridSearchCV;
- Salvamento de modelo com Pickle;
- Organização de projeto de Machine Learning para portfólio.

---

## Possíveis melhorias futuras

- Adicionar uma conclusão executiva ao notebook;
- Criar gráficos finais exportados na pasta `img/`;
- Criar uma versão do modelo para deploy;
- Testar outros algoritmos de regressão;
- Comparar modelos como XGBoost, LightGBM e Linear Regression;
- Criar um dashboard interativo com Streamlit;
- Criar uma API para consumir o modelo salvo;
- Adicionar testes para validação dos dados de entrada.

---

## Autor

**Igor Henrique Marques dos Santos**

- LinkedIn: [linkedin.com/in/igorhmarques](https://www.linkedin.com/in/igorhmarques/)
- GitHub: [github.com/imarques-codes](https://github.com/imarques-codes)
- E-mail: [igorhmsantos@gmail.com](mailto:igorhmsantos@gmail.com)
