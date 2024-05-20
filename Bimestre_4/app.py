#Importação das bibliotecas
import streamlit as st 
import pandas as pd
import joblib
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


from sktime.forecasting.fbprophet import Prophet
from statsforecast import StatsForecast
from statsforecast.models import Naive

from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.utils.plotting import plot_series

from statsforecast import StatsForecast
from statsforecast.models import Naive

#carregando os dados 
#dados = pd.read_csv('https://raw.githubusercontent.com/alura-tech/alura-tech-pos-data-science-credit-scoring-streamlit/main/df_clean.csv')


############################# Streamlit ############################
st.markdown('<style>div[role="listbox"] ul{background-color: #6e42ad}; </style>', unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; '> Insira a data e veja o possível preço do petróleo</h1>", unsafe_allow_html = True)

st.warning('Insira a data e clique no botão **ENVIAR** no final da página.')

# Idade
st.write('### Idade')
input_idade = st.date_input( 'Selecione a data' )


#Predições 
if st.button('Prever'):
    model = joblib.load( 'modelo/prophet.joblib' )
    print("cheguei aqui")
    final_pred = model.predict(input_idade) 
    print("mais um")
    print(final_pred)
 