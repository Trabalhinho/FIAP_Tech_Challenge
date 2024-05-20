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

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

from sktime.datasets import load_airline

import datetime
from datetime import timedelta
import numpy as np



def sktime_forecast( dataset, horizon, forecaster, confidence = 0.9, frequency = "D" ):
    """Loop over a time series dataframe, train an sktime forecasting model, and visualize the results.

    Args:
        dataset (pd.DataFrame): Input time series DataFrame with datetime index
        horizon (int): Forecast horizon
        forecaster (sktime.forecasting): Configured forecaster
        validation (bool, optional): . Defaults to False.
        confidence (float, optional): Confidence level. Defaults to 0.9.
        frequency (str, optional): . Defaults to "D".
    """

    # Adjust frequency
    forecast_df = dataset.resample( rule = frequency ).mean()

    # Interpolate missing periods (if any)
    forecast_df = forecast_df.interpolate( method = 'time' )

    for col in dataset.columns:
        df = forecast_df[col].dropna()
        forecaster.fit(df)

        last_date = df.index.max()
        fh = ForecastingHorizon(
            pd.date_range( str( last_date ), periods = horizon, freq = frequency ) + timedelta(1),
            is_relative=False,
        )

        y_pred = forecaster.predict(fh)

        df_pred = pd.DataFrame( y_pred )

        return df_pred  


#carregando os dados 
dados = pd.read_csv('https://raw.githubusercontent.com/Trabalhinho/FIAP_Tech_Challenge/main/Bimestre_4/dados_tratados.csv')


############################# Streamlit ############################
st.markdown('<style>div[role="listbox"] ul{background-color: #6e42ad}; </style>', unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; '> Insira a data e veja o possível preço do petróleo</h1>", unsafe_allow_html = True)

st.warning('Insira a data e clique no botão **PREVER** no final da página.')

# Idade
st.write('### Data')
input_data = st.date_input( 'Selecione a data' )

#Predições 
if st.button('Prever'):

    data_maxima = dados['data'].max()

    data_maxima = datetime.datetime.strptime( data_maxima, "%Y-%m-%d" ).date()

    periodo_pred = input_data - data_maxima

    if periodo_pred.days <= 0:
        st.text('Temos o resultado real dessa data:' )
        st.text( dados['preco_petroleo'].where( dados['data'] == input_data ).dropna() )

    else:

        dados_fechamento_final = dados[ [ 'data', 'preco_petroleo' ] ]

        dados_fechamento_final['data'] = pd.to_datetime( dados_fechamento_final['data'] )

        dados_fechamento_final.set_index( 'data', inplace = True )

        # Ajusta a ordem dos dados para as menores datas virem primeiro
        dados_fechamento_final = dados_fechamento_final.iloc[::-1]

        dados_fechamento_final = dados_fechamento_final.resample( rule = 'D' ).mean()
        dados_fechamento_final = dados_fechamento_final.interpolate( method = 'time' )

        dados_fechamento_final = dados_fechamento_final.tail(1200)

        model = joblib.load( 'modelo/prophet.joblib' )
        
        dados_pred_final = sktime_forecast( dataset = dados_fechamento_final, horizon = periodo_pred.days, forecaster = model )
        
        st.text(dados_pred_final.tail(1))
 