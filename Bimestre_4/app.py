#Importação das bibliotecas
import streamlit as st 
import pandas as pd
import joblib
from joblib import load

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
    model = joblib.load('modelo/prophet.joblib')
    final_pred = model.predict(input_idade)
    print(final_pred)
 