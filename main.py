import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from mtranslate import translate

# Garante o download do dicionário do VADER
nltk.download('vader_lexicon', quiet=True)

# Inicializa o analisador de sentimento
sia = SentimentIntensityAnalyzer()

# Configuração da interface do usuário com Streamlit
st.title("Análise de Sentimento Simples")
st.write("Digite um texto em português para analisar o sentimento predominante.")

# Campo para o usuário digitar o texto
texto_usuario = st.text_area("Seu texto aqui:", "Eu adorei este produto, ele é simplesmente fantástico!")

# Botão que engatilha a análise
if st.button("Analisar Sentimento"):
    
    # Traduz o texto do português para o inglês (VADER funciona melhor em inglês)
    texto_traduzido = translate(texto_usuario, "en", "pt")
    
    # Calcula as pontuações de sentimento
    scores = sia.polarity_scores(texto_traduzido)
    compound = scores['compound']
    
    # Define a etiqueta (label) com base na pontuação compound
    if compound >= 0.05:
        resultado = "Positivo"
        cor = "green"
    elif compound <= -0.05:
        resultado = "Negativo"
        cor = "red"
    else:
        resultado = "Neutro"
        cor = "gray"
        
    # Exibe os resultados na tela
    st.markdown(f"### Resultado: :{cor}[{resultado}]")
    st.write(f"**Pontuação Compound:** {compound:.4f}")