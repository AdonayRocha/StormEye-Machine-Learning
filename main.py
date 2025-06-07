# main.py
# -*- coding: utf-8 -*-

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Estilo dos gráficos
sns.set(style='whitegrid')

# Ignora avisos desnecessários
warnings.filterwarnings("ignore", category=FutureWarning)

"""**Carregamento Inicial dos Dados**  
Carrega o arquivo CSV 'eventos_cats.csv' que deve estar na mesma pasta que este script ou no caminho relativo definido.  
Esse arquivo contém os dados das catástrofes para análise e modelagem.
"""
def carregar_dados():
    csv_path = os.path.join(os.path.dirname(__file__), "eventos_cats.csv")
    df = pd.read_csv(csv_path, encoding='utf-8-sig', sep=';')
    return df

"""**Tratamento dos Dados**  
Remove linhas com valores nulos ou vazios para garantir qualidade do dataset antes da análise e modelagem.
"""
def tratar_dados(df):
    row_inicio = df.shape[0]
    df_cleaned = df.dropna()
    df_cleaned = df_cleaned[~df_cleaned.isin(['', None])].dropna()
    row_final = df_cleaned.shape[0]
    print(f"Número de registros antes da remoção: {row_inicio}")
    print(f"Número de registros após a remoção: {row_final}")
    print(f"Registros removidos: {row_inicio - row_final}")
    return df_cleaned

"""**Conversão de Colunas Numéricas**  
Algumas colunas numéricas vêm como string com vírgulas, esta etapa converte para float usando ponto decimal.
"""
def converter_colunas_numericas(df):
    for col in ['temperatura_media', 'velocidade_do_vento']:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.replace(',', '.').astype(float)
    return df

"""**Resumo Estatístico e Estrutural dos Dados**  
Mostra informações básicas e estatísticas descritivas para análise inicial e identificação de possíveis anomalias.
"""
def mostrar_estatisticas(df):
    print("\nInformações Gerais")
    print(df.info())
    print("\nInformações estatísticas")
    print(df.describe())

"""**Análise de Correlação**  
Gera um heatmap para visualizar correlações lineares entre variáveis numéricas importantes do dataset.
"""
def plotar_correlacao(df):
    cols_numericas = ['nivel_gravidade', 'temperatura_media', 'altitude', 'velocidade_do_vento', 'umidade']
    corr = df[cols_numericas].corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Matriz de Correlação')
    plt.tight_layout()
    plt.show()

"""**Pré-processamento para Modelagem**  
Transforma variáveis categóricas em numéricas e codifica o alvo (target) para preparar os dados para treinamento do modelo.
"""
def preparar_dados_para_modelo(df, target='nome_catastrofe'):
    for col in df.columns:
        if df[col].dtype == object and col != target:
            try:
                df[col] = df[col].str.replace(',', '.').astype(float)
            except:
                pass

    le = LabelEncoder()
    df['target_encoded'] = le.fit_transform(df[target])

    X = df.drop([target, 'target_encoded'], axis=1)
    y = df['target_encoded']

    # Cria variáveis dummy para colunas categóricas restantes
    X = pd.get_dummies(X)

    return X, y, le

"""**Treinamento do Modelo Random Forest**  
Treina o classificador Random Forest com os dados de treino e hiperparâmetros definidos.
"""
def treinar_modelo(X_train, y_train, n_estimators=100, max_depth=None, min_samples_split=2, random_state=42):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model

"""**Avaliação do Modelo**  
Avalia o modelo treinado usando acurácia, relatório de classificação e matriz de confusão com visualização gráfica.
"""
def avaliar_modelo(model, X_test, y_test, le):
    y_pred = model.predict(X_test)
    y_test_names = le.inverse_transform(y_test)
    y_pred_names = le.inverse_transform(y_pred)

    print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
    print("\nRelatório de Classificação:\n", classification_report(y_test_names, y_pred_names))

    cm = confusion_matrix(y_test_names, y_pred_names, labels=le.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Matriz de Confusão')
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.show()

    df_result = pd.DataFrame({
        'Real': y_test_names,
        'Previsto': y_pred_names
    })
    print("\n10 primeiras amostras - Real x Previsto:")
    print(df_result.head(10))

"""**Função para Previsão de Catástrofe**  
Permite realizar previsões individuais fornecendo os parâmetros das variáveis de entrada.  
A função cria um DataFrame alinhado com as colunas de treino, tratando variáveis dummy corretamente.
"""
def prever_catastrofe(model, le, X_train_columns, **kwargs):
    import pandas as pd
    df_novo = pd.DataFrame(columns=X_train_columns)

    for key, value in kwargs.items():
        if key in df_novo.columns:
            df_novo.at[0, key] = value
        else:
            # Tenta preencher variáveis dummy no formato coluna_valor
            dummy_col = key + '_' + str(value)
            if dummy_col in df_novo.columns:
                df_novo.at[0, dummy_col] = 1
            else:
                print(f"Aviso: coluna {key} ou dummy '{dummy_col}' não está no conjunto de treino e será ignorada.")

    df_novo = df_novo.fillna(0)

    y_pred = model.predict(df_novo)
    return le.inverse_transform(y_pred)[0]

"""**Execução Principal**  
Executa o fluxo completo desde carregamento, tratamento, análise, modelagem, avaliação e previsão de exemplo.
"""
def main():
    # Carregar dados
    df = carregar_dados()

    # Mostrar informações iniciais
    print("Dimensões do DataFrame:", df.shape)
    print("Colunas disponíveis:", df.columns.tolist())
    print(df.head())

    # Tratar dados
    df = tratar_dados(df)

    # Converter colunas numéricas
    df = converter_colunas_numericas(df)

    # Estatísticas e informações
    mostrar_estatisticas(df)

    # Plotar correlação
    plotar_correlacao(df)

    # Preparar dados para ML
    X, y, le = preparar_dados_para_modelo(df)

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinar modelo
    model = treinar_modelo(X_train, y_train)

    # Avaliar modelo
    avaliar_modelo(model, X_test, y_test, le)

    # Exemplo de previsão
    resultado = prever_catastrofe(
        model, le, X_train.columns,
        nivel_gravidade=5,
        localizacao='cidade_x',
        temperatura_media=10.5,
        altitude=900,
        tipo_de_regiao='rural',  # pode ser "rural" ou "urbano"
        velocidade_do_vento=80,
        umidade=80
    )
    print("\nCatástrofe prevista:", resultado)

if __name__ == "__main__":
    main()
