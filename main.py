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
import requests  # Adicionado para fazer chamadas HTTP diretas, porém não será usado na consulta do GDACS com a biblioteca

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
            dummy_col = key + '_' + str(value)
            if dummy_col in df_novo.columns:
                df_novo.at[0, dummy_col] = 1
            else:
                print(f"Aviso: coluna {key} ou dummy '{dummy_col}' não está no conjunto de treino e será ignorada.")

    df_novo = df_novo.fillna(0)
    y_pred = model.predict(df_novo)
    return le.inverse_transform(y_pred)[0]


"""**Função para consultar o GDACS**  
Consulta os últimos eventos do GDACS utilizando a biblioteca `gdacs-api` e o método `latest_events()`.  
Permite filtrar por tipo de evento e quantidade (limit) dos eventos retornados.  
Retorna os dados da API no formato JSON.
"""
def consultar_gdacs(limit=10, event_type=None):
    from gdacs.api import GDACSAPIReader  # Importa o cliente da API GDACS
    client = GDACSAPIReader()
    try:
        # Obtém os eventos mais recentes usando o método latest_events()
        events = client.latest_events(limit=limit, event_type=event_type)
        return events
    except Exception as e:
        print(f"Erro ao consultar GDACS: {e}")
        return None


"""**Execução Principal**  
Executa o fluxo completo desde carregamento, tratamento, análise, modelagem, avaliação e previsão de exemplo.
"""
def main():
    df = None
    X_train = X_test = y_train = y_test = None
    model = None
    le = None

    while True:
        print("\n=== Menu StormEye ===")
        print("1. Executar Machine Learning")
        print("2. Ler gráfico de correlação")
        print("3. Ler gráfico de matriz de confusão")
        print("4. Consultar GDACS")
        print("5. Sair")
        escolha = input("Escolha uma opção: ").strip()

        if escolha == '1':
            # Carregar dados
            df = carregar_dados()
            print("Dimensões do DataFrame:", df.shape)
            print("Colunas disponíveis:", df.columns.tolist())
            print(df.head())

            # Tratar e converter
            df = tratar_dados(df)
            df = converter_colunas_numericas(df)
            mostrar_estatisticas(df)

            # Preparar dados
            X, y, le = preparar_dados_para_modelo(df)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Treinar
            model = treinar_modelo(X_train, y_train)
            print("Modelo treinado com sucesso.")

            # Previsão interativa
            print("\n-- Previsão de Catástrofe --")
            entradas = {}
            entradas['nivel_gravidade'] = float(input("Nível de gravidade (número): "))
            entradas['localizacao'] = input("Localização (string): ")
            entradas['temperatura_media'] = float(input("Temperatura média (float): "))
            entradas['altitude'] = float(input("Altitude (float): "))
            entradas['tipo_de_regiao'] = input("Tipo de região ('rural'/'urbano'): ")
            entradas['velocidade_do_vento'] = float(input("Velocidade do vento (float): "))
            entradas['umidade'] = float(input("Umidade (float): "))

            resultado = prever_catastrofe(model, le, X_train.columns, **entradas)
            print("\nCatástrofe prevista:", resultado)

        elif escolha == '2':
            if df is None:
                print("Execute primeiro a opção 1 para carregar e preparar os dados.")
            else:
                # Gráfico de correlação
                plotar_correlacao(df)

        elif escolha == '3':
            if model is None or X_test is None or y_test is None:
                print("Execute primeiro a opção 1 para treinar o modelo.")
            else:
                # Gráfico de matriz de confusão
                avaliar_modelo(model, X_test, y_test, le)

        elif escolha == '4':
            # Consultar GDACS (exemplo: últimos 10 eventos de todos os tipos)
            dados_gdacs = consultar_gdacs(limit=10)
            if dados_gdacs is not None:
                print("\n=== Últimos eventos GDACS (JSON Bruto) ===")
                print(dados_gdacs)
            else:
                print("Nenhum dado retornado pela consulta GDACS.")

        elif escolha == '5':
            print("Saindo...")
            break

        else:
            print("Opção inválida. Tente novamente.")


if __name__ == "__main__":
    main()
