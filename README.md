
# StormEye - Alertas e Sobrevivencialismo com o Digital

## Resumo da Solução

O StormEye é um aplicativo móvel inovador que transforma informações sobre desastres naturais em segurança prática e acessível. Ele oferece alertas em tempo real sobre eventos climáticos extremos — como tempestades, granizo, tornados, ventos fortes e terremotos — e disponibiliza um acervo de cartilhas especializadas em sobrevivencialismo, acessíveis offline. Desenvolvido para ser intuitivo e eficiente, o aplicativo é uma ferramenta essencial para moradores de áreas de risco, aventureiros e qualquer pessoa que deseje estar preparada para emergências.

**Justificativa do uso de Machine Learning:**  
Para aprimorar a eficácia dos alertas, o StormEye integra técnicas de machine learning capazes de processar grandes volumes de dados meteorológicos em tempo real, identificar padrões e antecipar eventos catastróficos com maior precisão. Isso possibilita notificações antecipadas e personalizadas, aumentando a chance de salvar vidas e reduzir danos.

---

## ✅ Funcionalidades

- Carregamento e limpeza de dados CSV com informações de catástrofes  
- Conversão de colunas numéricas com formatação local para float  
- Análise estatística e estrutural dos dados  
- Visualização de correlações entre variáveis numéricas  
- Pré-processamento e codificação de variáveis categóricas  
- Treinamento e avaliação de modelo Random Forest para classificação  
- Visualização de matriz de confusão e relatório detalhado  
- Função para previsão individual de catástrofes com entrada customizada  

---

## 🚀 Como Executar

### Pré-requisitos

- Python 3.7+ instalado  
- Bibliotecas: pandas, numpy, matplotlib, seaborn, scikit-learn  
  (Instale com: `pip install pandas numpy matplotlib seaborn scikit-learn`)  

### Passos para executar

1. Clone ou baixe o projeto com o arquivo `main.py` e o dataset `eventos_cats.csv` na mesma pasta.  
2. Abra o terminal na pasta do projeto.  
3. Execute o script Python:  
```bash
python main.py
```

---

### Atenção

- Durante a execução, serão exibidos dois gráficos em janelas separadas (fora do terminal), referentes à matriz de correlação e matriz de confusão.  
- Aguarde a visualização e feche as janelas para continuar a execução do script.  

---

## Estrutura do Código

- **carregar_dados()** — Carrega o CSV com os dados das catástrofes  
- **tratar_dados()** — Remove valores nulos ou vazios para limpeza  
- **converter_colunas_numericas()** — Converte colunas numéricas com vírgula para float  
- **mostrar_estatisticas()** — Mostra resumo estatístico e info do DataFrame  
- **plotar_correlacao()** — Gera heatmap de correlação entre variáveis numéricas  
- **preparar_dados_para_modelo()** — Codifica dados para modelo ML (label encoding e dummies)  
- **treinar_modelo()** — Treina Random Forest com hiperparâmetros ajustáveis  
- **avaliar_modelo()** — Avalia modelo, imprime métricas e plota matriz de confusão  
- **prever_catastrofe()** — Realiza previsão para amostra customizada  
- **main()** — Fluxo principal chamando todas as funções em sequência  

---

## Exemplo de Previsão no Código

```python
resultado = prever_catastrofe(
    model, le, X_train.columns,
    nivel_gravidade=5,
    localizacao='cidade_x',
    temperatura_media=10.5,
    altitude=900,
    tipo_de_regiao='rural',
    velocidade_do_vento=80,
    umidade=80
)
print("Catástrofe prevista:", resultado)
```
---

© 2025 Catástrofes ML Project
