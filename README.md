# Classificação de Arboviroses com Machine Learning

Projeto de classificação de casos de **Dengue** e **Chikungunya** usando dados reais do SINAN (Sistema de Informação de Agravos de Notificação do Ministério da Saúde). O objetivo é prever se um caso notificado é confirmado ou descartado com base em dados clínicos e epidemiológicos auto-reportáveis — simulando uma triagem online.

## O que esse projeto faz

A partir de registros de notificação de doenças (2017–2019), o pipeline processa os dados, treina três modelos de forma complementar e combina as predições em um ensemble ponderado:

- **ArbovirosesMLP** — rede neural tabular com embeddings para variáveis categóricas
- **LightGBM** — gradient boosting rápido e complementar ao MLP
- **XGBoost** — segundo modelo de boosting para maior robustez do ensemble
- **Ensemble ponderado** — média das probabilidades dos três modelos, ponderada pelo recall de cada um

**Classificação binária:** `1` = caso confirmado, `0` = caso descartado

O modelo foi calibrado com threshold de **0.4** (em vez do padrão 0.5) para priorizar recall — a lógica é minimizar falsos negativos, ou seja, evitar que um caso real de dengue/chikungunya passe sem ser detectado. Isso resulta em alguns falsos positivos a mais, mas na prática clínica é preferível alertar desnecessariamente do que deixar passar um caso verdadeiro.

## Dados e Features

**Fonte:** [DataSUS / SINAN](https://datasus.saude.gov.br/) — notificações epidemiológicas reais do Brasil, 2017–2019

As features foram selecionadas para simular um questionário de triagem online — apenas informações auto-reportáveis pelo paciente:

- **Demográficos:** idade, sexo, raça, gestação, escolaridade, ocupação (CBO), estado de residência
- **Temporal:** data de início dos sintomas (→ mês, dia, semana epidemiológica YYYYWW, dias até notificação)
- **Sintomas (12):** febre, mialgia, cefaleia, exantema, vômito, náusea, dor nas costas, conjuntivite, artrite, artralgia, petéquias, dor retroorbital
- **Comorbidades (7):** diabetes, doença hematológica, hepatopatia, doença renal, hipertensão, úlcera péptica, doença autoimune
- **Manifestações hemorrágicas (6):** epistaxe, sangramento gengival, metrorragia, petéquias, hematúria, outros sangramentos
- **Engineered:** contagem de sintomas, comorbidades e manifestações hemorrágicas; 66 interações binárias entre pares de sintomas; remoção automática de colunas com >99% de dominância

Campos com vazamento de dados (hospitalizações, desfechos, sinais de alarme, exames confirmatórios) foram removidos intencionalmente. Valores binários do SINAN (1=Sim, 2=Não, 9=Ignorado) são convertidos para 1/0.

## Pipeline de Machine Learning

### Pré-processamento (`DataProcessor`)

1. **Carga e renomeação** — carrega CSVs do SINAN, renomeia 170+ colunas para nomes descritivos em inglês
2. **Features temporais** — extrai mês, dia, semana epidemiológica e calcula dias entre início dos sintomas e notificação (clipped 0–90 dias)
3. **Encoding categórico** — `OrdinalEncoder` com shift +1 para reservar 0 como token de desconhecido (compatível com `nn.Embedding`)
4. **Feature engineering** — contagens agregadas + 66 features de interação entre pares de sintomas
5. **Filtragem de baixa variância** — remove colunas com >99% do mesmo valor

### Modelos

#### ArbovirosesMLP
Rede neural para dados tabulares com embeddings para variáveis categóricas:
- Embeddings aprendidos para cada variável categórica (ex: ocupação CBO → 50 dims, estado → 15 dims)
- BatchNorm nas entradas numéricas
- 4 camadas ocultas: 1024 → 512 → 256 → 128 neurônios (LeakyReLU + BatchNorm + Dropout)
- `BCEWithLogitsLoss` com `pos_weight` para desbalanceamento de classes
- AdamW (lr=1e-4, weight_decay=1e-4), ReduceLROnPlateau (patience=3), early stopping (patience=6)

#### GradientBoostingDiseaseClassifier
Wrapper unificado com suporte a dois backends:
- `'lgbm'` — LightGBM, 2000 estimadores, suporte a GPU
- `'xgb'` — XGBoost, 2000 estimadores
- Ambos com hiperparâmetros otimizados via Optuna (maximizando recall)

#### ModelsOrchestrator
Orquestra o pipeline de ponta a ponta: carrega e processa os dados, treina os três modelos, calcula pesos de TP-rate (recall) por modelo, e gera predições combinadas.

### Ensemble ponderado por recall

O peso de cada modelo no ensemble é proporcional ao seu **recall** (taxa de verdadeiros positivos) calculado no conjunto de teste:

```
weighted_prob = (mlp_prob × recall_mlp + lgbm_prob × recall_lgbm + xgb_prob × recall_xgb) / (recall_mlp + recall_lgbm + recall_xgb)
```

Isso dá mais influência aos modelos que melhor detectam casos positivos. Métricas de recall por modelo:

| Modelo | Dengue | Chikungunya |
|--------|--------|-------------|
| MLP | 85% | 88% |
| LightGBM | 95% | 96% |
| XGBoost | 91% | 96% |

### Avaliação

Varredura de thresholds de 0.30 a 0.60 reportando Acurácia, Precisão, Recall e F1 para cada modelo individualmente e para o ensemble. Importância de features via permutação (MLP) e F-score (LGBM/XGB).

### Inferência sem dados de treino

O método `save_artifacts()` exporta tudo necessário para rodar predições sem os CSVs originais:
- Modelos: `mlp.pth`, `lgbm_model.joblib`, `xgb_model.joblib`
- Encoder categórico: `encoder.joblib`
- Metadados: `artifacts.json` (colunas, medianas, embedding sizes, pesos de recall)
- Exemplos de teste: `test_examples.npz` (200 amostras para demonstração)

O `load_for_inference()` reconstrói o orchestrator e modelos a partir desses artefatos, sem precisar carregar milhões de linhas de CSV.

## Web App

Uma interface web construída com **FastAPI** permite usar os modelos treinados de forma interativa:

- **Formulário de triagem** — o usuário preenche dados demográficos, sintomas e comorbidades, e recebe a predição dos três modelos com probabilidades individuais e resultado combinado
- **Demo com dados reais** — botão que seleciona um paciente aleatório do conjunto de teste (dados que os modelos nunca viram) e compara a predição com o diagnóstico real
- **Autocomplete de ocupação** — busca fuzzy nos 2.426 códigos CBO

A interface web foi desenvolvida com auxílio de IA.

## Estrutura

```
health_index_project/
├── data/                          # Arquivos CSV do SINAN (não versionados)
│   ├── DENGBR17.csv / DENGBR18.csv / DENGBR19.csv
│   └── CHIKBR17.csv / CHIKBR18.csv / CHIKBR19.csv
│
├── models_saved/                  # Artefatos dos modelos treinados
│   ├── dengue/                    # mlp.pth, lgbm/xgb.joblib, encoder.joblib, artifacts.json
│   └── chikungunya/
│
├── app/                           # Web app (FastAPI)
│   ├── main.py                    # Backend — endpoints de predição e demo
│   └── static/                    # Frontend — HTML, CSS, JS
│
└── src/
    ├── data_processing/
    │   ├── disease_dataset_process.py   # DataProcessor — limpeza e feature engineering
    │   ├── sinan_mappings.py            # Mapeamentos SINAN (sexo, raça, UF, escolaridade, etc.)
    │   └── cbo_map.py                   # 2.426 códigos CBO de ocupação
    ├── models_classes/
    │   ├── mlp_disease_neural_net.py    # ArbovirosesMLP (PyTorch)
    │   ├── lgbm_classifier.py           # GradientBoostingDiseaseClassifier (LGBM + XGB)
    │   └── models_orchestrator.py       # ModelsOrchestrator — orquestra pipeline + save/load
    └── notebooks/
        └── test_orchestrator.ipynb      # Notebook de treino e avaliação
```

## Como usar

### Treinar modelos (notebook)
```python
from src.models_classes.models_orchestrator import ModelsOrchestrator

orchestrator = ModelsOrchestrator(type_disease='dengue')
x_train_cat, x_test_cat, x_train_num, x_test_num, y_train, y_test, emb_sizes = orchestrator.prepare_data()

mlp  = orchestrator.train_mlp(emb_sizes, save_path='models_saved/dengue/mlp.pth')
lgbm = orchestrator.train_lgbm(x_train_cat=x_train_cat, x_train_num=x_train_num, y_train=y_train)
xgb  = orchestrator.train_xgb(x_train_cat=x_train_cat, x_train_num=x_train_num, y_train=y_train)

confirmation_df = orchestrator.evaluate_combined(
    mlp_model=mlp, lgbm_model=lgbm, xgb_model=xgb,
    x_test_cat=x_test_cat, x_test_num=x_test_num
)

orchestrator.save_artifacts('models_saved/dengue/', mlp, lgbm, xgb, emb_sizes,
                            x_test_cat=x_test_cat, x_test_num=x_test_num, y_test=y_test)
```

### Rodar o web app
```bash
pip install fastapi uvicorn
uvicorn app.main:app --reload
# Abrir http://localhost:8000
```

## Dependências

- Python 3.x
- `torch`, `lightgbm`, `xgboost`
- `scikit-learn`, `pandas`, `numpy`
- `joblib`, `optuna`
- `fastapi`, `uvicorn` (para o web app)
- `matplotlib`, `seaborn` (para notebooks)

## Datasets

Os arquivos CSV não estão versionados por conta do tamanho (~1 GB no total). Baixe os dados do SINAN para Dengue e Chikungunya (2017–2019) no [DataSUS](https://datasus.saude.gov.br/) e coloque na pasta `data/`.
