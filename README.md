# Classificação de Arboviroses com Machine Learning

Projeto de classificação de casos de **Dengue** e **Chikungunya** usando dados reais do SINAN (Sistema de Informação de Agravos de Notificação do Ministério da Saúde). O objetivo é prever se um caso notificado é confirmado ou descartado com base em dados clínicos e epidemiológicos auto-reportáveis — simulando um triagem online.

## O que esse projeto faz

A partir de registros de notificação de doenças (2017–2019), o pipeline processa os dados, treina três modelos de forma complementar e combina as predições em um ensemble ponderado:

- **ArbovirosesMLP** — rede neural tabular com embeddings para variáveis categóricas
- **LightGBM** — gradient boosting rápido e complementar ao MLP
- **XGBoost** — segundo modelo de boosting para maior robustez do ensemble
- **Ensemble ponderado** — média das probabilidades dos três modelos, ponderada pelo recall de cada um

**Classificação binária:** `1` = caso confirmado, `0` = caso descartado

**Resultados (Dengue, threshold=0.4):** Recall ~0.95 | Precisão ~0.84

## Estrutura

```
health_index_project/
├── data/                          # Arquivos CSV do SINAN (não versionados por tamanho)
│   ├── DENGBR17.csv / DENGBR18.csv / DENGBR19.csv
│   └── CHIKBR17.csv / CHIKBR18.csv / CHIKBR19.csv
│
├── models_saved/                  # Checkpoints dos modelos treinados (.pth)
│
└── src/
    ├── data_processing/
    │   └── disease_dataset_process.py   # Classe DataProcessor — limpeza e feature engineering
    ├── models_classes/
    │   ├── mlp_disease_neural_net.py    # Classe ArbovirosesMLP (PyTorch)
    │   ├── lgbm_classifier.py           # Classe GradientBoostingDiseaseClassifier (LGBM + XGB)
    │   └── models_orchestrator.py       # Classe ModelsOrchestrator — orquestra tudo
    └── notebooks/
        ├── dengue_rede_neural_artificial_clean.ipynb  # Notebook principal
        └── test_orchestrator.ipynb                    # Teste de integração do orquestrador
```

## Modelos

### ArbovirosesMLP
Rede neural para dados tabulares com embeddings para variáveis categóricas:
- Embeddings + BatchNorm nas entradas numéricas
- 4 camadas ocultas: 1024 → 512 → 256 → 128 neurônios (LeakyReLU + BatchNorm + Dropout 0.2)
- Treinado com AdamW, BCEWithLogitsLoss com `pos_weight` para desbalanceamento de classes
- Early stopping (patience=6), scheduler ReduceLROnPlateau
- Métodos: `evaluate()` (varredura de thresholds), `plot_feature_importance()` (permutação)

### GradientBoostingDiseaseClassifier
Suporta dois backends via parâmetro `model`:
- `'lgbm'` — LightGBM, 2000 estimadores, suporte a GPU
- `'xgb'` — XGBoost, 2000 estimadores
- Métodos: `evaluate()` (varredura de thresholds), `plot_feature_importance()`

### ModelsOrchestrator
Orquestra o pipeline de ponta a ponta: carrega e processa os dados, treina os três modelos, calcula pesos por recall e gera o `confirmation_df` com as predições individuais e ensemble.

```python
from src.models_classes.models_orchestrator import ModelsOrchestrator

orchestrator = ModelsOrchestrator(type_disease='dengue')
x_train_cat, x_test_cat, x_train_num, x_test_num, y_train, y_test, emb_sizes = orchestrator.prepare_data()

mlp  = orchestrator.train_mlp(emb_sizes, save_path='models_saved/best_mlp.pth')
lgbm = orchestrator.train_lgbm(x_train_cat, x_train_num, y_train)
xgb  = orchestrator.train_xgb(x_train_cat, x_train_num, y_train)

confirmation_df = orchestrator.evaluate_combined(
    mlp_model=mlp, lgbm_model=lgbm, xgb_model=xgb,
    x_test_cat=x_test_cat, x_test_num=x_test_num
)
```

O `confirmation_df` retornado contém, para cada amostra do teste:

| coluna | descrição |
|---|---|
| `actual` | label real |
| `mlp_pred` / `lgbm_pred` / `xgb_pred` | predição binária de cada modelo |
| `mlp_confidence` / `lgbm_confidence` / `xgb_confidence` | probabilidade de cada modelo |
| `unanimous` | `True` quando os três predizem 1 |
| `weighted_confidence` | média ponderada pelo recall de cada modelo |
| `weighted_positive` | `weighted_confidence > threshold` |

## Dados e Features

**Fonte:** [DataSUS / SINAN](https://datasus.saude.gov.br/) — notificações epidemiológicas do Brasil, 2017–2019

As features foram selecionadas para simular um questionário de triagem online — apenas informações auto-reportáveis pelo paciente:

- **Demográficos:** idade, sexo, raça, gestação, escolaridade, ocupação, estado de residência
- **Temporal:** data de início dos sintomas (→ mês, dia, semana epidemiológica, dias até notificação)
- **Sintomas (12):** febre, mialgia, cefaleia, exantema, vômito, náusea, dor nas costas, conjuntivite, artrite, artralgia, petéquias, dor retroorbital
- **Comorbidades (7):** diabetes, doença hematológica, hepatopatia, doença renal, hipertensão, úlcera péptica, doença autoimune
- **Manifestações hemorrágicas (6):** epistaxe, sangramento gengival, metrorragia, petéquias, hematúria, outros sangramentos
- **Engineered:** contagem de sintomas, comorbidades e manifestações hemorrágicas; 66 interações binárias entre pares de sintomas

Campos com vazamento de dados (hospitalizações, desfechos, sinais de alarme, exames confirmatórios) foram removidos intencionalmente.

## Avaliação

Varredura de thresholds de 0.30 a 0.60 reportando Acurácia, Precisão, Recall e F1 para cada modelo individualmente e para o ensemble. Importância de features via permutação (MLP) e F-score (LGBM/XGB).

## Dependências

- Python 3.x
- `torch`
- `lightgbm`
- `xgboost`
- `scikit-learn`
- `pandas`, `numpy`
- `matplotlib`, `seaborn`

## Datasets

Os arquivos CSV não estão versionados por conta do tamanho (~1 GB no total). Baixe os dados do SINAN para Dengue e Chikungunya (2017–2019) no [DataSUS](https://datasus.saude.gov.br/) e coloque na pasta `data/`.
