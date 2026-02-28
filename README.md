# Classificação de Arboviroses com Machine Learning

Projeto de classificação de casos de **Dengue** e **Chikungunya** usando dados reais do SINAN (Sistema de Informação de Agravos de Notificação do Ministério da Saúde). O objetivo é prever se um caso notificado é confirmado ou descartado com base nos dados clínicos e epidemiológicos do paciente.

## O que esse projeto faz

A partir de registros de notificação de doenças (2017–2019), o pipeline processa os dados, treina dois modelos de forma complementar e combina as predições em um ensemble:

- **MLP (Rede Neural)** — captura padrões não-lineares complexos via embeddings e camadas profundas
- **LightGBM** — gradient boosting rápido, bom ponto de comparação e complemento ao MLP
- **Ensemble** — média simples das probabilidades dos dois modelos

**Classificação binária:** `1` = caso confirmado, `0` = caso descartado

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
    │   ├── lgbm_classifier.py           # Classe LGBMDiseaseClassifier
    │   └── models_orchestrator.py       # Classe ModelsOrchestrator — orquestra tudo
    └── notebooks/
        ├── dengue_rede_neural_artificial.ipynb        # Notebook original com exploração
        ├── dengue_rede_neural_artificial_clean.ipynb  # Versão refatorada e limpa
        └── test_orchestrator.ipynb                    # Teste de integração do orquestrador
```

## Modelos

### ArbovirosesMLP
Rede neural para dados tabulares com embeddings para variáveis categóricas:
- Embeddings + normalização batch nas entradas
- 4 camadas ocultas: 2048 → 1024 → 512 → 256 neurônios (LeakyReLU + Dropout 0.2)
- Treinado com AdamW, BCEWithLogitsLoss com peso de classe, early stopping (patience=8)

### LGBMDiseaseClassifier
- 2000 estimadores, learning rate 0.03
- Suporte a GPU via LightGBM (fallback para CPU)

### ModelsOrchestrator
Classe que orquestra o pipeline de ponta a ponta: carrega os dados, treina os dois modelos, gera predições combinadas e avalia o ensemble.

```python
from src.models_classes.models_orchestrator import ModelsOrchestrator

orchestrator = ModelsOrchestrator(type_disease="dengue")
x_train_cat, x_test_cat, x_train_num, x_test_num, y_train, y_test, emb_sizes = orchestrator.prepare_data()

mlp   = orchestrator.train_mlp(emb_sizes, save_path="models_saved/best_mlp.pth")
lgbm  = orchestrator.train_lgbm(x_train_cat, x_train_num, y_train)

orchestrator.evaluate_combined(mlp, lgbm, x_test_cat, x_test_num)
```

## Dados e Features

**Fonte:** [DataSUS / SINAN](https://datasus.saude.gov.br/) — notificações epidemiológicas do Brasil, 2017–2019

O processamento inclui:
- Dados demográficos do paciente (idade, sexo, raça, gestação)
- Sintomas (febre, mialgia, cefaleia, exantema, artralgia, conjuntivite, etc.)
- Comorbidades (diabetes, hipertensão, doenças renais/hepáticas, etc.)
- Agregações derivadas: contagem de sintomas, comorbidades e manifestações hemorrágicas
- 66 interações entre pares de sintomas (ex: `febre × mialgia`)
- Features de data: mês, dias entre início dos sintomas e notificação

Campos que poderiam causar vazamento de dados (hospitalizações, desfechos, exames confirmatórios) foram removidos intencionalmente.

## Avaliação

Os modelos são avaliados com varredura de thresholds de 0.30 a 0.60, reportando Acurácia, Precisão, Recall e F1. A importância de features é calculada via permutação (MLP) e F-score (LightGBM).

## Dependências

- Python 3.x
- `torch`
- `lightgbm`
- `scikit-learn`
- `pandas`, `numpy`
- `matplotlib`, `seaborn`

## Datasets

Os arquivos CSV não estão versionados por conta do tamanho (~1 GB no total). Baixe os dados do SINAN para Dengue e Chikungunya (2017–2019) no [DataSUS](https://datasus.saude.gov.br/) e coloque na pasta `data/`.
