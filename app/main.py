import sys
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# Add src to path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional

from models_classes.models_orchestrator import ModelsOrchestrator
from models_classes.mlp_disease_neural_net import device
from data_processing.sinan_mappings import CBO_MAP

# Reverse CBO: code → occupation name (keys as int, float, and string for matching)
REVERSE_CBO = {}
for k, v in CBO_MAP.items():
    name = k.title()
    REVERSE_CBO[v] = name
    REVERSE_CBO[float(v)] = name
    REVERSE_CBO[str(v)] = name

# Reverse maps: SINAN code → Portuguese label
REVERSE_SEX = {'M': 'Masculino', 'F': 'Feminino', 'I': 'Ignorado'}
REVERSE_RACE = {1: 'Branca', 2: 'Preta', 3: 'Amarela', 4: 'Parda', 5: 'Indígena', 9: 'Ignorado'}
REVERSE_UF = {
    11: 'Rondônia', 12: 'Acre', 13: 'Amazonas', 14: 'Roraima', 15: 'Pará',
    16: 'Amapá', 17: 'Tocantins', 21: 'Maranhão', 22: 'Piauí', 23: 'Ceará',
    24: 'Rio Grande do Norte', 25: 'Paraíba', 26: 'Pernambuco', 27: 'Alagoas',
    28: 'Sergipe', 29: 'Bahia', 31: 'Minas Gerais', 32: 'Espírito Santo',
    33: 'Rio de Janeiro', 35: 'São Paulo', 41: 'Paraná', 42: 'Santa Catarina',
    43: 'Rio Grande do Sul', 50: 'Mato Grosso do Sul', 51: 'Mato Grosso',
    52: 'Goiás', 53: 'Distrito Federal',
}
REVERSE_PREGNANCY = {1: '1º Trimestre', 2: '2º Trimestre', 3: '3º Trimestre',
                     4: 'Idade gestacional ignorada', 5: 'Não', 6: 'Não se aplica', 9: 'Ignorado'}
REVERSE_EDUCATION = {0: 'Analfabeto', 1: '1ª a 4ª série incompleta', 2: '4ª série completa',
                     3: '5ª à 8ª série incompleta', 4: 'Ensino fundamental completo',
                     5: 'Ensino médio incompleto', 6: 'Ensino médio completo',
                     7: 'Educação superior incompleta', 8: 'Educação superior completa',
                     9: 'Ignorado', 10: 'Não se aplica'}

app = FastAPI(title='Classificador de Arboviroses')

# Serve static files (HTML, CSS, JS)
app.mount('/static', StaticFiles(directory=Path(__file__).parent / 'static'), name='static')

# Load models on startup
models = {}

MODELS_DIR = Path(__file__).resolve().parent.parent / 'models_saved'

# Symptom/feature labels in Portuguese for display
FEATURE_LABELS_PT = {
    'fever': 'Febre', 'myalgia': 'Mialgia', 'headache': 'Cefaleia',
    'rash': 'Exantema', 'vomiting': 'Vômito', 'nausea': 'Náusea',
    'back_pain': 'Dor nas costas', 'conjunctivitis': 'Conjuntivite',
    'arthritis': 'Artrite', 'joint_pain': 'Artralgia',
    'petechiae': 'Petéquias', 'retro_orbital_pain': 'Dor retro-orbital',
    'diabetes': 'Diabetes', 'hypertension': 'Hipertensão',
    'blood_disorder': 'Doença hematológica', 'liver_disease': 'Hepatopatia',
    'kidney_disease': 'Doença renal', 'peptic_ulcer': 'Doença ácido-péptica',
    'autoimmune_disease': 'Doença autoimune',
    'nosebleed': 'Epistaxe', 'gum_bleeding': 'Gengivorragia',
    'metrorrhagia': 'Metrorragia', 'petechiae_hemorrh': 'Petéquias hemorrágicas',
    'hematuria': 'Hematúria', 'other_bleeding': 'Outros sangramentos',
}

BINARY_FEATURES = [
    'fever', 'myalgia', 'headache', 'rash', 'vomiting', 'nausea',
    'back_pain', 'conjunctivitis', 'arthritis', 'joint_pain',
    'petechiae', 'retro_orbital_pain',
    'diabetes', 'blood_disorder', 'liver_disease', 'kidney_disease',
    'hypertension', 'peptic_ulcer', 'autoimmune_disease',
    'nosebleed', 'gum_bleeding', 'metrorrhagia',
    'petechiae_hemorrh', 'hematuria', 'other_bleeding',
]


@app.on_event('startup')
def load_models():
    for disease in ['dengue', 'chikungunya']:
        disease_dir = MODELS_DIR / disease
        if disease_dir.exists() and (disease_dir / 'artifacts.json').exists():
            orch, mlp, lgbm, xgb, test_examples = ModelsOrchestrator.load_for_inference(disease_dir)
            models[disease] = {
                'orch': orch, 'mlp': mlp, 'lgbm': lgbm, 'xgb': xgb,
                'test_examples': test_examples,
            }
            n = len(test_examples['y']) if test_examples else 0
            print(f'Loaded {disease} models ({n} test examples)')
        else:
            print(f'Warning: {disease} models not found at {disease_dir}')


class PatientInput(BaseModel):
    disease_type: str

    # Demographics
    age: int
    sex: str
    pregnancy_status: str = 'Não se aplica'
    race: str = 'Ignorado'
    education_level: str = 'Ignorado'
    occupation_code: str = ''
    residence_state: str = 'São Paulo'

    # Dates
    symptom_onset_date: str
    notification_date: Optional[str] = None

    # Symptoms
    fever: int = 0
    myalgia: int = 0
    headache: int = 0
    rash: int = 0
    vomiting: int = 0
    nausea: int = 0
    back_pain: int = 0
    conjunctivitis: int = 0
    arthritis: int = 0
    joint_pain: int = 0
    petechiae: int = 0
    retro_orbital_pain: int = 0

    # Comorbidities
    diabetes: int = 0
    blood_disorder: int = 0
    liver_disease: int = 0
    kidney_disease: int = 0
    hypertension: int = 0
    peptic_ulcer: int = 0
    autoimmune_disease: int = 0

    # Hemorrhagic
    nosebleed: int = 0
    gum_bleeding: int = 0
    metrorrhagia: int = 0
    petechiae_hemorrh: int = 0
    hematuria: int = 0
    other_bleeding: int = 0


@app.get('/', response_class=HTMLResponse)
def index():
    html_path = Path(__file__).parent / 'static' / 'index.html'
    return html_path.read_text(encoding='utf-8')


@app.post('/predict')
def predict(patient: PatientInput):
    disease = patient.disease_type
    if disease not in models:
        return {'error': f'Modelos para "{disease}" não encontrados. Rode o notebook para gerar os artefatos.'}

    m = models[disease]
    patient_dict = patient.model_dump(exclude={'disease_type'})

    result = m['orch'].predict(patient_dict, m['mlp'], m['lgbm'], m['xgb'])
    return result


@app.get('/random_example/{disease}')
def random_example(disease: str):
    if disease not in models:
        return {'error': f'Modelos para "{disease}" não disponíveis.'}

    m = models[disease]
    test_ex = m['test_examples']
    if test_ex is None:
        return {'error': 'Exemplos de teste não salvos. Re-rode o notebook com save_artifacts().'}

    # Pick random row
    idx = random.randint(0, len(test_ex['y']) - 1)
    x_cat = test_ex['x_cat'][idx:idx+1].to(device)
    x_num = test_ex['x_num'][idx:idx+1].to(device)
    true_label = int(test_ex['y'][idx])

    orch = m['orch']
    cat_cols = list(orch.categorical_columns)
    num_cols = list(orch.numerical_columns)

    # Run prediction directly on tensors (already encoded)
    dummy = torch.zeros(1, dtype=torch.long).to(device)
    loader = DataLoader(TensorDataset(x_cat, x_num, dummy), batch_size=1, shuffle=False)
    mlp_prob = float(m['mlp'].predict_proba(loader).numpy()[0])
    lgbm_prob = float(m['lgbm'].predict_proba(x_cat, x_num, cat_cols, num_cols)[0])
    xgb_prob = float(m['xgb'].predict_proba(x_cat, x_num, cat_cols, num_cols)[0])

    threshold = 0.4
    avg_prob = (mlp_prob + lgbm_prob + xgb_prob) / 3
    tp = orch.tp_weights
    if tp:
        total = tp['mlp'] + tp['lgbm'] + tp['xgb']
        weighted_prob = (mlp_prob * tp['mlp'] + lgbm_prob * tp['lgbm'] + xgb_prob * tp['xgb']) / total
    else:
        weighted_prob = avg_prob

    # Decode symptoms for display
    num_values = x_num[0].cpu().numpy()
    symptoms_present = []
    symptoms_absent = []
    for i, col in enumerate(num_cols):
        if col in FEATURE_LABELS_PT:
            label = FEATURE_LABELS_PT[col]
            if float(num_values[i]) == 1.0:
                symptoms_present.append(label)
            else:
                symptoms_absent.append(label)

    # Get age
    age = None
    if 'age' in num_cols:
        age = float(num_values[num_cols.index('age')])

    # Decode demographics from categorical tensor
    encoder = orch.data_processor.encoder
    x_cat_vals = test_ex['x_cat'][idx]
    demographics = {}
    DECODE_MAP = {
        'sex': REVERSE_SEX,
        'race': REVERSE_RACE,
        'residence_state': REVERSE_UF,
        'pregnancy_status': REVERSE_PREGNANCY,
        'education_level': REVERSE_EDUCATION,
        'occupation_code': REVERSE_CBO,
    }
    for i, col in enumerate(cat_cols):
        enc_val = int(x_cat_vals[i])
        if enc_val > 0 and enc_val <= len(encoder.categories_[i]):
            raw_val = encoder.categories_[i][enc_val - 1]
            if hasattr(raw_val, 'item'):
                raw_val = raw_val.item()
            if col in DECODE_MAP:
                demographics[col] = DECODE_MAP[col].get(raw_val, str(raw_val))
            else:
                demographics[col] = str(raw_val)
        else:
            demographics[col] = None

    return {
        'true_label': true_label,
        'true_label_text': 'Confirmado' if true_label == 1 else 'Descartado',
        'symptoms_present': symptoms_present,
        'symptoms_absent': symptoms_absent,
        'age': age,
        'sex': demographics.get('sex'),
        'race': demographics.get('race'),
        'state': demographics.get('residence_state'),
        'pregnancy': demographics.get('pregnancy_status'),
        'education': demographics.get('education_level'),
        'occupation': demographics.get('occupation_code'),
        'mlp_probability': mlp_prob,
        'lgbm_probability': lgbm_prob,
        'xgb_probability': xgb_prob,
        'mlp_prediction': int(mlp_prob > threshold),
        'lgbm_prediction': int(lgbm_prob > threshold),
        'xgb_prediction': int(xgb_prob > threshold),
        'average_probability': avg_prob,
        'weighted_probability': weighted_prob,
        'final_prediction': int(weighted_prob > threshold),
        'unanimous': all(p > threshold for p in [mlp_prob, lgbm_prob, xgb_prob]),
    }


@app.get('/available_diseases')
def available_diseases():
    return list(models.keys())
