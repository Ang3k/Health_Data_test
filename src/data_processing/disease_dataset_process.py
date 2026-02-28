import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from itertools import combinations

from lightgbm import LGBMClassifier

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.inspection import permutation_importance

print(f'Python: {sys.executable}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA built with: {torch.version.cuda}')
print(f'CUDA available: {torch.cuda.is_available()}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


COLUMN_RENAME_MAP = {
    # === NOTIFICATION INFO ===
    'TP_NOT': 'notification_type',          # Type of notification (individual, aggregate, etc.)
    'ID_AGRAVO': 'disease_code',            # ICD/SINAN code identifying the disease
    'DT_NOTIFIC': 'notification_date',      # Date the case was reported
    'SEM_NOT': 'notification_epi_week',     # Epidemiological week of the notification
    'NU_ANO': 'notification_year',          # Year the case was reported
    'SG_UF_NOT': 'notif_state',            # State (UF) where the case was notified
    'ID_MUNICIP': 'notif_municipality',     # Municipality where the case was notified
    'ID_REGIONA': 'notif_health_region',    # Health region where the case was notified
    'ID_UNIDADE': 'health_facility',        # Health facility that filed the notification
    'DT_SIN_PRI': 'symptom_onset_date',     # Date patient first showed symptoms
    'SEM_PRI': 'symptom_epi_week',          # Epidemiological week of first symptoms

    # === PATIENT DEMOGRAPHICS ===
    'ANO_NASC': 'birth_year',              # Patient's year of birth
    'NU_IDADE_N': 'age',                   # Patient's age (encoded with unit prefix: days/months/years)
    'CS_SEXO': 'sex',                      # Patient's sex (M=male, F=female, I=ignored)
    'CS_GESTANT': 'pregnancy_status',       # Pregnancy trimester (1st, 2nd, 3rd) or N/A
    'CS_RACA': 'race',                     # Patient's race/ethnicity
    'CS_ESCOL_N': 'education_level',        # Patient's education level
    'ID_OCUPA_N': 'occupation_code',        # Patient's occupation (CBO code)

    # === PATIENT RESIDENCE ===
    'SG_UF': 'residence_state',            # State where the patient lives
    'ID_MN_RESI': 'residence_municipality', # Municipality where the patient lives
    'ID_RG_RESI': 'residence_health_region',# Health region where the patient lives
    'ID_PAIS': 'residence_country',         # Country where the patient lives

    # === SYMPTOMS (1=Yes, 2=No, 9=Unknown) ===
    'FEBRE': 'fever',                       # Fever
    'MIALGIA': 'myalgia',                  # Muscle pain
    'CEFALEIA': 'headache',                # Headache
    'EXANTEMA': 'rash',                    # Skin rash
    'VOMITO': 'vomiting',                  # Vomiting
    'NAUSEA': 'nausea',                    # Nausea
    'DOR_COSTAS': 'back_pain',             # Back pain
    'CONJUNTVIT': 'conjunctivitis',        # Conjunctivitis (eye inflammation)
    'ARTRITE': 'arthritis',                # Joint inflammation
    'ARTRALGIA': 'joint_pain',             # Joint pain
    'PETEQUIA_N': 'petechiae',             # Small red/purple spots on skin (bleeding under skin)
    'LEUCOPENIA': 'leucopenia',            # Low white blood cell count
    'LACO': 'tourniquet_test',             # Tourniquet test (prova do laço) for capillary fragility
    'DOR_RETRO': 'retro_orbital_pain',     # Pain behind the eyes

    # === COMORBIDITIES (1=Yes, 2=No, 9=Unknown) ===
    'DIABETES': 'diabetes',                 # Has diabetes
    'HEMATOLOG': 'blood_disorder',          # Has blood/hematological disease
    'HEPATOPAT': 'liver_disease',           # Has liver disease
    'RENAL': 'kidney_disease',              # Has kidney disease
    'HIPERTENSA': 'hypertension',           # Has hypertension
    'ACIDO_PEPT': 'peptic_ulcer',           # Has peptic acid disease / ulcer
    'AUTO_IMUNE': 'autoimmune_disease',     # Has autoimmune disease

    # === CHIKUNGUNYA LAB TESTS ===
    'DT_CHIK_S1': 'chik_test1_date',       # Date of Chikungunya serological test 1
    'DT_CHIK_S2': 'chik_test2_date',       # Date of Chikungunya serological test 2
    'RES_CHIKS1': 'chik_test1_result',      # Result of Chikungunya test 1
    'RES_CHIKS2': 'chik_test2_result',      # Result of Chikungunya test 2
    'DT_PRNT': 'prnt_date',                # Date of PRNT test (plaque reduction neutralization)
    'RESUL_PRNT': 'prnt_result',            # Result of PRNT test

    # === DENGUE LAB TESTS ===
    'DT_SORO': 'serology_date',            # Date of serological test (IgM)
    'RESUL_SORO': 'serology_result',        # Result of serology (positive, negative, inconclusive)
    'DT_NS1': 'ns1_test_date',             # Date of NS1 antigen test
    'RESUL_NS1': 'ns1_result',             # Result of NS1 test
    'DT_VIRAL': 'viral_isolation_date',     # Date of viral isolation test
    'RESUL_VI_N': 'viral_isolation_result', # Result of viral isolation
    'DT_PCR': 'pcr_date',                  # Date of RT-PCR test
    'RESUL_PCR_': 'pcr_result',            # Result of RT-PCR test
    'SOROTIPO': 'serotype',                # Dengue serotype identified (DENV-1, 2, 3, or 4)
    'HISTOPA_N': 'histopathology',         # Histopathology result
    'IMUNOH_N': 'immunohistochemistry',    # Immunohistochemistry result

    # === HOSPITALIZATION ===
    'HOSPITALIZ': 'hospitalized',           # Whether patient was hospitalized (1=Yes, 2=No)
    'DT_INTERNA': 'hospitalization_date',   # Date of hospitalization
    'UF': 'hospital_state',                # State of the hospital
    'MUNICIPIO': 'hospital_municipality',   # Municipality of the hospital

    # === INFECTION ORIGIN ===
    'TPAUTOCTO': 'autochthonous_case',      # Whether infection was local or imported
    'COUFINF': 'infection_state',           # State where infection likely occurred
    'COPAISINF': 'infection_country',       # Country where infection likely occurred
    'COMUNINF': 'infection_municipality',   # Municipality where infection likely occurred

    # === CLASSIFICATION & OUTCOME ===
    'CLASSI_FIN': 'final_classification',   # Final diagnosis (confirmed, discarded, inconclusive)
    'CRITERIO': 'confirmation_criteria',    # How it was confirmed (lab, clinical, epidemiological)
    'DOENCA_TRA': 'work_related',           # Whether the disease is work-related
    'CLINC_CHIK': 'chik_clinical_form',     # Clinical form of Chikungunya (acute, subacute, chronic)
    'EVOLUCAO': 'case_outcome',             # Patient outcome (cured, died, etc.)
    'DT_OBITO': 'death_date',              # Date of death (if applicable)
    'DT_ENCERRA': 'case_closure_date',      # Date the case was officially closed

    # === ALARM SIGNS (dengue warning signs, 1=Yes, 2=No) ===
    'ALRM_HIPOT': 'alarm_hypotension',     # Postural hypotension (drop in blood pressure)
    'ALRM_PLAQ': 'alarm_low_platelets',    # Platelet count drop
    'ALRM_VOM': 'alarm_persistent_vomit',  # Persistent vomiting
    'ALRM_SANG': 'alarm_bleeding',         # Bleeding from mucous membranes
    'ALRM_HEMAT': 'alarm_hematocrit_rise', # Rising hematocrit
    'ALRM_ABDOM': 'alarm_abdominal_pain',  # Intense abdominal pain
    'ALRM_LETAR': 'alarm_lethargy',        # Lethargy / irritability
    'ALRM_HEPAT': 'alarm_liver_enlarged',  # Enlarged liver (hepatomegaly)
    'ALRM_LIQ': 'alarm_fluid_accumul',     # Fluid accumulation (pleural effusion, ascites)
    'DT_ALRM': 'alarm_signs_date',         # Date alarm signs were observed

    # === SEVERITY SIGNS (severe dengue, 1=Yes, 2=No) ===
    'GRAV_PULSO': 'severe_weak_pulse',      # Weak or absent pulse
    'GRAV_CONV': 'severe_convulsions',      # Convulsions
    'GRAV_ENCH': 'severe_cap_refill',       # Slow capillary refill (>2 sec)
    'GRAV_INSUF': 'severe_resp_distress',   # Respiratory distress
    'GRAV_TAQUI': 'severe_tachycardia',     # Tachycardia (fast heart rate)
    'GRAV_EXTRE': 'severe_cold_extremities',# Cold extremities / cyanosis
    'GRAV_HIPOT': 'severe_hypotension',     # Hypotension / shock
    'GRAV_HEMAT': 'severe_hematemesis',     # Vomiting blood
    'GRAV_MELEN': 'severe_melena',          # Black tarry stool (GI bleeding)
    'GRAV_METRO': 'severe_metrorrhagia',    # Abnormal uterine bleeding
    'GRAV_SANG': 'severe_bleeding',         # Severe bleeding
    'GRAV_AST': 'severe_ast_elevated',      # AST/ALT > 1000 (liver enzymes)
    'GRAV_MIOC': 'severe_myocarditis',      # Myocarditis (heart inflammation)
    'GRAV_CONSC': 'severe_altered_consc',   # Altered consciousness
    'GRAV_ORGAO': 'severe_organ_damage',    # Other organ involvement
    'DT_GRAV': 'severity_signs_date',       # Date severity signs were observed

    # === HEMORRHAGIC MANIFESTATIONS ===
    'MANI_HEMOR': 'hemorrhagic_manifest',   # Had hemorrhagic manifestations (1=Yes, 2=No)
    'EPISTAXE': 'nosebleed',                # Epistaxis (nosebleed)
    'GENGIVO': 'gum_bleeding',              # Gingival bleeding (gums)
    'METRO': 'metrorrhagia',                # Abnormal uterine bleeding
    'PETEQUIAS': 'petechiae_hemorrh',       # Petechiae (hemorrhagic context)
    'HEMATURA': 'hematuria',                # Blood in urine
    'SANGRAM': 'other_bleeding',            # Other bleeding
    'LACO_N': 'tourniquet_test_hemorrh',    # Tourniquet test (hemorrhagic context)
    'PLASMATICO': 'plasma_leakage',         # Evidence of plasma leakage
    'EVIDENCIA': 'hemorrhagic_evidence',    # Evidence of hemorrhagic manifestation
    'PLAQ_MENOR': 'platelets_below_100k',   # Platelet count < 100,000
    'CON_FHD': 'dengue_hemorrhagic_fever',  # Confirmed Dengue Hemorrhagic Fever (DHF)
    'COMPLICA': 'complications',            # Complications present

    # === ADMINISTRATIVE / SYSTEM ===
    'DT_INVEST': 'investigation_date',      # Date the case was investigated
    'DT_DIGITA': 'data_entry_date',         # Date the record was entered into the system
    'TP_SISTEMA': 'system_type',            # Type of information system used
    'NDUPLIC_N': 'duplicate_flag',          # Whether this record is a duplicate
    'CS_FLXRET': 'return_flow_flag',        # Case flow return flag (inter-state data sharing)
    'FLXRECEBI': 'flow_received',           # Flow received flag (inter-state data sharing)
    'MIGRADO_W': 'migrated_from_windows',   # Record migrated from old Windows SINAN system
    'DT_NASC' : 'birth_date'               # Patient's date of birth
}

DROP_COLUMNS = [
    # === ADMINISTRATIVE (no predictive value) ===
    'investigation_date',       # filled during/after investigation
    'duplicate_flag',           # system control field
    'return_flow_flag',         # system control field
    'flow_received',            # system control field
    'system_type',              # system control field
    'notification_type',        # administrative notification type
    'notification_epi_week',    # redundant with notification_month/day derived features

    # === POST-CLASSIFICATION (filled after or because of final_classification) ===
    'confirmation_criteria',    # directly tied to classification (lab, clinical, epidemiological)
    'case_closure_date',        # required when classification is filled
    'case_outcome',             # outcome recorded after classification (cura, obito, etc.)
    'death_date',               # post-outcome
    'work_related',             # enabled only if classification=1, cleared if classification=2
    'chik_clinical_form',       # required only if classification=13 (Chikungunya)

    # === INFECTION ORIGIN (filled only when classification=confirmed, cleared on discard) ===
    'autochthonous_case',       # required only if classification=1
    'infection_state',          # required only if classification=1
    'infection_country',        # required only if classification=1
    'infection_municipality',   # required only if classification=1

    # === ALARM SIGNS (filled only when classification=11 or 12, leaks the label directly) ===
    'alarm_hypotension',
    'alarm_low_platelets',
    'alarm_persistent_vomit',
    'alarm_bleeding',
    'alarm_hematocrit_rise',
    'alarm_abdominal_pain',
    'alarm_lethargy',
    'alarm_liver_enlarged',
    'alarm_fluid_accumul',
    'alarm_signs_date',

    # === SEVERITY SIGNS (filled only when classification=12, leaks the label directly) ===
    'severe_weak_pulse',
    'severe_convulsions',
    'severe_cap_refill',
    'severe_resp_distress',
    'severe_tachycardia',
    'severe_cold_extremities',
    'severe_hypotension',
    'severe_hematemesis',
    'severe_melena',
    'severe_metrorrhagia',
    'severe_bleeding',
    'severe_ast_elevated',
    'severe_myocarditis',
    'severe_altered_consc',
    'severe_organ_damage',
    'severity_signs_date',

    # === DHF / COMPLICATIONS (old classification system, directly informs final_classification) ===
    'dengue_hemorrhagic_fever', # confirmed DHF = classification decision
    'complications',            # dengue with complications = classification decision

    # === CHIKUNGUNYA TESTS (disabled for dengue cases per dictionary, ~97.7% NaN) ===
    'chik_test1_date',          # enabled only for Chikungunya, dataset is 100% Dengue (A90)
    'chik_test2_date',
    'prnt_date',
    'chik_test1_result',
    'chik_test2_result',
    'prnt_result',

    # === HOSPITALIZATION (post-assessment decision, not available at diagnosis time) ===
    'hospitalized',             # decision made after clinical evaluation
    'hospitalization_date',     # only filled if hospitalized
    'hospital_state',           # only filled if hospitalized
    'hospital_municipality',    # only filled if hospitalized

    # === DATE FIELDS (not useful as raw values for ANN, keeping only notification_date, symptom_onset_date, birth_year) ===
    'notification_year',        # redundant with notification_date
    'serology_date',            # lab test date, not useful as raw value
    'ns1_test_date',            # lab test date
    'viral_isolation_date',     # lab test date
    'pcr_date',                 # lab test date

    # === NOT SELF-REPORTABLE (require clinical procedure or lab exam) ===
    'leucopenia',               # blood test
    'tourniquet_test',          # clinical procedure (prova do laço)
    'tourniquet_test_hemorrh',  # clinical procedure (hemorrhagic context)
    'plasma_leakage',           # clinical evaluation
    'platelets_below_100k',     # blood test
    'hemorrhagic_evidence',     # clinical evaluation

    # === GEOGRAPHICAL (not available/useful in a self-reported questionnaire) ===
    'notif_state',
    'notif_municipality',
    'notif_health_region',
    'health_facility',
    'residence_municipality',
    'residence_country',
]

LAB_DROP_COLUMNS = [
    'disease_code',
    'serology_result',          # result of serological test (positive, negative, inconclusive)
    'ns1_result',               # result of NS1 antigen test
    'viral_isolation_result',   # result of viral isolation test
    'pcr_result',               # result of RT-PCR test
    'serotype',                 # dengue serotype identified (DENV-1, 2, 3, or 4)
    'histopathology',           # histopathology result
    'immunohistochemistry',     # immunohistochemistry result
    'hemorrhagic_manifest'
]

CATEGORICAL_COLUMNS = [
    'sex',
    'pregnancy_status',
    'race',
    'education_level',
    'occupation_code',
    'symptom_month',
    'symptom_day',
    'residence_state',
    'symptom_epi_week'
]

BINARY_COLUMNS = [
    # === SINTOMAS ===
    'fever', 'myalgia', 'headache', 'rash', 'vomiting', 'nausea',
    'back_pain', 'conjunctivitis', 'arthritis', 'joint_pain',
    'petechiae', 'retro_orbital_pain',

    # === COMORBIDADES ===
    'diabetes', 'blood_disorder', 'liver_disease', 'kidney_disease',
    'hypertension', 'peptic_ulcer', 'autoimmune_disease',

    # === MANIFESTAÇÕES HEMORRÁGICAS ===
    'nosebleed', 'gum_bleeding', 'metrorrhagia',
    'petechiae_hemorrh', 'hematuria', 'other_bleeding',
]

SYMPTOM_COLS = [
    'fever', 'myalgia', 'headache', 'rash', 'vomiting', 'nausea', 'back_pain', 'conjunctivitis',
    'arthritis', 'joint_pain', 'petechiae', 'retro_orbital_pain',
]

COMORBIDITY_COLS = ['diabetes', 'blood_disorder', 'liver_disease', 'kidney_disease', 'hypertension', 'peptic_ulcer', 'autoimmune_disease']
HEMORRHAGIC_COLS = ['nosebleed', 'gum_bleeding', 'metrorrhagia', 'petechiae_hemorrh', 'hematuria', 'other_bleeding']

DENGUE_MAPPING = {
    5 : 0,   # Discarded
    10 : 1,  # Confirmed
    11 : 1,  # Confirmed and alarming
    12 : 1,  # Confirmed with complications
}

CHIK_MAPPING = {
    5 : 0,   # Discarded
    13 : 1,  # Confirmed Chikungunya
}

DATA_PATHS = {
    'dengue': [
        "C:\\Users\\angej\\Documents\\2_Programação\\health_index_project\\data\\DENGBR17.csv",
        "C:\\Users\\angej\\Documents\\2_Programação\\health_index_project\\data\\DENGBR18.csv",
        "C:\\Users\\angej\\Documents\\2_Programação\\health_index_project\\data\\DENGBR19.csv",
    ],
    'chikungunya': [
        "C:\\Users\\angej\\Documents\\2_Programação\\health_index_project\\data\\CHIKBR17.csv",
        "C:\\Users\\angej\\Documents\\2_Programação\\health_index_project\\data\\CHIKBR18.csv",
        "C:\\Users\\angej\\Documents\\2_Programação\\health_index_project\\data\\CHIKBR19.csv",
    ],
}


class DataProcessor:
    def __init__(self, type_disease):
        self.type_disease = type_disease

    def load_data(self):
        paths = DATA_PATHS[self.type_disease]
        dfs = [pd.read_csv(p, low_memory=False) for p in paths]
        df = pd.concat(dfs, ignore_index=True)

        int_cols = df.select_dtypes("int64").columns
        df[int_cols] = df[int_cols].apply(pd.to_numeric, downcast="integer")

        float_cols = df.select_dtypes("float64").columns
        df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast="float")

        df = df.rename(columns=COLUMN_RENAME_MAP)
        df = df.drop(columns=DROP_COLUMNS, errors='ignore')
        df = df.drop(columns=LAB_DROP_COLUMNS, errors='ignore')

        return df

    def process_date_features(self, df):
        df['notification_date'] = pd.to_datetime(df['notification_date'], errors='coerce')
        df['symptom_onset_date'] = pd.to_datetime(df['symptom_onset_date'], errors='coerce')

        df['symptom_month'] = df['symptom_onset_date'].dt.month
        df['symptom_day'] = df['symptom_onset_date'].dt.day
        df['symptom_month_end'] = df['symptom_onset_date'].dt.is_month_end
        df['symptom_year_end'] = df['symptom_onset_date'].dt.is_year_end

        # Dias entre início dos sintomas e notificação (janela crítica da dengue: 3-6 dias)
        df['days_to_notification'] = (df['notification_date'] - df['symptom_onset_date']).dt.days
        df['days_to_notification'] = df['days_to_notification'].fillna(df['days_to_notification'].median())
        df['days_to_notification'] = df['days_to_notification'].clip(0, 90)

        # Derivando idade a partir da data de nascimento
        if self.type_disease == 'dengue':
            df['birth_date'] = pd.to_datetime(df['birth_date'])
            df['birth_year'] = df['birth_date'].dt.year
            df['age'] = df['birth_year'].apply(lambda x: 2025 - x if pd.notnull(x) else None)
            df = df.drop(columns=['birth_date', 'birth_year', 'notification_date', 'symptom_onset_date'])

        elif self.type_disease == 'chikungunya':
            df['age'] = df['age'] - 4000
            df = df.drop(columns=['notification_date', 'symptom_onset_date'])

        return df

    def process_features(self, df):
        # Convertendo colunas booleanas (1=Yes, 2=No) para 1/0
        bools = df.select_dtypes(include=['bool']).columns
        df[bools] = df[bools].astype(int)

        # This keeps all indices non-negative, which is required by nn.Embedding.
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df[CATEGORICAL_COLUMNS] = oe.fit_transform(df[CATEGORICAL_COLUMNS]) + 1

        # Fill with 0, which is the reserved "unknown" token from the shift above.
        df[CATEGORICAL_COLUMNS] = df[CATEGORICAL_COLUMNS].fillna(0).astype(int)

        # 1=Sim, 2=Não, 9=Ignorado → 1=Sim, 0=Não/Ignorado/NaN
        df[BINARY_COLUMNS] = df[BINARY_COLUMNS].replace({2: 0, 9: 0}).fillna(0).astype(int)

        # age: preencher NaN com mediana
        df['age'] = df['age'].fillna(df['age'].median())
        df['residence_health_region'] = df['residence_health_region'].fillna(df['residence_health_region'].median()).astype(int)

        df['symptom_count']     = df[SYMPTOM_COLS].sum(axis=1)
        df['comorbidity_count'] = df[COMORBIDITY_COLS].sum(axis=1)
        df['hemorrhagic_count'] = df[HEMORRHAGIC_COLS].sum(axis=1)

        interaction_cols = {
            f'{a}_and_{b}': (df[a] * df[b]).astype(int)
            for a, b in combinations(SYMPTOM_COLS, 2)
        }

        df = pd.concat([df, pd.DataFrame(interaction_cols, index=df.index)], axis=1)

        # Remove colunas onde >95% dos valores são iguais (baixa variância)
        dominance_threshold = 0.99

        dominant_ratio = df.drop(columns=['final_classification']).apply(
            lambda col: col.value_counts(normalize=True).iloc[0]
        )
        cols_to_drop_low_variance = dominant_ratio[dominant_ratio >= dominance_threshold].index.tolist()

        # Não dropar colunas categóricas — variância delas é esperada ser concentrada após encoding
        cols_to_drop_low_variance = [c for c in cols_to_drop_low_variance if c not in CATEGORICAL_COLUMNS]

        df = df.drop(columns=cols_to_drop_low_variance)

        print(f'Colunas removidas (>{dominance_threshold*100:.0f}% mesmo valor): {len(cols_to_drop_low_variance)}')
        print(cols_to_drop_low_variance)

        return df

    def process_target(self, df):
        if self.type_disease == 'dengue':
            df = df[df['final_classification'].isin([5, 10, 11, 12])]
        elif self.type_disease == 'chikungunya':
            df = df[df['final_classification'].isin([5, 13])]

        mapping = DENGUE_MAPPING if self.type_disease == 'dengue' else CHIK_MAPPING
        df['final_classification'] = df['final_classification'].map(mapping).fillna(0).astype(int)
        df['final_classification'].value_counts()

        return df

    def load_data_process(self):
        df = self.load_data()
        df = self.process_date_features(df)
        df = self.process_features(df)
        df = self.process_target(df)

        return df, CATEGORICAL_COLUMNS, BINARY_COLUMNS
