"""Pré-processamento e modelos candidatos do projeto."""
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from src.config import RANDOM_STATE, QUANTITATIVE_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES

def build_preprocessors():
    escalonado = ColumnTransformer([
        ("quantitativas", Pipeline([
            ("imputacao", SimpleImputer(strategy="median")),
            ("escala", StandardScaler()),
        ]), QUANTITATIVE_FEATURES),
        ("binarias", "passthrough", BINARY_FEATURES),
        ("categoricas", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ], remainder="drop")
    arvores = ColumnTransformer([
        ("quantitativas", Pipeline([("imputacao", SimpleImputer(strategy="median"))]), QUANTITATIVE_FEATURES),
        ("binarias", "passthrough", BINARY_FEATURES),
        ("categoricas", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ], remainder="drop")
    return escalonado, arvores

def get_models():
    escalonado, arvores = build_preprocessors()
    return {
        "KNN — RandomOverSampler": ImbPipeline([
            ("preprocessamento", clone(escalonado)),
            ("balanceamento", RandomOverSampler(random_state=RANDOM_STATE)),
            ("modelo", KNeighborsClassifier(n_neighbors=11, weights="distance")),
        ]),
        "Decision Tree — RandomOverSampler": ImbPipeline([
            ("preprocessamento", clone(arvores)),
            ("balanceamento", RandomOverSampler(random_state=RANDOM_STATE)),
            ("modelo", DecisionTreeClassifier(
                criterion="gini", max_depth=None, min_samples_split=2,
                min_samples_leaf=1, random_state=RANDOM_STATE,
            )),
        ]),
        "Random Forest — Class Weight": Pipeline([
            ("preprocessamento", clone(arvores)),
            ("modelo", RandomForestClassifier(
                n_estimators=100, max_features="sqrt", max_depth=None,
                min_samples_leaf=1, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ]),
    }
