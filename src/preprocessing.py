"""Construção do target, limpeza e feature engineering."""
import numpy as np
import pandas as pd
from src.config import TARGET, FEATURES

def check_missing(df: pd.DataFrame) -> pd.DataFrame:
    total = df.isna().sum()
    pct = (total / len(df) * 100).round(2)
    return pd.DataFrame({"nulos": total, "pct": pct}).query("nulos > 0").sort_values("nulos", ascending=False)

def build_target(application: pd.DataFrame, credit: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    application_unique = application.drop_duplicates(subset="ID", keep="first").copy()
    ids_comuns = set(application_unique["ID"]).intersection(set(credit["ID"]))
    credit_comum = credit[credit["ID"].isin(ids_comuns)].copy()
    mapa_status = {"C": -1, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "X": np.nan}
    credit_comum["STATUS_NUM"] = credit_comum["STATUS"].map(mapa_status)
    pior_status = credit_comum.groupby("ID")["STATUS_NUM"].max()
    pior_status_valido = pior_status.dropna()
    target = (pior_status_valido >= 1).astype(int).rename(TARGET)
    return application_unique, target

def build_processed_dataset(application: pd.DataFrame, credit: pd.DataFrame) -> pd.DataFrame:
    application_unique, target = build_target(application, credit)
    base = (
        application_unique[application_unique["ID"].isin(target.index)]
        .merge(target, left_on="ID", right_index=True, how="inner")
        .copy()
    )
    base["OCCUPATION_TYPE"] = base["OCCUPATION_TYPE"].fillna("Unknown")
    base["AGE_YEARS"] = -base["DAYS_BIRTH"] / 365.25
    base["FLAG_EMPLOYMENT_SENTINEL"] = base["DAYS_EMPLOYED"].eq(365243).astype(int)
    dias_emprego_validos = base["DAYS_EMPLOYED"].replace(365243, np.nan)
    base["EMPLOYMENT_YEARS"] = -dias_emprego_validos / 365.25
    base["FLAG_FAMILY_INCONSISTENT"] = (base["CNT_CHILDREN"] > base["CNT_FAM_MEMBERS"]).astype(int)
    for col in ["FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
        base[col] = base[col].map({"N": 0, "Y": 1}).astype(int)
    base["CNT_CHILDREN"] = base["CNT_CHILDREN"].astype(int)
    base["CNT_FAM_MEMBERS"] = base["CNT_FAM_MEMBERS"].astype(int)
    base = base.drop(columns=["DAYS_BIRTH", "DAYS_EMPLOYED", "FLAG_MOBIL"])
    base = base[["ID"] + FEATURES + [TARGET]].copy()
    return base
