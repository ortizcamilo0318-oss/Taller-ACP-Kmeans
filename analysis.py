"""Procesamiento y algoritmos del taller: preparación, ACP y K-Means.

Todas las tablas y métricas que muestra la aplicación se calculan aquí,
a partir del CSV original, cada vez que cambia la configuración.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).parent / "data" / "Student_Depression_Dataset.csv"
SEED = 42

# Variables ordinales: se codifican respetando el orden natural de sus categorías.
ORDINAL_MAPS = {
    "Family_background": {"Lower Middle class": 1, "Middle class": 2, "Rich class": 3},
    "Health_condition": {"Critical": 1, "Abnormal": 2, "Normal": 3, "Good": 4},
}
ORDINAL_NUM = {"Family_background": "Nivel_socioeconomico", "Health_condition": "Estado_salud"}

# Nombre de la columna de trabajo -> etiqueta legible en español.
LABELS = {
    "Age": "Edad (años)",
    "Current_semester": "Semestre actual",
    "CGPA": "Promedio acumulado (CGPA)",
    "Cost_without_study": "Gastos de vida (sin costos académicos)",
    "Nivel_socioeconomico": "Nivel socioeconómico (1-3)",
    "Estado_salud": "Salud física (1-4)",
}
MODEL_VARS = list(LABELS)
DEFAULT_VARS = MODEL_VARS

CATEGORICAL = {
    "Gender": "Género",
    "Family_background": "Contexto familiar",
    "Relationship_status": "Relación sentimental",
    "Marital_status": "Estado civil",
    "Living_style": "Con quién vive",
    "friend_circle": "Tiene círculo de amigos",
    "Health_condition": "Condición de salud",
    "Home_division": "División de origen",
    "Depression": "Depresión reportada",
}


def load_raw(path=DATA_PATH):
    """Lee el CSV tal como viene de la fuente."""
    return pd.read_csv(path)


def quality_report(df):
    """Resumen de calidad: tipo, faltantes y valores únicos por variable."""
    return pd.DataFrame(
        {
            "Tipo": df.dtypes.astype(str),
            "Faltantes": df.isna().sum(),
            "Valores únicos": df.nunique(),
        }
    )


def prepare(df, drop_semester_zero=False):
    """Limpia la base y agrega las variables ordinales codificadas.

    - Elimina duplicados exactos y filas con faltantes en las variables del modelo.
    - Opcionalmente excluye registros con semestre 0 (posible inconsistencia).
    """
    out = df.drop_duplicates().copy()
    for col, mapping in ORDINAL_MAPS.items():
        out[ORDINAL_NUM[col]] = out[col].map(mapping)
    out = out.dropna(subset=MODEL_VARS)
    if drop_semester_zero:
        out = out[out["Current_semester"] > 0]
    return out.reset_index(drop=True)


def standardize(df, variables):
    """Estandariza (media 0, desviación 1) las variables elegidas."""
    scaler = StandardScaler()
    z = scaler.fit_transform(df[variables])
    return pd.DataFrame(z, columns=variables, index=df.index)


def run_pca(z):
    """ACP completo sobre los datos estandarizados."""
    pca = PCA(random_state=SEED)
    scores = pca.fit_transform(z)
    names = [f"CP{i + 1}" for i in range(z.shape[1])]
    variance = pd.DataFrame(
        {
            "Componente": names,
            "Valor propio": pca.explained_variance_,
            "Varianza explicada (%)": pca.explained_variance_ratio_ * 100,
            "Varianza acumulada (%)": pca.explained_variance_ratio_.cumsum() * 100,
        }
    )
    loadings = pd.DataFrame(pca.components_.T, index=z.columns, columns=names)
    scores = pd.DataFrame(scores, columns=names, index=z.index)
    return {"variance": variance, "loadings": loadings, "scores": scores}


def suggest_components(variance, threshold=80.0):
    """Número mínimo de componentes que alcanza el umbral de varianza acumulada."""
    cum = variance["Varianza acumulada (%)"].to_numpy()
    return int(min(np.searchsorted(cum, threshold - 1e-9) + 1, len(cum)))


def kaiser_components(variance):
    """Componentes con valor propio mayor que 1 (criterio de Kaiser)."""
    return int((variance["Valor propio"] > 1).sum())


def evaluate_k(x, k_min=2, k_max=10):
    """Inercia (codo) y silueta promedio para cada K."""
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(x)
        rows.append({"K": k, "Inercia": km.inertia_, "Silueta": silhouette_score(x, km.labels_)})
    return pd.DataFrame(rows)


def fit_kmeans(x, k):
    """Ajusta K-Means y devuelve etiquetas con letras (A, B, ...) y siluetas individuales.

    Se usan letras para no sugerir un orden entre los grupos.
    """
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(x)
    letters = np.array([chr(65 + i) for i in range(k)])
    labels = pd.Series(letters[km.labels_], index=x.index, name="Grupo")
    sil = pd.Series(silhouette_samples(x, km.labels_), index=x.index, name="Silueta")
    return labels, sil


def cluster_profile(df, labels, variables):
    """Promedios por grupo en unidades originales y tamaño de cada grupo."""
    prof = df.groupby(labels)[variables].mean()
    prof.insert(0, "Estudiantes", labels.value_counts().sort_index())
    prof.insert(1, "% de la muestra", prof["Estudiantes"] / len(df) * 100)
    return prof


def cluster_zscores(df, labels, variables):
    """Promedio por grupo expresado en desviaciones estándar respecto a la muestra."""
    z = (df[variables] - df[variables].mean()) / df[variables].std(ddof=0)
    return z.groupby(labels).mean()


def describe_cluster(zrow, threshold=0.4):
    """Frase corta con los rasgos que más distinguen a un grupo."""
    parts = []
    for var, val in zrow.sort_values(key=np.abs, ascending=False).items():
        if abs(val) < threshold:
            continue
        level = "por encima del promedio" if val > 0 else "por debajo del promedio"
        if abs(val) >= 1:
            level = "muy " + level
        parts.append(f"{LABELS[var].split(' (')[0].lower()} {level} ({val:+.2f} d.e.)")
    return "; ".join(parts) if parts else "cercano al promedio en todas las variables"
