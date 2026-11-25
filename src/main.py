from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid")

RAIZ = Path(__file__).resolve().parent.parent
RUTA_CRUDO = RAIZ / "data" / "raw" / "Aggregated_online_casino.csv"
RUTA_LIMPIO = RAIZ / "data" / "processed" / "clientes_limpios.csv"
RUTA_FINAL = RAIZ / "data" / "processed" / "clientes_con_clusters.csv"
CARPETA_FIG = RAIZ / "outputs" / "figuras"
CARPETA_TAB = RAIZ / "outputs" / "tablas"

for carpeta in (RUTA_LIMPIO.parent, CARPETA_FIG, CARPETA_TAB):
    carpeta.mkdir(parents=True, exist_ok=True)

VARS_MODELO = ["deposit_amount", "deposits", "activeweeks", "intensity", "propnight"]
VARS_LOG = ["deposit_amount", "deposits", "activeweeks", "intensity"]
SEMILLA = 42


def ordenar_clusters(df, etiquetas):
    # ordenamos de menor a mayor gasto para que el numero de grupo sea facil de leer
    medianas = df.groupby(etiquetas)["deposit_amount"].median().sort_values()
    mapa = {viejo: nuevo for nuevo, viejo in enumerate(medianas.index)}
    return np.array([mapa[e] for e in etiquetas])


def main():
    # ==========================================================
    # Parte 2: Obtencion y exploracion del dataset
    # ==========================================================
    df = pd.read_csv(RUTA_CRUDO)
    print("=== PARTE 2: EXPLORACION DEL DATASET ===")
    print("Primeros 10 registros:")
    print(df.head(10).to_string())
    print("\nDimensiones:", df.shape)
    print("Columnas:", df.columns.tolist())
    print("\nTipos de datos:")
    print(df.dtypes)

    desc = df.describe().T.round(2)
    print("\nEstadisticas descriptivas:")
    print(desc.to_string())
    desc.to_csv(CARPETA_TAB / "estadisticas_generales.csv")

    print("\nValores nulos por columna:")
    print(df.isnull().sum())
    print("Registros duplicados:", df.duplicated().sum())

    # 3 graficas exploratorias basicas
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["deposit_amount"], bins=45, log_scale=True, color="#2a6f97", ax=ax)
    ax.set_title("Distribucion del total depositado")
    ax.set_xlabel("Total depositado en USD (escala log)")
    ax.set_ylabel("Clientes")
    fig.tight_layout()
    fig.savefig(CARPETA_FIG / "01_distribucion_deposito.png", dpi=150)
    plt.close(fig)

    conteo_meses = df["activemonths"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=conteo_meses.index, y=conteo_meses.values, color="#2a6f97", ax=ax)
    ax.set_title("Meses activos por cliente")
    ax.set_xlabel("Meses activos (de 12)")
    ax.set_ylabel("Clientes")
    fig.tight_layout()
    fig.savefig(CARPETA_FIG / "02_meses_activos.png", dpi=150)
    plt.close(fig)

    cols_corr = VARS_MODELO + ["netloss", "withdrawal_amount", "declines"]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sns.heatmap(df[cols_corr].corr(method="spearman"), annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Correlacion entre variables (Spearman)")
    fig.tight_layout()
    fig.savefig(CARPETA_FIG / "03_correlaciones.png", dpi=150)
    plt.close(fig)

    # ==========================================================
    # Parte 3: Limpieza y preparacion de datos
    # ==========================================================
    print("\n=== PARTE 3: LIMPIEZA Y PREPARACION ===")
    negativos = int((df[["deposits", "deposit_amount", "withdrawals", "withdrawal_amount"]] < 0).any(axis=1).sum())
    prop_fuera = int(((df["propnight"] < 0) | (df["propnight"] > 1)).sum())
    meses_mal = int((df["activemonths"] > 12).sum())
    print(f"Inconsistencias -> negativos: {negativos}, propnight fuera de rango: {prop_fuera}, meses>12: {meses_mal}")

    # quitamos duplicados, nulos y la columna logdeclines que ya venia repetida
    df = df.drop_duplicates(subset="ID").dropna().drop(columns=["logdeclines"]).reset_index(drop=True)
    df["ID"] = df["ID"].astype(str)
    for col in ["deposits", "withdrawals", "declines", "nightdeposits", "activeweeks", "activemonths"]:
        df[col] = df[col].astype(int)

    # revisamos atipicos con IQR pero no los borramos porque son clientes reales que gastan mas
    print("\nAtipicos detectados por IQR (se conservan en el dataset):")
    for col in VARS_MODELO:
        q1, q3 = df[col].quantile([0.25, 0.75])
        lim = q3 + 1.5 * (q3 - q1)
        cant = int((df[col] > lim).sum())
        print(f"  {col}: {cant} registros arriba de {lim:.2f}")

    df.to_csv(RUTA_LIMPIO, index=False)

    # log1p para bajar el sesgo y StandardScaler para poner todas las variables en la misma escala
    X = df[VARS_MODELO].copy()
    for col in VARS_LOG:
        X[col] = np.log1p(X[col])
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)

    fig, ejes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(df["deposit_amount"], bins=40, color="#c1121f", ax=ejes[0])
    ejes[0].set_title("Antes (deposit_amount original)")
    sns.histplot(Z[:, 0], bins=40, color="#2a6f97", ax=ejes[1])
    ejes[1].set_title("Despues (log1p + StandardScaler)")
    fig.tight_layout()
    fig.savefig(CARPETA_FIG / "04_escalado.png", dpi=150)
    plt.close(fig)

    # ==========================================================
    # Parte 4: Implementacion del algoritmo K-Means
    # ==========================================================
    print("\n=== PARTE 4: K-MEANS ===")
    eval_filas = []
    for k in range(2, 7):  # evaluamos 5 valores de K: 2, 3, 4, 5 y 6
        km = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA)
        et = km.fit_predict(Z)
        eval_filas.append({
            "k": k,
            "inercia": round(km.inertia_, 1),
            "silhouette": round(silhouette_score(Z, et), 4),
        })
    eval_df = pd.DataFrame(eval_filas)
    print("Evaluacion de K (2 a 6):")
    print(eval_df.to_string(index=False))
    eval_df.to_csv(CARPETA_TAB / "evaluacion_k.csv", index=False)

    fig, ejes = plt.subplots(1, 2, figsize=(10, 4))
    ejes[0].plot(eval_df["k"], eval_df["inercia"], marker="o", color="#2a6f97")
    ejes[0].set_title("Metodo del codo")
    ejes[0].set_xlabel("K")
    ejes[0].set_ylabel("Inercia")
    ejes[1].plot(eval_df["k"], eval_df["silhouette"], marker="o", color="#6a994e")
    ejes[1].set_title("Silhouette por K")
    ejes[1].set_xlabel("K")
    ejes[1].set_ylabel("Silhouette")
    fig.tight_layout()
    fig.savefig(CARPETA_FIG / "05_codo_silhouette.png", dpi=150)
    plt.close(fig)

    # comparamos K=3 vs K=4
    comp_filas = []
    for k in (3, 4):
        km_tmp = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA)
        et_tmp = ordenar_clusters(df, km_tmp.fit_predict(Z))
        tamanos = pd.Series(et_tmp).value_counts().sort_index()
        comp_filas.append({
            "configuracion": f"K = {k}",
            "inercia": round(km_tmp.inertia_, 1),
            "silhouette": round(silhouette_score(Z, et_tmp), 4),
            "clientes_por_cluster": " / ".join(str(v) for v in tamanos.values),
        })
    comp_df = pd.DataFrame(comp_filas)
    print("\nComparacion K=3 vs K=4:")
    print(comp_df.to_string(index=False))
    comp_df.to_csv(CARPETA_TAB / "comparacion_k.csv", index=False)

    # modelo final con K=4 y fit_predict()
    modelo_final = KMeans(n_clusters=4, n_init=10, random_state=SEMILLA)
    etiquetas = ordenar_clusters(df, modelo_final.fit_predict(Z))
    df["cluster"] = etiquetas

    print("\nRegistros por cluster:")
    print(df["cluster"].value_counts().sort_index().to_string())

    # centroides regresados a unidades originales
    centros_z = pd.DataFrame(Z, columns=VARS_MODELO).groupby(etiquetas).mean()
    centros = pd.DataFrame(scaler.inverse_transform(centros_z), columns=VARS_MODELO, index=centros_z.index)
    for col in VARS_LOG:
        centros[col] = np.expm1(centros[col])
    centros.index.name = "cluster"
    centros = centros.round(2)
    print("\nCentroides en unidades originales:")
    print(centros.to_string())
    centros.to_csv(CARPETA_TAB / "centroides.csv")

    # grafica de clusters con PCA
    pca = PCA(n_components=2, random_state=SEMILLA)
    coords = pca.fit_transform(Z)
    df["pca1"] = coords[:, 0].round(4)
    df["pca2"] = coords[:, 1].round(4)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    colores = ["#8da9c4", "#5b4b8a", "#e9a23b", "#b23a48"]
    for c in sorted(df["cluster"].unique()):
        m = df["cluster"] == c
        ax.scatter(coords[m, 0], coords[m, 1], s=14, alpha=0.6, color=colores[c], label=f"Cluster {c}")
    ax.set_title("Agrupamiento de clientes (PCA)")
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CARPETA_FIG / "06_clusters_pca.png", dpi=150)
    plt.close(fig)

    # ==========================================================
    # Parte 5: Analisis e interpretacion de resultados
    # ==========================================================
    print("\n=== PARTE 5: ANALISIS DE GRUPOS ===")
    nombres = {
        0: "Ocasionales",
        1: "De madrugada",
        2: "Regulares",
        3: "Alto gasto",
    }
    df["segmento"] = df["cluster"].map(nombres)

    stats_grupo = df.groupby("segmento")[VARS_MODELO + ["netloss", "withdrawals"]].agg(["mean", "median", "std"]).round(2)
    stats_grupo.to_csv(CARPETA_TAB / "estadisticas_por_grupo.csv")

    total_dep = df["deposit_amount"].sum()
    grupos = df.groupby("segmento")
    tabla_comp = pd.DataFrame({
        "cluster": grupos["cluster"].first(),
        "clientes": grupos.size(),
        "pct_clientes": (100 * grupos.size() / len(df)).round(1),
        "deposito_mediano": grupos["deposit_amount"].median().round(0),
        "depositos_mediana": grupos["deposits"].median(),
        "semanas_mediana": grupos["activeweeks"].median(),
        "prop_madrugada_media": grupos["propnight"].mean().round(2),
        "pct_dinero": (100 * grupos["deposit_amount"].sum() / total_dep).round(1),
    }).sort_values("cluster")
    print("\nTabla comparativa:")
    print(tabla_comp.to_string())
    tabla_comp.to_csv(CARPETA_TAB / "tabla_comparativa.csv")

    df.to_csv(RUTA_FINAL, index=False)
    print("\nListo, dataset final guardado en:", RUTA_FINAL)


if __name__ == "__main__":
    main()
