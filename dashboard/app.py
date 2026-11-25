from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Segmentacion K-Means Casino", layout="wide")

RAIZ = Path(__file__).resolve().parent.parent
RUTA_FINAL = RAIZ / "data" / "processed" / "clientes_con_clusters.csv"
CARPETA_TAB = RAIZ / "outputs" / "tablas"

COLORES = {
    "Ocasionales": "#8da9c4",
    "De madrugada": "#5b4b8a",
    "Regulares": "#e9a23b",
    "Alto gasto": "#b23a48",
}
ORDEN = ["Ocasionales", "De madrugada", "Regulares", "Alto gasto"]

ETIQUETAS = {
    "deposit_amount": "Total depositado (USD)",
    "deposits": "Numero de depositos",
    "activeweeks": "Semanas activas",
    "intensity": "Depositos por semana",
    "propnight": "Proporcion de madrugada",
    "netloss": "Perdida neta (USD)",
}

if not RUTA_FINAL.exists():
    st.error("Falta el archivo procesado. Corre primero: python src/main.py")
    st.stop()


@st.cache_data
def cargar_datos():
    return pd.read_csv(RUTA_FINAL)


@st.cache_data
def cargar_tabla(nombre, **kwargs):
    return pd.read_csv(CARPETA_TAB / nombre, **kwargs)


datos = cargar_datos()

# Navegacion de las 2 paginas que pide la actividad
pagina = st.sidebar.radio("Pagina", ["Analisis general", "Segmentacion K-Means"])

# 3 filtros interactivos en la barra lateral
st.sidebar.title("Filtros")
segmentos = st.sidebar.multiselect("1. Segmento", ORDEN, default=ORDEN)

montos = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 410000]
monto_min, monto_max = st.sidebar.select_slider(
    "2. Total depositado (USD)", options=montos, value=(montos[0], montos[-1])
)

horario = st.sidebar.radio(
    "3. Horario de depositos",
    ["Todos", "De madrugada (mas de 50%)", "Fuera de madrugada (50% o menos)"],
)

filtrado = datos[
    datos["segmento"].isin(segmentos)
    & datos["deposit_amount"].between(monto_min, monto_max)
]
if horario.startswith("De madrugada"):
    filtrado = filtrado[filtrado["propnight"] > 0.5]
elif horario.startswith("Fuera de madrugada"):
    filtrado = filtrado[filtrado["propnight"] <= 0.5]

st.sidebar.caption(f"Mostrando {len(filtrado):,} de {len(datos):,} clientes")

st.title("Segmentacion de jugadores de casino en linea con K-Means")
st.caption("Dataset: Ghaharian et al. (2023), 2,286 clientes (marzo 2019 a febrero 2020)")

if filtrado.empty:
    st.warning("No hay clientes con esos filtros, prueba ampliar el rango.")
    st.stop()

# 4 indicadores KPI
k1, k2, k3, k4 = st.columns(4)
k1.metric("Clientes", f"{len(filtrado):,}")
k2.metric("Total depositado", f"${filtrado['deposit_amount'].sum():,.0f}")
k3.metric("Deposito mediano", f"${filtrado['deposit_amount'].median():,.0f}")
k4.metric("Perdida neta total", f"${filtrado['netloss'].sum():,.0f}")
st.divider()

if pagina == "Analisis general":
    st.header("Pagina 1. Analisis general del dataset")
    st.write("Aqui se muestran las graficas y estadisticas generales de los clientes antes de agruparlos.")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(filtrado, x="deposit_amount", nbins=45, log_y=True,
                           color_discrete_sequence=["#2a6f97"],
                           labels={"deposit_amount": "Total depositado (USD)"})
        fig.update_layout(title="Distribucion de depositos (escala log en clientes)",
                          yaxis_title="Clientes")
        st.plotly_chart(fig, width="stretch")

    with col2:
        meses = filtrado["activemonths"].value_counts().sort_index().reset_index()
        meses.columns = ["Meses activos", "Clientes"]
        fig = px.bar(meses, x="Meses activos", y="Clientes", text="Clientes",
                     color_discrete_sequence=["#2a6f97"])
        fig.update_layout(title="Meses activos por cliente (de 12)", xaxis=dict(dtick=1))
        st.plotly_chart(fig, width="stretch")

    cols_corr = ["deposit_amount", "deposits", "activeweeks", "intensity", "propnight", "netloss"]
    corr = filtrado[cols_corr].corr(method="spearman").round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig.update_layout(title="Correlacion entre variables principales (Spearman)")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Estadisticas generales del dataset original")
    st.dataframe(cargar_tabla("estadisticas_generales.csv", index_col=0), width="stretch")

else:
    st.header("Pagina 2. Segmentacion mediante K-Means")
    st.write("Resultados de agrupar a los clientes en 4 segmentos segun su gasto, frecuencia y horario.")

    col_a, col_b = st.columns([1, 1.3])
    with col_a:
        conteo = filtrado["segmento"].value_counts().reindex(ORDEN).dropna().reset_index()
        conteo.columns = ["Segmento", "Clientes"]
        fig = px.bar(conteo, x="Clientes", y="Segmento", orientation="h", color="Segmento",
                     color_discrete_map=COLORES, text="Clientes")
        fig.update_layout(title="Distribucion de clientes por cluster", showlegend=False)
        st.plotly_chart(fig, width="stretch")

    with col_b:
        fig = px.scatter(filtrado, x="pca1", y="pca2", color="segmento",
                         color_discrete_map=COLORES,
                         category_orders={"segmento": ORDEN},
                         hover_data=["deposit_amount", "deposits", "activeweeks"],
                         labels={"pca1": "PCA 1", "pca2": "PCA 2", "segmento": "Segmento"},
                         opacity=0.65)
        fig.update_layout(title="Visualizacion de agrupamientos (PCA)")
        st.plotly_chart(fig, width="stretch")

    # Dos graficas adicionales para comparar caracteristicas de los grupos
    col_c, col_d = st.columns(2)
    with col_c:
        rep = filtrado.groupby("segmento").agg(clientes=("ID", "size"), dinero=("deposit_amount", "sum"))
        rep = (100 * rep / rep.sum()).reindex(ORDEN).dropna().reset_index()
        largo = rep.melt(id_vars="segmento", var_name="Medida", value_name="Porcentaje")
        largo["Medida"] = largo["Medida"].map({"clientes": "% de clientes", "dinero": "% del dinero"})
        fig = px.bar(largo, x="segmento", y="Porcentaje", color="Medida", barmode="group",
                     color_discrete_sequence=["#8da9c4", "#b23a48"], labels={"segmento": ""})
        fig.update_layout(title="Porcentaje de clientes vs porcentaje del dinero")
        st.plotly_chart(fig, width="stretch")

    with col_d:
        var = st.selectbox("Variable a comparar", list(ETIQUETAS.keys()),
                           format_func=lambda c: ETIQUETAS[c])
        usar_log = var in ("deposit_amount", "deposits", "activeweeks", "intensity")
        fig = px.box(filtrado, x="segmento", y=var, color="segmento", log_y=usar_log,
                     color_discrete_map=COLORES, category_orders={"segmento": ORDEN},
                     labels={"segmento": "", var: ETIQUETAS[var]})
        fig.update_layout(title=f"{ETIQUETAS[var]} por grupo", showlegend=False)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Estadisticas descriptivas por grupo")
    resumen = filtrado.groupby("segmento").agg(
        Clientes=("ID", "size"),
        Deposito_mediano=("deposit_amount", "median"),
        Depositos_mediana=("deposits", "median"),
        Semanas_activas_mediana=("activeweeks", "median"),
        Prop_madrugada_media=("propnight", "mean"),
        Perdida_neta_mediana=("netloss", "median"),
    ).reindex(ORDEN).dropna().round(2)
    st.dataframe(resumen, width="stretch")

    st.subheader("Descripcion de los grupos")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Ocasionales (43.1%)**: Deposito mediano de $50 y 1 solo deposito en el año casi nunca de madrugada.")
        st.markdown("**De madrugada (15.6%)**: Tambien gastan poco ($75 de mediana), pero hacen casi todos sus depositos entre 1 y 5 a. m. (86%).")
    with c2:
        st.markdown("**Regulares (26.8%)**: Deposito mediano de $648, regresan varias semanas al año con un gasto medio.")
        st.markdown("**Alto gasto (14.6%)**: Deposito mediano de $6,571 y 103 depositos al año, ellos solos juntan el 80.7% del dinero.")

    st.subheader("Hallazgos y recomendaciones")
    col_h, col_r = st.columns(2)
    with col_h:
        st.markdown("**Hallazgos principales**")
        st.markdown("- Un grupo chico (14.6% de los clientes) deja el 80.7% de todo el dinero depositado.")
        st.markdown("- Jugar de madrugada no significa gastar mas, la mayoria de ese grupo solo deposita $75 en promedio.")
        st.markdown("- Casi el 59% de los clientes solo prueba la pagina una o dos veces y aporta apenas el 3.3% del dinero.")
    with col_r:
        st.markdown("**Recomendaciones**")
        st.markdown("- Poner limites de deposito y recordatorios de juego responsable sobre todo en el grupo de Alto gasto.")
        st.markdown("- Vigilar al grupo de Regulares por si suben sus depositos por semana y pasan a ser intensivos.")
        st.markdown("- Agregar avisos en el horario de madrugada (1 a 5 a. m.) ya que ahi hay un perfil distinto de usuarios.")
