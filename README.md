# Segmentacion de jugadores de casino en linea con K-Means

Proyecto integrador de mineria de datos. Agrupa a 2,286 clientes de un casino en linea usando K-Means segun cuanto depositan, cuantas veces juegan y en que horario lo hacen y muestra los resultados en un dashboard hecho con Streamlit.

## Problema y objetivo

En un casino en linea hay usuarios que solo entran una vez y otros que depositan miles de dolares al año, por eso los promedios generales confunden. El objetivo general es identificar los distintos tipos de jugadores a partir de sus depositos usando K-Means.

**Objetivos especificos**
1. Explorar y limpiar los datos de transacciones de los clientes.
2. Aplicar el algoritmo K-Means para formar grupos con comportamiento parecido.
3. Construir un dashboard interactivo para ver los resultados y las caracteristicas de cada grupo.

## Herramientas utilizadas

- Python 3.10+
- Pandas y NumPy
- Scikit-learn (`StandardScaler`, `KMeans`, `PCA`, `silhouette_score`)
- Matplotlib y Seaborn
- Streamlit y Plotly

## Datos

- **Fuente:** Ghaharian, K. et al. (2023). *Two Aggregated Datasets of Gamblers' Payment Transaction Records*. Mendeley Data. DOI 10.17632/6xtgb5cx3z.1 (licencia CC BY 4.0).
- **Archivo:** `data/raw/Aggregated_online_casino.csv` con 2,286 filas y 16 columnas (marzo 2019 a febrero 2020).

## Estructura del repositorio

```
proyecto-kmeans-casino/
├── data/
│   ├── raw/Aggregated_online_casino.csv
│   └── processed/
│       ├── clientes_limpios.csv
│       └── clientes_con_clusters.csv
├── src/
│   └── main.py              exploracion, limpieza, K-Means y analisis
├── dashboard/
│   └── app.py               dashboard interactivo con 2 paginas
├── outputs/
│   ├── figuras/             graficas generadas en PNG
│   └── tablas/              tablas de metricas, centroides y comparacion
├── docs/                    documentacion en PDF
├── requirements.txt
└── README.md
```

## Como ejecutarlo

```bash
# 1. Instalar librerias
pip install -r requirements.txt

# 2. Correr el analisis y generar graficas y tablas
python src/main.py

# 3. Abrir el dashboard
streamlit run dashboard/app.py
```

## Resumen del metodo y resultados

- **Variables usadas:** `deposit_amount`, `deposits`, `activeweeks`, `intensity` y `propnight`.
- **Limpieza y escalado:** se revisaron nulos, duplicados e inconsistencias (0 errores), se quito la columna redundante `logdeclines`, se dejaron los atipicos por ser clientes reales y se aplico `log1p` mas `StandardScaler`.
- **Eleccion de K:** se probaron valores de K de 2 a 6 y se eligio **K = 4** porque a diferencia de K = 3 separa a los jugadores regulares de los de alto gasto.
- **Grupos encontrados:** Ocasionales (43.1%), De madrugada (15.6%), Regulares (26.8%) y Alto gasto (14.6%, quienes concentran el 80.7% del dinero).
