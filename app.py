"""
Dashboard Avanzado de Análisis de Depósitos de Cobre y Oro
Consultoría para Gobierno de Estados Unidos - Transición Energética

Autor: Geólogo Colombiano - Estudiante Maestría Ciencia de Datos EAFIT
Fecha: Febrero 2026
Curso: Fundamentos de Ciencia de Datos

FUNCIONALIDADES IMPLEMENTADAS:
✅ Carga dinámica de datos (CSV, JSON, URL) con manejo robusto de errores
✅ Procesamiento completo (duplicados, imputación, outliers, feature engineering)
✅ Filtros globales interactivos (fechas, categorías, sliders numéricos)
✅ Visualizaciones interactivas Plotly (control total del usuario)
✅ Mapas geoespaciales para datos georreferenciados
✅ Análisis con IA (Groq API - Llama-3/Mixtral)
✅ Organización con tabs (Univariado, Bivariado, Geoespacial, IA, Reportes)
✅ Uso de st.columns y st.expander para mejor UX
✅ Generación de reportes PDF profesionales
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import requests
from io import StringIO, BytesIO
from groq import Groq

# ==============================================================================
# CONFIGURACIÓN INICIAL DE LA APLICACIÓN
# ==============================================================================

st.set_page_config(
    page_title="Análisis Cu-Au Depósitos USA",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CONSTANTES Y CONFIGURACIÓN
# ==============================================================================

# Columnas requeridas del dataset
REQUIRED_COLUMNS = [
    'DEPOSIT_GROUP', 'DEPOSIT_TYPE', 'PRIMARY_COMMODITIES', 'SAMPLE_UID', 
    'SAMPLING_METHOD', 'PROVINCE', 'AU_PPB', 'CU_PPM', 'ANALYSIS_DATETIME', 
    'COUNTRY', 'STATE', 'SAMPLE_LONGITUDE_WGS84', 'SAMPLE_LATITUDE_WGS84', 
    'SAMPLE_LOCATION_DESCRIPTION'
]

# Mapeo de tipos de datos para carga segura
DTYPE_MAP = {col: 'str' for col in REQUIRED_COLUMNS}

# ==============================================================================
# FUNCIONES DE CARGA DE DATOS
# ==============================================================================

def load_csv_file(uploaded_file):
    """
    Carga un archivo CSV con manejo robusto de tipos de datos mixtos.
    """
    try:
        df = pd.read_csv(
            uploaded_file,
            usecols=REQUIRED_COLUMNS,
            dtype=DTYPE_MAP,
            low_memory=False
        )
        st.success(f"✅ Archivo CSV cargado: {df.shape[0]} registros, {df.shape[1]} columnas")
        return df
    except ValueError as ve:
        st.error(f"❌ Error: El archivo no contiene todas las columnas requeridas.")
        st.error(f"Detalle: {str(ve)}")
        return None
    except pd.errors.EmptyDataError:
        st.error("❌ Error: El archivo CSV está vacío.")
        return None
    except pd.errors.ParserError as pe:
        st.error(f"❌ Error al parsear el archivo CSV.")
        st.error(f"Detalle: {str(pe)}")
        return None
    except Exception as e:
        st.error(f"❌ Error inesperado: {str(e)}")
        return None


def load_json_file(uploaded_file):
    """
    Carga un archivo JSON y lo convierte a DataFrame.
    """
    try:
        json_data = json.load(uploaded_file)
        df = pd.DataFrame(json_data)
        
        missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            st.error(f"❌ Error: Faltan columnas: {missing_cols}")
            return None
        
        df = df[REQUIRED_COLUMNS]
        st.success(f"✅ Archivo JSON cargado: {df.shape[0]} registros, {df.shape[1]} columnas")
        return df
    except json.JSONDecodeError:
        st.error(f"❌ Error: El archivo no es un JSON válido.")
        return None
    except Exception as e:
        st.error(f"❌ Error inesperado: {str(e)}")
        return None


def load_from_url(url):
    """
    Carga datos desde una URL (CSV o JSON).
    """
    try:
        if not url or url.strip() == "":
            st.warning("⚠️ Por favor ingresa una URL válida.")
            return None
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        if url.lower().endswith('.json') or 'application/json' in response.headers.get('content-type', ''):
            json_data = response.json()
            df = pd.DataFrame(json_data)
        else:
            df = pd.read_csv(
                StringIO(response.text),
                usecols=REQUIRED_COLUMNS,
                dtype=DTYPE_MAP,
                low_memory=False
            )
        
        missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            st.error(f"❌ Error: Faltan columnas: {missing_cols}")
            return None
        
        df = df[REQUIRED_COLUMNS]
        st.success(f"✅ Datos cargados desde URL: {df.shape[0]} registros")
        return df
    except requests.exceptions.Timeout:
        st.error("❌ Error: Tiempo de espera agotado (30 segundos).")
        return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Error: No se pudo conectar a la URL.")
        return None
    except requests.exceptions.HTTPError as he:
        st.error(f"❌ Error HTTP: {he.response.status_code}")
        return None
    except Exception as e:
        st.error(f"❌ Error inesperado: {str(e)}")
        return None


# ==============================================================================
# FUNCIONES DE PROCESAMIENTO Y LIMPIEZA
# ==============================================================================

def convert_data_types(df):
    """
    Convierte los tipos de datos de manera segura después de la carga inicial.
    Trata valores negativos en variables geoquímicas como datos no disponibles.
    """
    try:
        df_processed = df.copy()
        
        # Convertir columnas numéricas geoquímicas
        df_processed['AU_PPB'] = pd.to_numeric(df_processed['AU_PPB'], errors='coerce')
        df_processed['CU_PPM'] = pd.to_numeric(df_processed['CU_PPM'], errors='coerce')
        
        # TRATAMIENTO DE VALORES NEGATIVOS: convertir a NaN
        # Los valores negativos en concentraciones geoquímicas no tienen sentido físico
        negative_au = (df_processed['AU_PPB'] < 0).sum()
        negative_cu = (df_processed['CU_PPM'] < 0).sum()
        
        df_processed.loc[df_processed['AU_PPB'] < 0, 'AU_PPB'] = np.nan
        df_processed.loc[df_processed['CU_PPM'] < 0, 'CU_PPM'] = np.nan
        
        if negative_au > 0 or negative_cu > 0:
            st.warning(f"⚠️ Valores negativos convertidos a NaN: Au={negative_au}, Cu={negative_cu}")
        
        # Convertir coordenadas
        df_processed['SAMPLE_LONGITUDE_WGS84'] = pd.to_numeric(df_processed['SAMPLE_LONGITUDE_WGS84'], errors='coerce')
        df_processed['SAMPLE_LATITUDE_WGS84'] = pd.to_numeric(df_processed['SAMPLE_LATITUDE_WGS84'], errors='coerce')
        
        # Convertir fecha
        df_processed['ANALYSIS_DATETIME'] = pd.to_datetime(df_processed['ANALYSIS_DATETIME'], errors='coerce')
        
        # Convertir columnas categóricas
        categorical_cols = [
            'DEPOSIT_GROUP', 'DEPOSIT_TYPE', 'PRIMARY_COMMODITIES', 
            'SAMPLING_METHOD', 'PROVINCE', 'COUNTRY', 'STATE'
        ]
        for col in categorical_cols:
            df_processed[col] = df_processed[col].astype('category')
        
        st.info("✅ Tipos de datos convertidos exitosamente.")
        return df_processed
    except Exception as e:
        st.error(f"❌ Error al convertir tipos de datos: {str(e)}")
        return df


def remove_duplicates(df):
    """
    Elimina registros duplicados del DataFrame.
    """
    try:
        initial_rows = len(df)
        df_clean = df.drop_duplicates()
        removed_rows = initial_rows - len(df_clean)
        
        if removed_rows > 0:
            st.warning(f"⚠️ Se eliminaron {removed_rows} registros duplicados ({removed_rows/initial_rows*100:.2f}%)")
        else:
            st.info("ℹ️ No se encontraron duplicados.")
        
        return df_clean
    except Exception as e:
        st.error(f"❌ Error al eliminar duplicados: {str(e)}")
        return df


def impute_missing_values(df, method='median'):
    """
    Imputa valores faltantes en columnas numéricas según el método seleccionado.
    """
    try:
        df_imputed = df.copy()
        numeric_cols = ['AU_PPB', 'CU_PPM', 'SAMPLE_LONGITUDE_WGS84', 'SAMPLE_LATITUDE_WGS84']
        
        for col in numeric_cols:
            missing_count = df_imputed[col].isna().sum()
            
            if missing_count > 0:
                if method == 'mean':
                    fill_value = df_imputed[col].mean()
                    df_imputed[col].fillna(fill_value, inplace=True)
                    st.info(f"📊 {col}: {missing_count} valores imputados con media ({fill_value:.2f})")
                elif method == 'median':
                    fill_value = df_imputed[col].median()
                    df_imputed[col].fillna(fill_value, inplace=True)
                    st.info(f"📊 {col}: {missing_count} valores imputados con mediana ({fill_value:.2f})")
                elif method == 'zero':
                    df_imputed[col].fillna(0, inplace=True)
                    st.info(f"📊 {col}: {missing_count} valores imputados con cero")
        
        return df_imputed
    except Exception as e:
        st.error(f"❌ Error al imputar valores faltantes: {str(e)}")
        return df


def detect_outliers_iqr(df, column):
    """
    Detecta outliers usando el método IQR (Rango Intercuartílico).
    """
    try:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        return outliers
    except Exception as e:
        st.error(f"❌ Error al detectar outliers en {column}: {str(e)}")
        return pd.Series([False] * len(df))


def handle_outliers(df, treatment='keep'):
    """
    Maneja outliers en columnas geoquímicas según el tratamiento seleccionado.
    """
    try:
        df_processed = df.copy()
        geochemical_cols = ['AU_PPB', 'CU_PPM']
        
        if treatment == 'keep':
            st.info("ℹ️ Los outliers se mantienen sin modificación.")
            return df_processed
        
        for col in geochemical_cols:
            outliers = detect_outliers_iqr(df_processed, col)
            n_outliers = outliers.sum()
            
            if n_outliers > 0:
                if treatment == 'remove':
                    df_processed = df_processed[~outliers]
                    st.warning(f"⚠️ {col}: {n_outliers} outliers eliminados")
                elif treatment == 'winsorize':
                    Q1 = df_processed[col].quantile(0.25)
                    Q3 = df_processed[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    df_processed.loc[df_processed[col] < lower_bound, col] = lower_bound
                    df_processed.loc[df_processed[col] > upper_bound, col] = upper_bound
                    st.info(f"📊 {col}: {n_outliers} outliers winzorizados")
        
        return df_processed
    except Exception as e:
        st.error(f"❌ Error al manejar outliers: {str(e)}")
        return df


def create_calculated_columns(df):
    """
    Versión mejorada: Incluye estadísticas robustas por estado para mitigar
    el sesgo de cantidad de muestras.
    """
    try:
        df_enhanced = df.copy()
        
        # 1. Conversión base
        df_enhanced['AU_PPM'] = df_enhanced['AU_PPB'] / 1000
        
        # 2. Cálculo de Densidad por Estado (Confianza Estadística)
        state_counts = df_enhanced['STATE'].value_counts().to_dict()
        df_enhanced['STATE_SAMPLE_COUNT'] = df_enhanced['STATE'].map(state_counts)
        
        # 3. Normalización Robusta por Estado (Z-Score Local)
        # Esto identifica anomalías relativas a su propio entorno geográfico
        def get_zscore_local(group):
            # Usamos mediana y MAD para evitar sesgo de outliers en estados pequeños
            med = group.median()
            mad = (group - med).abs().median()
            # Evitar división por cero si todas las muestras son iguales
            return (group - med) / (mad if mad != 0 else 1.0)

        df_enhanced['AU_ZSCORE_LOCAL'] = df_enhanced.groupby('STATE')['AU_PPM'].transform(get_zscore_local)
        df_enhanced['CU_ZSCORE_LOCAL'] = df_enhanced.groupby('STATE')['CU_PPM'].transform(get_zscore_local)

        # 4. Cálculo de Percentiles (Tu lógica original se mantiene para no romper el dashboard)
        cu_valid_mask = (df_enhanced['CU_PPM'] > 0) & (df_enhanced['CU_PPM'].notna())
        au_valid_mask = (df_enhanced['AU_PPM'] > 0) & (df_enhanced['AU_PPM'].notna())
        df_enhanced['CU_PERCENTILE'] = np.nan
        df_enhanced['AU_PERCENTILE'] = np.nan
        
        if cu_valid_mask.sum() > 0:
            df_enhanced.loc[cu_valid_mask, 'CU_PERCENTILE'] = df_enhanced.loc[cu_valid_mask, 'CU_PPM'].rank(pct=True)
        if au_valid_mask.sum() > 0:
            df_enhanced.loc[au_valid_mask, 'AU_PERCENTILE'] = df_enhanced.loc[au_valid_mask, 'AU_PPM'].rank(pct=True)

        df_enhanced['CU_AU_PERCENTILE_INDEX'] = df_enhanced['CU_PERCENTILE'] + df_enhanced['AU_PERCENTILE']

        # 5. Clasificación y Flags
        def classify_commodity(row):
            cu, au = row['CU_PPM'] or 0, row['AU_PPM'] or 0
            if cu >= 1000 and au >= 0.1: return 'Polimetálico Cu-Au'
            elif cu >= 1000: return 'Cu dominante'
            elif au >= 0.1: return 'Au dominante'
            else: return 'Baja ley'
        
        df_enhanced['COMMODITY_CLASS'] = df_enhanced.apply(classify_commodity, axis=1)
        df_enhanced['IS_POLYMETALLIC'] = df_enhanced['COMMODITY_CLASS'] == 'Polimetálico Cu-Au'
        df_enhanced['ANALYSIS_YEAR'] = df_enhanced['ANALYSIS_DATETIME'].dt.year
        df_enhanced['HAS_VALID_COORDS'] = df_enhanced['SAMPLE_LONGITUDE_WGS84'].notna() & df_enhanced['SAMPLE_LATITUDE_WGS84'].notna()
        
        st.success("✅ Análisis robusto completado: Se calcularon Z-Scores locales para mitigar sesgo por cantidad de muestras.")
        return df_enhanced
    except Exception as e:
        st.error(f"❌ Error en cálculos: {str(e)}")
        return df

# ==============================================================================
# FUNCIONES DE INTELIGENCIA ARTIFICIAL (GROQ)
# ==============================================================================

def get_groq_client(api_key):
    """
    Inicializa el cliente de Groq API.
    """
    try:
        client = Groq(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"❌ Error al inicializar Groq: {str(e)}")
        return None


def generate_ai_insights(df, api_key, model="llama-3.3-70b-versatile"):
    """
    Genera insights profesionales usando modelos de lenguaje de Groq.
    Versión ajustada para mitigar sesgo de cantidad de muestras.
    """
    try:
        if df.empty:
            return "⚠️ No hay datos disponibles para generar insights."
        
        client = get_groq_client(api_key)
        if client is None:
            return "❌ Error al conectar con Groq API. Verifica tu API key."
        
        # Preparar estadísticas DETALLADAS
        stats_summary = df[['CU_PPM', 'AU_PPM', 'CU_AU_PERCENTILE_INDEX']].describe().to_string()
        total_records = len(df)
        polymetallic_count = df['IS_POLYMETALLIC'].sum() if 'IS_POLYMETALLIC' in df.columns else 0
        polymetallic_pct = (polymetallic_count / total_records * 100) if total_records > 0 else 0
        unique_states = df['STATE'].nunique() if 'STATE.columns' else 0
        avg_cu = df['CU_PPM'].mean() if 'CU_PPM' in df.columns else 0
        avg_au = df['AU_PPM'].mean() if 'AU_PPM' in df.columns else 0

        # === NUEVO: ANÁLISIS DE CONFIANZA POR ESTADO ===
        # Identificamos estados con baja densidad de datos (< 30 muestras es un estándar estadístico común)
        state_counts = df['STATE'].value_counts()
        low_confidence_states = state_counts[state_counts < 30].index.tolist()
        high_confidence_states_count = len(state_counts[state_counts >= 30])

        # Top 10 estados por total de muestras
        top_states_str = "\n".join([f"   {i+1}. {state}: {count:,} muestras" 
                                    for i, (state, count) in enumerate(state_counts.head(10).items())])

        # Concentraciones promedio por estado con conteo de muestras (MODIFICADO)
        # Agregamos el tamaño de la muestra al string para que la IA lo vea
        state_stats = df.groupby('STATE').agg({'CU_PPM': 'mean', 'STATE': 'size'}).rename(columns={'STATE': 'count'})
        state_cu_avg = state_stats.sort_values(by='CU_PPM', ascending=False).head(10) # Subimos a 10 para ver más contexto
        
        cu_by_state_str = "\n".join([
            f"   {i+1}. {state}: {row['CU_PPM']:.2f} ppm Cu (Basado en {int(row['count'])} muestras)" 
            for i, (state, row) in enumerate(state_cu_avg.iterrows())
        ])

        # ... (Mantener lógica de polimetálicos igual) ...
        poly_by_state = df[df['IS_POLYMETALLIC']].groupby('STATE').size().sort_values(ascending=False).head(10)
        poly_states_str = "\n".join([f"   {i+1}. {state}: {count:,} muestras polimetálicas" 
                                     for i, (state, count) in enumerate(poly_by_state.items())])

        poly_cu_avg = df[df['IS_POLYMETALLIC']]['CU_PPM'].mean()
        poly_au_avg = df[df['IS_POLYMETALLIC']]['AU_PPM'].mean()
        non_poly_cu_avg = df[~df['IS_POLYMETALLIC']]['CU_PPM'].mean()
        non_poly_au_avg = df[~df['IS_POLYMETALLIC']]['AU_PPM'].mean()
        
        # === PROMPT ACTUALIZADO CON ADVERTENCIA DE SESGO ===
        prompt = f"""Eres un geólogo experto consultor para el Gobierno de Estados Unidos.

        Analiza estos datos geoquímicos considerando la CONFIANZA ESTADÍSTICA:

        MÉTRICAS CLAVE:
        - Total de muestras: {total_records:,}
        - Estados analizados: {unique_states}
        - Estados con alta confianza (>30 muestras): {high_confidence_states_count}
        - Estados con baja confianza (muestreo limitado): {', '.join(low_confidence_states[:5])}...

        TOP 10 ESTADOS POR DENSIDAD DE MUESTREO (Más confiables):
        {top_states_str}

        LÍDERES EN CONCENTRACIÓN DE Cu (¡Ojo! Verifica el número de muestras):
        {cu_by_state_str}

        TOP 10 ESTADOS POR MUESTRAS POLIMETÁLICAS:
        {poly_states_str}

        DATOS COMPARATIVOS (PPM):
        - Cu promedio en polimetálicos: {poly_cu_avg:.2f} | No polimetálicos: {non_poly_cu_avg:.2f}
        - Au promedio en polimetálicos: {poly_au_avg:.4f} | No polimetálicos: {non_poly_au_avg:.4f}

        PREGUNTAS DE NEGOCIO A RESPONDER:

        1. **¿Qué estados presentan el mayor potencial real de cobre?**
           - Cruza la información de "Concentración" con "Densidad de Muestreo".
           - Prioriza estados que tengan tanto leyes altas como un número representativo de muestras (>30).
           - Advierte si un estado parece rico pero tiene muy pocos datos.

        2. **¿Cuáles son los estados prioritarios para exploración polimetálica Cu-Au?**
           - Identifica dónde hay una coincidencia sólida de ambos metales con base en el volumen de hallazgos.

        3. **¿Cómo distinguir una anomalía real de un error de muestreo?**
           - Explica brevemente cómo los promedios en estados con pocas muestras pueden ser engañosos.

        FORMATO DE RESPUESTA:
        Usa encabezados en negrita y sé muy crítico con la calidad de los datos. Si un estado tiene leyes altas pero pocas muestras, menciónalo como 'Potencial por confirmar'.
        """
        
        # Llamada a la API (se mantiene igual)
        with st.spinner("🤖 Generando insights con IA..."):
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Eres un geólogo experto en exploración que no se deja engañar por sesgos estadísticos. Tu tono es profesional, crítico y técnico."},
                    {"role": "user", "content": prompt}
                ],
                model=model,
                temperature=0.5, # Bajamos la temperatura para que sea más preciso y menos creativo
                max_tokens=2000
            )
            return chat_completion.choices[0].message.content

    except Exception as e:
        return f"❌ Error al generar insights: {str(e)}"


# ==============================================================================
# FUNCIONES DE VISUALIZACIÓN AVANZADA
# ==============================================================================

def create_correlation_heatmap(df):
    """
    Crea un heatmap de correlaciones entre variables geoquímicas.
    Solo incluye Cu y Au (ambas en PPM) para garantizar comparabilidad.
    """
    try:
        # Solo variables geoquímicas en PPM e índice polimetálico
        relevant_cols = []

        if 'CU_PPM' in df.columns:
            relevant_cols.append('CU_PPM')
        if 'AU_PPM' in df.columns:
            relevant_cols.append('AU_PPM')
        if 'CU_AU_PERCENTILE_INDEX' in df.columns:
            relevant_cols.append('CU_AU_PERCENTILE_INDEX')        

        if len(relevant_cols) < 2:
            st.warning("⚠️ No hay suficientes variables para calcular correlaciones")
            return None
        
        # Filtrar solo filas con valores válidos (no NaN, no negativos)
        df_clean = df[relevant_cols].dropna()
        
        if len(df_clean) < 2:
            st.warning("⚠️ No hay suficientes datos válidos para correlaciones")
            return None
        
        corr_matrix = df_clean.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix.values, 3),
            texttemplate='%{text}',
            textfont={"size": 12},
            colorbar=dict(title="Correlación")
        ))
        
        fig.update_layout(
            title='Matriz de Correlación - Variables Geoquímicas (PPM)',
            xaxis_title='Variables',
            yaxis_title='Variables',
            height=500,
            width=600
        )
        
        return fig
    except Exception as e:
        st.error(f"❌ Error al crear heatmap: {str(e)}")
        return None


def create_temporal_analysis(df):
    """
    Crea gráficos de evolución temporal de concentraciones.
    Todas las concentraciones en PPM para comparabilidad.
    """
    try:
        if 'ANALYSIS_YEAR' not in df.columns or df['ANALYSIS_YEAR'].isna().all():
            return None
        
        yearly_stats = df.groupby('ANALYSIS_YEAR').agg({
            'CU_PPM': ['mean', 'count'],
            'AU_PPM': 'mean',  # Usar AU_PPM en vez de AU_PPB
            'IS_POLYMETALLIC': 'sum'
        }).reset_index()
        
        yearly_stats.columns = ['Year', 'Cu_Mean', 'Sample_Count', 'Au_Mean', 'Polymetallic_Count']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Evolución Promedio Cu (ppm)', 'Evolución Promedio Au (ppm)',
                          'Número de Muestras por Año', 'Muestras Polimetálicas por Año')
        )
        
        fig.add_trace(
            go.Scatter(x=yearly_stats['Year'], y=yearly_stats['Cu_Mean'],
                      mode='lines+markers', name='Cu', line=dict(color='#FF6B6B', width=3),
                      marker=dict(size=8)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=yearly_stats['Year'], y=yearly_stats['Au_Mean'],
                      mode='lines+markers', name='Au', line=dict(color='#FFD93D', width=3),
                      marker=dict(size=8)),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=yearly_stats['Year'], y=yearly_stats['Sample_Count'],
                  name='Muestras', marker_color='#4ECDC4'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(x=yearly_stats['Year'], y=yearly_stats['Polymetallic_Count'],
                  name='Polimetálicos', marker_color='#95E1D3'),
            row=2, col=2
        )
        
        fig.update_layout(height=700, showlegend=False, title_text="Análisis de Evolución Temporal (Concentraciones en PPM)")
        return fig
    except Exception as e:
        st.error(f"❌ Error al crear análisis temporal: {str(e)}")
        return None


def create_geospatial_map(df, color_by='CU_PPM'):
    """
    Crea un mapa interactivo con los depósitos georreferenciados.
    Todas las concentraciones en PPM.
    
    IMPORTANTE: Solo muestra muestras con coordenadas válidas.
    
    Args:
        df: DataFrame con los datos
        color_by: Variable para colorear los puntos ('CU_PPM' o 'CU_AU_RATIO')
    """
    try:
        # Filtrar solo muestras con coordenadas válidas
        df_map = df[df['HAS_VALID_COORDS'] == True].copy()
                
        # Calcular estadísticas de cobertura geográfica
        total_samples = len(df)
        samples_with_coords = len(df_map)
        samples_without_coords = total_samples - samples_with_coords
        coverage_pct = (samples_with_coords / total_samples * 100) if total_samples > 0 else 0
        
        if len(df_map) == 0:
            st.warning(f"⚠️ No hay muestras disponibles para mostrar en el mapa con los criterios seleccionados.")
            return None
        
        # Configurar título y escala de color según la variable seleccionada
        if color_by == 'CU_PPM':
            title_text = f'Distribución Geográfica - Contenido de Cu (ppm) ({samples_with_coords:,} muestras)'
            color_label = 'Cu (ppm)'
            color_scale = 'Viridis'
        else:  # CU_AU_PERCENTILE_INDEX
            title_text = f'Distribución Geográfica - Índice Polimetálico ({samples_with_coords:,} muestras)'
            color_label = 'Índice (0-2)'
            color_scale = 'RdYlGn'  # Rojo-Amarillo-Verde para índice            

        fig = px.scatter_mapbox(
            df_map,
            lat='SAMPLE_LATITUDE_WGS84',
            lon='SAMPLE_LONGITUDE_WGS84',
            color=color_by,
            size='AU_PPM',
            hover_data={
                'SAMPLE_LATITUDE_WGS84': False,
                'SAMPLE_LONGITUDE_WGS84': False,
                'CU_PPM': ':.2f',
                'AU_PPM': ':.4f',
                'CU_AU_PERCENTILE_INDEX': ':.3f',
                'STATE': True,
                'DEPOSIT_TYPE': True,
                'COMMODITY_CLASS': True
            },
            color_continuous_scale=color_scale,
            size_max=15,
            zoom=3,
            mapbox_style='open-street-map',
            title=title_text,
            labels={color_by: color_label}
        )
        
        fig.update_layout(height=600, margin={"r":0,"t":40,"l":0,"b":0})
        return fig
    except Exception as e:
        st.error(f"❌ Error al crear mapa: {str(e)}")
        return None


def create_state_analysis(df):
    """
    Crea análisis de muestras por estado con barras apiladas por categoría.
    Todas las concentraciones en PPM.
    Muestra los estados que tienen datos en el DataFrame filtrado.
    """
    try:
        if 'STATE' not in df.columns or 'COMMODITY_CLASS' not in df.columns:
            return None
        
        # Agrupar por estado y categoría
        state_category_counts = df.groupby(['STATE', 'COMMODITY_CLASS']).size().reset_index(name='Count')
        
        # Pivotar para obtener columnas por categoría
        state_pivot = state_category_counts.pivot(index='STATE', columns='COMMODITY_CLASS', values='Count').fillna(0)
        
        # Calcular total por estado para ordenar
        state_pivot['Total'] = state_pivot.sum(axis=1)
        state_pivot = state_pivot.sort_values('Total', ascending=False)
        
        # Limitar a top 20 para mejor visualización
        if len(state_pivot) > 20:
            state_pivot = state_pivot.head(20)
            title_suffix = f' (Top 20 de {df["STATE"].nunique()} estados)'
        else:
            title_suffix = f' ({len(state_pivot)} estados)'
        
        # Eliminar columna Total para el gráfico
        state_pivot = state_pivot.drop('Total', axis=1)
        
        # Crear gráfico de barras apiladas
        fig = go.Figure()
        
        # Definir colores para cada categoría
        category_colors = {
            'Baja ley': '#95A5A6',
            'Cu dominante': '#E74C3C',
            'Au dominante': '#F39C12',
            'Polimetálico Cu-Au': '#27AE60'
        }
        
        # Agregar una traza por cada categoría
        for category in state_pivot.columns:
            if category in state_pivot.columns:
                fig.add_trace(go.Bar(
                    y=state_pivot.index,
                    x=state_pivot[category],
                    name=category,
                    orientation='h',
                    marker=dict(color=category_colors.get(category, '#3498DB')),
                    text=state_pivot[category].astype(int),
                    textposition='inside'
                ))
        
        fig.update_layout(
            title=f'Número de Muestras por Estado y Categoría{title_suffix}',
            xaxis_title='Cantidad de Muestras',
            yaxis_title='Estado',
            barmode='stack',
            height=max(400, len(state_pivot) * 25),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    except Exception as e:
        st.error(f"❌ Error al crear análisis por estado: {str(e)}")
        return None


def create_distribution_plots(df, variable):
    """
    Crea histogramas y boxplots para análisis univariado.
    """
    try:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f'Histograma: {variable}', f'Boxplot: {variable}'),
            specs=[[{"type": "histogram"}, {"type": "box"}]]
        )
        
        fig.add_trace(
            go.Histogram(x=df[variable], name='Frecuencia',
                        marker_color='#3498DB', nbinsx=50),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Box(y=df[variable], name=variable,
                   marker_color='#E74C3C', boxmean='sd'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False, title_text=f"Análisis de Distribución: {variable}")
        return fig
    except Exception as e:
        st.error(f"❌ Error al crear gráficos de distribución: {str(e)}")
        return None


def create_scatter_analysis(df, x_var, y_var, color_by=None):
    """
    Crea gráfico de dispersión para análisis bivariado.
    """
    try:
        if color_by and color_by in df.columns:
            fig = px.scatter(
                df, x=x_var, y=y_var, color=color_by,
                title=f'{y_var} vs {x_var} (coloreado por {color_by})',
                trendline='ols',
                labels={x_var: x_var, y_var: y_var},
                height=500
            )
        else:
            fig = px.scatter(
                df, x=x_var, y=y_var,
                title=f'{y_var} vs {x_var}',
                trendline='ols',
                labels={x_var: x_var, y_var: y_var},
                height=500
            )
        
        fig.update_traces(marker=dict(size=8, opacity=0.6))
        return fig
    except Exception as e:
        st.error(f"❌ Error al crear scatter plot: {str(e)}")
        return None


def show_missing_values_chart(df):
    """
    Muestra gráfico de valores faltantes por columna.
    """
    st.subheader("❌ Valores Faltantes por Columna")
    
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Columna': missing_data.index,
        'Valores Faltantes': missing_data.values,
        'Porcentaje': missing_percent.values
    })
    
    missing_df = missing_df[missing_df['Valores Faltantes'] > 0].sort_values('Valores Faltantes', ascending=False)
    
    if len(missing_df) > 0:
        fig = px.bar(
            missing_df,
            x='Columna',
            y='Porcentaje',
            text='Valores Faltantes',
            title='Porcentaje de Valores Faltantes por Columna',
            labels={'Porcentaje': 'Porcentaje (%)'},
            color='Porcentaje',
            color_continuous_scale='Reds'
        )
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        fig.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅ No hay valores faltantes en el dataset.")


def show_outliers_boxplot(df):
    """
    Muestra boxplots para detectar visualmente outliers.
    """
    st.subheader("📊 Detección Visual de Outliers (Boxplots)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'CU_PPM' in df.columns and df['CU_PPM'].notna().sum() > 0:
            fig_cu = px.box(df, y='CU_PPM', title='Distribución de Cobre (Cu) en PPM')
            fig_cu.update_layout(height=400)
            st.plotly_chart(fig_cu, use_container_width=True)
    
    with col2:
        if 'AU_PPB' in df.columns and df['AU_PPB'].notna().sum() > 0:
            fig_au = px.box(df, y='AU_PPB', title='Distribución de Oro (Au) en PPB')
            fig_au.update_layout(height=400)
            st.plotly_chart(fig_au, use_container_width=True)


# ==============================================================================
# GENERACIÓN DE REPORTES PDF
# ==============================================================================

def generate_pdf_report(df, insights_text=""):
    """
    Genera un reporte PDF orientado a preguntas de negocio.
    Requiere: reportlab
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        story = []
        styles = getSampleStyleSheet()
        
        # Estilos personalizados
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=22,
            textColor=colors.HexColor('#003DA5'),
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#003DA5'),
            spaceAfter=10,
            spaceBefore=15,
            fontName='Helvetica-Bold'
        )
        
        subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=styles['Heading3'],
            fontSize=12,
            textColor=colors.HexColor('#2C3E50'),
            spaceAfter=8,
            spaceBefore=10,
            fontName='Helvetica-Bold'
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_JUSTIFY,
            spaceAfter=8
        )
        
        # ==================== PORTADA ====================
        story.append(Spacer(1, 1*inch))
        story.append(Paragraph("REPORTE ESTRATÉGICO", title_style))
        story.append(Paragraph("Análisis de Potencial Cu-Au en Estados Unidos", heading_style))
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph(f"Fecha: {datetime.now().strftime('%d de %B de %Y')}", styles['Normal']))
        story.append(Paragraph("Consultoría para Transición Energética - Gobierno de EE.UU.", styles['Normal']))
        story.append(Spacer(1, 0.5*inch))
        
        # Resumen ejecutivo en portada
        story.append(Paragraph("RESUMEN EJECUTIVO", subheading_style))
        
        summary_data = [
            ["Métrica", "Valor"],
            ["Total de muestras analizadas", f"{len(df):,}"],
            ["Muestras polimetálicas Cu-Au", f"{df['IS_POLYMETALLIC'].sum():,} ({df['IS_POLYMETALLIC'].sum()/len(df)*100:.1f}%)"],
            ["Estados analizados", f"{df['STATE'].nunique()}"],
            ["Concentración promedio Cu", f"{df['CU_PPM'].mean():.2f} ppm"],
            ["Concentración promedio Au", f"{df['AU_PPM'].mean():.4f} ppm"],
            ["Índice Polimetálico promedio", f"{df['CU_AU_PERCENTILE_INDEX'].mean():.3f}"]
        ]
        
        t = Table(summary_data, colWidths=[3.5*inch, 2*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003DA5')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        
        story.append(t)
        story.append(PageBreak())
        
        # ==================== PREGUNTA 1: ESTADOS CON MAYOR POTENCIAL ====================
        story.append(Paragraph("PREGUNTA 1: ¿Qué Estados Presentan Mayor Concentración de Cobre?", heading_style))

        # Tabla de top 15 estados POR CONCENTRACIÓN PROMEDIO DE CU
        cu_concentration = df.groupby('STATE').agg({
            'CU_PPM': 'mean',
            'SAMPLE_UID': 'count'
        }).sort_values('CU_PPM', ascending=False).head(15)

        cu_concentration.columns = ['Cu_Promedio', 'Total_Muestras']
        cu_concentration = cu_concentration.reset_index()

        table_data = [['Ranking', 'Estado', 'Cu Promedio (ppm)', 'N° Muestras']]
        for idx, row in enumerate(cu_concentration.itertuples(), 1):
            table_data.append([
                str(idx),
                row.STATE,
                f"{row.Cu_Promedio:.2f}",
                f"{row.Total_Muestras:,}"
            ])

        t2 = Table(table_data, colWidths=[0.8*inch, 2.5*inch, 1.8*inch, 1.5*inch])
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003DA5')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))

        story.append(t2)
        story.append(Spacer(1, 0.2*inch))

        # Interpretación breve
        interpretation = f"""
        Los estados listados presentan las concentraciones promedio de cobre más altas del análisis. 
        {cu_concentration.iloc[0]['STATE']} lidera con {cu_concentration.iloc[0]['Cu_Promedio']:.2f} ppm, 
        seguido por {cu_concentration.iloc[1]['STATE']} ({cu_concentration.iloc[1]['Cu_Promedio']:.2f} ppm) 
        y {cu_concentration.iloc[2]['STATE']} ({cu_concentration.iloc[2]['Cu_Promedio']:.2f} ppm).
        """
        story.append(Paragraph(interpretation, body_style))

        story.append(PageBreak())
        
        # ==================== PREGUNTA 2: ESTADOS PRIORITARIOS POLIMETÁLICOS ====================
        story.append(Paragraph("PREGUNTA 2: Estados Prioritarios para Sistemas Polimetálicos Cu-Au", heading_style))
        
        # Tabla de estados polimetálicos
        poly_stats = df[df['IS_POLYMETALLIC']].groupby('STATE').agg({
            'SAMPLE_UID': 'count',
            'CU_PPM': 'mean',
            'AU_PPM': 'mean',
            'CU_AU_PERCENTILE_INDEX': 'mean'
        }).reset_index()
        poly_stats.columns = ['Estado', 'Muestras_Poly', 'Cu_Prom', 'Au_Prom', 'Indice_Prom']
        poly_stats = poly_stats.sort_values('Muestras_Poly', ascending=False).head(10)
        
        table_data2 = [['Rank', 'Estado', 'Muestras\nPolimetálicas', 'Cu Prom\n(ppm)', 'Au Prom\n(ppm)', 'Índice\n(0-2)']]
        for idx, row in enumerate(poly_stats.itertuples(), 1):
            table_data2.append([
                str(idx),
                row.Estado,
                f"{row.Muestras_Poly:,}",
                f"{row.Cu_Prom:.2f}",
                f"{row.Au_Prom:.4f}",
                f"{row.Indice_Prom:.3f}"
            ])
            
        t3 = Table(table_data2, colWidths=[0.5*inch, 1.5*inch, 1.2*inch, 1*inch, 1*inch, 1*inch])
        t3.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27AE60')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
        ]))
        
        story.append(t3)
        story.append(Spacer(1, 0.3*inch))

        # Interpretación breve
        poly_interpretation = f"""
        Los estados prioritarios para exploración de sistemas polimetálicos Cu-Au se determinaron 
        por el número absoluto de muestras polimetálicas identificadas. {poly_stats.iloc[0]['Estado']} 
        encabeza el ranking con {poly_stats.iloc[0]['Muestras_Poly']:,} muestras polimetálicas, 
        presentando concentraciones promedio de {poly_stats.iloc[0]['Cu_Prom']:.2f} ppm Cu y 
        {poly_stats.iloc[0]['Au_Prom']:.4f} ppm Au.
        """
        story.append(Paragraph(poly_interpretation, body_style))

        story.append(PageBreak())
        
        # ==================== PREGUNTA 3: CARACTERÍSTICAS POLIMETÁLICOS ====================
        story.append(Paragraph("PREGUNTA 3: Características de Muestras Polimetálicas de Alto Valor", heading_style))
        
        # Estadísticas comparativas
        poly_df = df[df['IS_POLYMETALLIC']]
        non_poly_df = df[~df['IS_POLYMETALLIC']]
        
        comparison_data = [
            ["Característica", "Polimetálicas", "No Polimetálicas", "Diferencia"],
            ["Concentración Cu (ppm)", 
             f"{poly_df['CU_PPM'].mean():.2f}", 
             f"{non_poly_df['CU_PPM'].mean():.2f}",
             f"+{(poly_df['CU_PPM'].mean() - non_poly_df['CU_PPM'].mean()):.2f}"],
            ["Concentración Au (ppm)", 
             f"{poly_df['AU_PPM'].mean():.4f}", 
             f"{non_poly_df['AU_PPM'].mean():.4f}",
             f"+{(poly_df['AU_PPM'].mean() - non_poly_df['AU_PPM'].mean()):.4f}"],
            ["Índice Polimetálico (0-2)", 
             f"{poly_df['CU_AU_PERCENTILE_INDEX'].mean():.3f}", 
             f"{non_poly_df['CU_AU_PERCENTILE_INDEX'].mean():.3f}",
             f"{(poly_df['CU_AU_PERCENTILE_INDEX'].mean() - non_poly_df['CU_AU_PERCENTILE_INDEX'].mean()):+.3f}"],
            ["Cantidad de muestras", 
             f"{len(poly_df):,}", 
             f"{len(non_poly_df):,}",
             f"{len(poly_df)/len(df)*100:.1f}%"]
        ]
        
        t4 = Table(comparison_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.2*inch])
        t4.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E74C3C')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        
        story.append(t4)
        story.append(Spacer(1, 0.3*inch))

        # Interpretación de características
        characteristics_text = f"""
        Las muestras polimetálicas de alto valor se caracterizan por:

        - Concentraciones de Cu significativamente superiores: +{(poly_df['CU_PPM'].mean() - non_poly_df['CU_PPM'].mean()):.2f} ppm 
          respecto a muestras no polimetálicas.

        - Concentraciones de Au elevadas: +{(poly_df['AU_PPM'].mean() - non_poly_df['AU_PPM'].mean()):.4f} ppm 
          en promedio.

        - Índice Polimetálico superior a {poly_df['CU_AU_PERCENTILE_INDEX'].mean():.2f}, indicando 
          que ambos metales se encuentran en percentiles altos simultáneamente.

        Estas características distintivas permiten priorizar áreas con mayor potencial económico 
        para desarrollos polimetálicos Cu-Au.
        """
        story.append(Paragraph(characteristics_text, body_style))

        story.append(PageBreak())
        
        # ==================== INSIGHTS DE IA ====================
        if insights_text:
            story.append(Paragraph("ANÁLISIS CON INTELIGENCIA ARTIFICIAL", heading_style))
            story.append(Paragraph("Insights generados mediante modelos de lenguaje avanzados (Groq API)", styles['Italic']))
            story.append(Spacer(1, 0.1*inch))
            
            # Procesar insights línea por línea
            for line in insights_text.split('\n'):
                line = line.strip()
                if line:
                    if line.startswith('**') and line.endswith('**'):
                        # Es un título
                        clean_line = line.replace('**', '')
                        story.append(Paragraph(clean_line, subheading_style))
                    elif line.startswith('-'):
                        # Es una viñeta
                        story.append(Paragraph(line, body_style))
                    else:
                        # Texto normal
                        story.append(Paragraph(line, body_style))
            
            story.append(PageBreak())
        
        # ==================== ESTADÍSTICAS DETALLADAS ====================
        story.append(Paragraph("ESTADÍSTICAS DESCRIPTIVAS COMPLETAS", heading_style))
        
        desc_stats = df[['CU_PPM', 'AU_PPM', 'CU_AU_PERCENTILE_INDEX']].describe()

        stats_data = [['Estadística', 'Cu (ppm)', 'Au (ppm)', 'Índice (0-2)']]

        for idx in desc_stats.index:
            row = [idx] + [f"{val:.4f}" if 'AU_PPM' in desc_stats.columns[i] else f"{val:.2f}" 
                          for i, val in enumerate(desc_stats.loc[idx])]
            stats_data.append(row)
        
        t5 = Table(stats_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        t5.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003DA5')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        
        story.append(t5)
        story.append(Spacer(1, 0.3*inch))
        
        # Nota final
        story.append(Paragraph("___", styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(
            f"Reporte generado el {datetime.now().strftime('%d/%m/%Y a las %H:%M')} | "
            f"Dashboard Cu-Au v2.0 | EAFIT - Ciencia de Datos",
            ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.grey)
        ))
        
        # Construir PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
        
    except ImportError as ie:
        st.warning(f"⚠️ Falta instalar librerías: {str(ie)}")
        st.info("Ejecuta: pip install reportlab")
        return None
    except Exception as e:
        st.error(f"❌ Error al generar PDF: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None


# ==============================================================================
# APLICACIÓN DE FILTROS GLOBALES
# ==============================================================================

def apply_filters(df, date_range=None, countries=None, states=None, deposit_groups=None,
                 cu_range=None, au_range=None, commodity_classes=None):
    """
    Aplica filtros globales al DataFrame.
    
    Args:
        df: DataFrame a filtrar
        date_range: Tupla (fecha_inicio, fecha_fin)
        countries: Lista de países
        states: Lista de estados
        deposit_groups: Lista de grupos de depósito (DEPOSIT_GROUP)
        cu_range: Tupla (cu_min, cu_max) en PPM
        au_range: Tupla (au_min, au_max) en PPM
        commodity_classes: Lista de clasificaciones
    
    Returns:
        DataFrame filtrado
    """
    df_filtered = df.copy()
    
    # Filtro de fechas
    if date_range and len(date_range) == 2:
        df_filtered = df_filtered[
            (df_filtered['ANALYSIS_DATETIME'].dt.date >= date_range[0]) &
            (df_filtered['ANALYSIS_DATETIME'].dt.date <= date_range[1])
        ]
    
    # Filtro de países
    if countries and len(countries) > 0:
        df_filtered = df_filtered[df_filtered['COUNTRY'].isin(countries)]
    
    # Filtro de estados
    if states and len(states) > 0:
        df_filtered = df_filtered[df_filtered['STATE'].isin(states)]
    
    # Filtro de grupos de depósito
    if deposit_groups and len(deposit_groups) > 0:
        df_filtered = df_filtered[df_filtered['DEPOSIT_GROUP'].isin(deposit_groups)]
    
    # Filtro de Cu (en PPM)
    if cu_range:
        df_filtered = df_filtered[
            (df_filtered['CU_PPM'] >= cu_range[0]) &
            (df_filtered['CU_PPM'] <= cu_range[1])
        ]
    
    # Filtro de Au (en PPM, no PPB)
    if au_range:
        # Convertir el rango de AU_PPM (no AU_PPB)
        df_filtered = df_filtered[
            (df_filtered['AU_PPM'] >= au_range[0]) &
            (df_filtered['AU_PPM'] <= au_range[1])
        ]
    
    # Filtro de clasificación
    if commodity_classes and len(commodity_classes) > 0:
        df_filtered = df_filtered[df_filtered['COMMODITY_CLASS'].isin(commodity_classes)]
    
    return df_filtered


# ==============================================================================
# INTERFAZ PRINCIPAL DE LA APLICACIÓN
# ==============================================================================

def main():
    """
    Función principal que ejecuta la aplicación Streamlit.
    """
    
    # Título y descripción
    st.title("⛏️ Dashboard de Análisis: Potencial Cu-Au en Estados Unidos")
    st.markdown("""
    ### Contexto del Proyecto
    Herramienta analítica para el **Gobierno de Estados Unidos** en el marco de la **transición energética**.
    Identifica estados con mayor potencial para depósitos de **cobre** y sistemas **polimetálicos Cu-Au**.
    
    ---
    """)
    
    # ==============================================================================
    # SIDEBAR: CARGA Y PROCESAMIENTO
    # ==============================================================================
    
    st.sidebar.header("⚙️ Configuración de Datos")
    
    # Método de carga
    st.sidebar.subheader("1️⃣ Método de Carga")
    load_method = st.sidebar.radio(
        "Selecciona el método de carga:",
        ["📁 Cargar archivo CSV", "📄 Cargar archivo JSON", "🌐 Cargar desde URL"],
        index=0
    )
    
    df_raw = None
    
    # Cargar datos según método seleccionado
    if load_method == "📁 Cargar archivo CSV":
        uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type=['csv'])
        if uploaded_file is not None:
            df_raw = load_csv_file(uploaded_file)
    
    elif load_method == "📄 Cargar archivo JSON":
        uploaded_file = st.sidebar.file_uploader("Sube tu archivo JSON", type=['json'])
        if uploaded_file is not None:
            df_raw = load_json_file(uploaded_file)
    
    elif load_method == "🌐 Cargar desde URL":
        url_input = st.sidebar.text_input("Ingresa la URL del archivo:", placeholder="https://ejemplo.com/datos.csv")
        if st.sidebar.button("🔄 Cargar datos desde URL"):
            if url_input:
                df_raw = load_from_url(url_input)
    
    # ==============================================================================
    # PROCESAMIENTO
    # ==============================================================================
    
    if df_raw is not None:
        st.sidebar.markdown("---")
        st.sidebar.subheader("2️⃣ Procesamiento de Datos")
        
        # Convertir tipos
        df_processed = convert_data_types(df_raw)
        
        # Eliminar duplicados
        remove_dupes = st.sidebar.checkbox("🗑️ Eliminar registros duplicados", value=True)
        if remove_dupes:
            df_processed = remove_duplicates(df_processed)
        
        # Imputación
        imputation_method = st.sidebar.selectbox(
            "🔢 Método de imputación:",
            ["No imputar", "Media", "Mediana", "Cero"],
            index=2
        )
        if imputation_method != "No imputar":
            method_map = {"Media": "mean", "Mediana": "median", "Cero": "zero"}
            df_processed = impute_missing_values(df_processed, method=method_map[imputation_method])
        
        # Outliers
        outlier_treatment = st.sidebar.selectbox(
            "📉 Tratamiento de outliers:",
            ["Mantener", "Eliminar", "Winsorizar"],
            index=0
        )
        treatment_map = {"Mantener": "keep", "Eliminar": "remove", "Winsorizar": "winsorize"}
        df_processed = handle_outliers(df_processed, treatment=treatment_map[outlier_treatment])
        
        # Crear columnas calculadas
        df_final = create_calculated_columns(df_processed)
        st.session_state['df_final'] = df_final
        
        # ==============================================================================
        # FILTROS GLOBALES
        # ==============================================================================
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎯 Filtros Globales")
        
        # Groq API Key
        groq_api_key = st.sidebar.text_input(
            "🔑 Groq API Key",
            type="password",
            help="Obtén tu clave en https://console.groq.com/"
        )
        
        # Filtro de fechas
        date_filter = None
        if 'ANALYSIS_DATETIME' in df_final.columns and not df_final['ANALYSIS_DATETIME'].isna().all():
            min_date = df_final['ANALYSIS_DATETIME'].min().date()
            max_date = df_final['ANALYSIS_DATETIME'].max().date()
            date_filter = st.sidebar.date_input(
                "📅 Rango de Fechas",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
        
        # Filtro de países
        countries_filter = st.sidebar.multiselect(
            "🌍 País",
            options=sorted(df_final['COUNTRY'].dropna().unique()),
            default=[],
            help="Selecciona uno o más países (vacío = todos)"
        )
        
        # Filtro de estados
        states_filter = st.sidebar.multiselect(
            "🗺️ Estados",
            options=sorted(df_final['STATE'].dropna().unique()),
            default=[],
            help="Selecciona uno o más estados (vacío = todos)"
        )
        
        # Filtro de grupos de depósito
        deposit_groups_filter = st.sidebar.multiselect(
            "⛏️ Grupo Depósito",
            options=sorted(df_final['DEPOSIT_GROUP'].dropna().unique()),
            default=[],
            help="Selecciona grupos de depósito (vacío = todos)"
        )
        
        # Filtro de clasificación commodity
        commodity_filter = None
        if 'COMMODITY_CLASS' in df_final.columns:
            commodity_filter = st.sidebar.multiselect(
                "💎 Clasificación Commodity",
                options=sorted(df_final['COMMODITY_CLASS'].dropna().unique()),
                default=[],
                help="Filtra por tipo de mineralización"
            )
        
        # Slider de Cu (PPM)
        cu_min = float(df_final['CU_PPM'].min())
        cu_max = float(df_final['CU_PPM'].max())
        
        # Manejar caso donde min == max (puede ocurrir después de winsorizar)
        if cu_min == cu_max:
            st.sidebar.warning(f"⚠️ Cu tiene un solo valor: {cu_min:.2f} ppm")
            cu_filter = (cu_min, cu_max)
        else:
            cu_filter = st.sidebar.slider(
                "🔶 Rango de Cu (ppm)",
                min_value=cu_min,
                max_value=cu_max,
                value=(cu_min, cu_max),
                format="%.2f",
                help="Concentración de cobre en partes por millón"
            )
        
        # Slider de Au (PPM, no PPB)
        au_ppm_min = float(df_final['AU_PPM'].min())
        au_ppm_max = float(df_final['AU_PPM'].max())
        
        # Manejar caso donde min == max (puede ocurrir después de winsorizar)
        if au_ppm_min == au_ppm_max:
            st.sidebar.warning(f"⚠️ Au tiene un solo valor: {au_ppm_min:.4f} ppm")
            au_filter = (au_ppm_min, au_ppm_max)
        else:
            au_filter = st.sidebar.slider(
                "🔸 Rango de Au (ppm)",
                min_value=au_ppm_min,
                max_value=au_ppm_max,
                value=(au_ppm_min, au_ppm_max),
                format="%.4f",
                help="Concentración de oro en partes por millón"
            )
        
        # Aplicar filtros
        df_filtered = apply_filters(
            df_final,
            date_range=date_filter if date_filter and len(date_filter) == 2 else None,
            countries=countries_filter if len(countries_filter) > 0 else None,
            states=states_filter if len(states_filter) > 0 else None,
            deposit_groups=deposit_groups_filter if len(deposit_groups_filter) > 0 else None,
            cu_range=cu_filter,
            au_range=au_filter,
            commodity_classes=commodity_filter if commodity_filter and len(commodity_filter) > 0 else None
        )
        
        # Mostrar conteo
        st.sidebar.markdown("---")
        st.sidebar.metric("📊 Registros Filtrados", f"{len(df_filtered):,} / {len(df_final):,}")
        
        # ==============================================================================
        # TABS CON ANÁLISIS
        # ==============================================================================
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "⚙️ Procesamiento",
            "📊 Univariado",
            "🔗 Bivariado",
            "🗺️ Geoespacial",
            "🤖 Insights IA",
            "📄 Reporte PDF"
        ])
        
        # TAB 1: Procesamiento
        with tab1:
            st.header("⚙️ Procesamiento y Calidad de Datos")
            
            # Métricas
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Muestras", f"{len(df_filtered):,}")
            with col2:
                st.metric("Estados", df_filtered['STATE'].nunique())
            with col3:
                st.metric("Polimetálicos", df_filtered['IS_POLYMETALLIC'].sum())
            with col4:
                st.metric("Cu Promedio", f"{df_filtered['CU_PPM'].mean():.2f} ppm")
            
            st.markdown("---")
            
            # Vista previa
            with st.expander("📋 Ver Datos Crudos"):
                st.dataframe(df_filtered.head(50), use_container_width=True)
            
            # Descarga
            st.download_button(
                label="💾 Descargar datos filtrados (CSV)",
                data=df_filtered.to_csv(index=False).encode('utf-8'),
                file_name=f'datos_filtrados_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv'
            )
            
            st.markdown("---")
            show_missing_values_chart(df_filtered)
            st.markdown("---")
            show_outliers_boxplot(df_filtered)
        
        # TAB 2: Univariado
        with tab2:
            st.header("📊 Análisis Univariado")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Distribución de Cobre (Cu)")
                fig_cu = create_distribution_plots(df_filtered, 'CU_PPM')
                if fig_cu:
                    st.plotly_chart(fig_cu, use_container_width=True)
            
            with col2:
                st.subheader("Distribución de Oro (Au) en PPM")
                fig_au = create_distribution_plots(df_filtered, 'AU_PPM')
                if fig_au:
                    st.plotly_chart(fig_au, use_container_width=True)
            
            st.markdown("---")
            
            # Distribución por commodity
            if 'COMMODITY_CLASS' in df_filtered.columns:
                st.subheader("Distribución por Clasificación de Muestras")
                commodity_counts = df_filtered['COMMODITY_CLASS'].value_counts()
                fig_pie = px.pie(
                    values=commodity_counts.values,
                    names=commodity_counts.index,
                    title='Clasificación de Muestras por Contenido Metálico',
                    hole=0.4
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            st.markdown("---")
            
            # Análisis por estado
            st.subheader("Análisis por Estado")
            
            # Nota informativa sobre filtros
            if len(df_filtered) < len(df_final):
                st.info(f"📊 Mostrando análisis para {df_filtered['STATE'].nunique()} estados únicos en las {len(df_filtered):,} muestras filtradas")
            
            # NUEVA TABLA: Top estados por concentración promedio de Cu
            st.subheader("🏆 Top 15 Estados por Concentración Promedio de Cobre")

            # Calcular concentración promedio de Cu por estado
            cu_by_state = df_filtered.groupby('STATE').agg({
                'CU_PPM': 'mean',
                'SAMPLE_UID': 'count'
            }).sort_values('CU_PPM', ascending=False).head(15)

            cu_by_state.columns = ['Cu Promedio (ppm)', 'Número de Muestras']
            cu_by_state = cu_by_state.reset_index()
            cu_by_state.index = range(1, len(cu_by_state) + 1)  # Ranking 1, 2, 3...

            # Formatear la tabla
            cu_by_state_display = cu_by_state.copy()
            cu_by_state_display['Cu Promedio (ppm)'] = cu_by_state_display['Cu Promedio (ppm)'].apply(lambda x: f"{x:.2f}")
            cu_by_state_display['Número de Muestras'] = cu_by_state_display['Número de Muestras'].apply(lambda x: f"{x:,}")

            st.dataframe(
                cu_by_state_display,
                use_container_width=True,
                height=400
            )

            st.markdown("---")

            fig_states = create_state_analysis(df_filtered)
            if fig_states:
                st.plotly_chart(fig_states, use_container_width=True)
            
            # Estadísticas (todo en PPM)
            with st.expander("📈 Ver Estadísticas Descriptivas (Concentraciones en PPM)"):
                st.dataframe(df_filtered[['CU_PPM', 'AU_PPM', 'CU_AU_PERCENTILE_INDEX']].describe(), use_container_width=True)
                st.caption("💡 Índice Polimetálico: 0-2 (>1.5 = alta calidad, >1.8 = excepcional)")
                
        # TAB 3: Bivariado
        with tab3:
            st.header("🔗 Análisis Bivariado")
            
            # Heatmap
            st.subheader("Matriz de Correlación")
            fig_corr = create_correlation_heatmap(df_filtered)
            if fig_corr:
                st.plotly_chart(fig_corr, use_container_width=True)
            
            st.markdown("---")
            
            # Scatter plots
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Cu vs Au (ambos en PPM)")
                fig_scatter = create_scatter_analysis(df_filtered, 'AU_PPM', 'CU_PPM', 'COMMODITY_CLASS')
                if fig_scatter:
                    st.plotly_chart(fig_scatter, use_container_width=True)
            
            with col2:
                st.subheader("Índice Polimetálico (Percentiles)")
                if 'CU_AU_PERCENTILE_INDEX' in df_filtered.columns:
                    fig_index = create_distribution_plots(df_filtered, 'CU_AU_PERCENTILE_INDEX')
                    if fig_index:
                        st.plotly_chart(fig_index, use_container_width=True)
                        st.caption("📊 Índice de 0-2: valores >1.5 indican alta calidad polimetálica")            
                        st.markdown("---")
            
            # Temporal
            st.subheader("Evolución Temporal")
            fig_temporal = create_temporal_analysis(df_filtered)
            if fig_temporal:
                st.plotly_chart(fig_temporal, use_container_width=True)
        
        # TAB 4: Geoespacial
        with tab4:
            st.header("🗺️ Análisis Geoespacial")
            
            # Calcular estadísticas de cobertura geográfica
            total_samples = len(df_filtered)
            samples_with_coords = df_filtered['HAS_VALID_COORDS'].sum()
            samples_without_coords = total_samples - samples_with_coords
            
            # Mostrar métricas de cobertura
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("📊 Total Muestras", f"{total_samples:,}")
            with col_info2:
                st.metric("📍 Con Coordenadas", f"{samples_with_coords:,}")
            with col_info3:
                st.metric("❓ Sin Coordenadas", f"{samples_without_coords:,}")
            
            st.info("💡 Los mapas son interactivos: zoom, pan, hover para detalles. Solo se muestran muestras con coordenadas geográficas válidas.")
            
            # Selector de variable para colorear
            st.subheader("Distribución Geográfica de Muestras")
            
            color_variable = st.selectbox(
                "🎨 Mostrar puntos por:",
                options=['CU_PPM', 'CU_AU_PERCENTILE_INDEX'],
                format_func=lambda x: 'Contenido de Cu (ppm)' if x == 'CU_PPM' else 'Índice Polimetálico (0-2)',
                help="Selecciona la variable para colorear los puntos en el mapa"
            )
            
            # Mapa principal
            fig_map = create_geospatial_map(df_filtered, color_by=color_variable)
            if fig_map:
                st.plotly_chart(fig_map, use_container_width=True)
            
            st.markdown("---")
            
            # Top estados
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top 10 Estados - Total Muestras")
                top_states = df_filtered['STATE'].value_counts().head(10)
                fig_top = px.bar(
                    x=top_states.values,
                    y=top_states.index,
                    orientation='h',
                    labels={'x': 'Cantidad', 'y': 'Estado'},
                    title='Estados con Más Muestras'
                )
                st.plotly_chart(fig_top, use_container_width=True)
            
            with col2:
                st.subheader("Top 10 Estados - Muestras Polimetálicas")
                poly_by_state = df_filtered[df_filtered['IS_POLYMETALLIC']].groupby('STATE').size().sort_values(ascending=False).head(10)
                fig_poly = px.bar(
                    x=poly_by_state.values,
                    y=poly_by_state.index,
                    orientation='h',
                    labels={'x': 'Cantidad', 'y': 'Estado'},
                    title='Muestras Polimetálicas por Estado',
                    color=poly_by_state.values,
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_poly, use_container_width=True)
        
        # TAB 5: IA
        with tab5:
            st.header("🤖 Insights Generados por Inteligencia Artificial")
            
            st.markdown("""
            ### ¿Cómo funciona?
            Utiliza la **API de Groq** con modelos avanzados (Llama-3/Mixtral) para generar 
            insights profesionales basados en las estadísticas de los datos filtrados.
            """)
            
            st.markdown("---")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                model_choice = st.selectbox(
                    "Modelo de IA:",
                    ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
                )
            
            with col2:
                if st.button("🤖 Generar Insights", type="primary", use_container_width=True):
                    if not groq_api_key:
                        st.warning("⚠️ Ingresa tu Groq API Key en la barra lateral")
                    elif df_filtered.empty:
                        st.warning("⚠️ No hay datos con los filtros actuales")
                    else:
                        insights = generate_ai_insights(df_filtered, groq_api_key, model=model_choice)
                        st.session_state['ai_insights'] = insights
                        st.success("✅ Insights generados")
            
            # Mostrar insights
            if 'ai_insights' in st.session_state:
                st.markdown("---")
                st.markdown("### 💡 Análisis Generado:")
                st.markdown(st.session_state['ai_insights'])
            else:
                st.info("👆 Haz clic en 'Generar Insights' para análisis con IA")
            
            # Info API
            with st.expander("ℹ️ ¿Cómo obtener API Key de Groq?"):
                st.markdown("""
                1. Ve a https://console.groq.com/
                2. Crea cuenta gratuita
                3. Genera nueva clave API
                4. Cópiala en la barra lateral
                """)
        
        # TAB 6: PDF
        with tab6:
            st.header("📄 Generación de Reporte PDF")
            
            st.markdown("""
            ### Contenido del Reporte
            - Resumen ejecutivo con métricas clave
            - Insights generados por IA (si disponibles)
            - Estadísticas descriptivas detalladas
            - Fecha de generación
            """)
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if st.button("📥 Generar PDF", type="primary", use_container_width=True):
                    insights_text = st.session_state.get('ai_insights', '')
                    pdf_buffer = generate_pdf_report(df_filtered, insights_text)
                    
                    if pdf_buffer:
                        st.download_button(
                            label="💾 Descargar PDF",
                            data=pdf_buffer,
                            file_name=f'reporte_cu_au_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf',
                            mime='application/pdf',
                            use_container_width=True
                        )
                        st.success("✅ Reporte PDF generado")
                    else:
                        st.error("❌ Error al generar PDF")
            
            # Vista previa
            with st.expander("👁️ Vista Previa de Estadísticas (Concentraciones en PPM)"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Muestras", f"{len(df_filtered):,}")
                    st.metric("Estados", df_filtered['STATE'].nunique())
                    st.metric("Polimetálicos", df_filtered['IS_POLYMETALLIC'].sum())
                with col2:
                    st.metric("Cu Promedio", f"{df_filtered['CU_PPM'].mean():.2f} ppm")
                    st.metric("Au Promedio", f"{df_filtered['AU_PPM'].mean():.4f} ppm")
                    if 'CU_AU_PERCENTILE_INDEX' in df_filtered.columns:
                        st.metric("Índice Polimetálico", f"{df_filtered['CU_AU_PERCENTILE_INDEX'].mean():.3f}")
    
    else:
        # Sin datos cargados
        st.info("👈 Carga un archivo de datos usando la barra lateral")
        
        with st.expander("ℹ️ Información sobre el dataset"):
            st.markdown(f"""
            ### Columnas Requeridas
            {chr(10).join([f"- {col}" for col in REQUIRED_COLUMNS])}
            
            ### Formatos Soportados
            - CSV, JSON, URL
            
            ### Fuente
            Geoscience Australia: https://portal.ga.gov.au/
            """)


# ==============================================================================
# EJECUTAR APLICACIÓN
# ==============================================================================

if __name__ == "__main__":
    main()