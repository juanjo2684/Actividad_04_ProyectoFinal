# 📊 Dashboard de Análisis: Potencial Cu-Au en Estados Unidos

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://actividad04proyectofinal-gdxlqr2gpgcgvhejgtgjlk.streamlit.app/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Claude](https://img.shields.io/badge/Built%20with-Claude%20Sonnet%204.5-blueviolet)](https://www.anthropic.com/claude)

## 👥 Integrantes del Equipo
- **Juan Morales**
- **Sebastian Ruiz**
- **Daniel Pareja**

**Institución:** Universidad EAFIT, Medellín, Colombia  
**Curso:** Fundamentos de Ciencia de Datos  
**Fecha:** Febrero 2026

---

## 📋 CONTEXTO DE NEGOCIO

### 1. Situación Actual
En el marco de la transición energética global hacia fuentes de energía renovable y electrificación del transporte, el cobre se ha consolidado como un mineral crítico estratégico debido a su rol esencial en infraestructura eléctrica, vehículos eléctricos, sistemas de almacenamiento de energía y generación renovable. El gobierno de Estados Unidos ha identificado la necesidad de asegurar el suministro doméstico de cobre para reducir la dependencia de importaciones.

Sin embargo, el cobre enfrenta una alta volatilidad de precios en los mercados internacionales, lo que representa un riesgo económico significativo para proyectos de exploración y desarrollo minero. Los depósitos polimetálicos que contienen cobre + oro ofrecen una ventaja estratégica al diversificar el riesgo económico.

### 2. Problema de Negocio
El gobierno estadounidense requiere identificar y priorizar los estados con mayor potencial para albergar depósitos de cobre, especialmente aquellos que presenten asociaciones polimetálicas de cobre-oro o oro-cobre, que permitan:

- ✅ Mitigar riesgos asociados a fluctuaciones en el precio del cobre mediante ingresos complementarios de oro
- ✅ Optimizar la asignación de recursos para exploración y desarrollo minero
- ✅ Fortalecer la seguridad del suministro de minerales críticos a nivel nacional

### 3. Objetivo del Proyecto
Desarrollar una herramienta analítica visual basada en el análisis de datos geoquímicos de la base de datos **Critical Mineral Deposits Geochemistry**, que permita al gobierno estadounidense responder a las siguientes preguntas estratégicas:

1. **¿Qué estados presentan la mayor concentración de depósitos con potencial de cobre?**
2. **¿Cuáles son los estados prioritarios para exploración de sistemas polimetálicos Cu-Au?**
3. **¿Qué características distinguen a los depósitos polimetálicos de alto valor?**

---

## 🚀 INSTALACIÓN Y EJECUCIÓN LOCAL

### Requisitos Previos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Git

### Pasos de Instalación

#### 1. Clonar el repositorio
```bash
git clone https://github.com/juanjo2684/Actividad_04_ProyectoFinal.git
cd Actividad_04_ProyectoFinal
```

#### 2. Crear entorno virtual (recomendado)
```bash
# En Windows
python -m venv venv
venv\Scripts\activate

# En macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

**Contenido de `requirements.txt`:**
```txt
streamlit
pandas
numpy
plotly
requests
groq
reportlab
openpyxl
python-dateutil
```

#### 4. Configurar API Key de Groq (opcional para IA)
Para usar la funcionalidad de Insights con IA:
1. Crear cuenta gratuita en [Groq Console](https://console.groq.com/)
2. Generar API Key
3. Ingresar la clave en la barra lateral del dashboard al ejecutarlo

#### 5. Ejecutar la aplicación
```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

### Estructura del Proyecto
```
Actividad_04_ProyectoFinal/
│
├── app.py                          # Aplicación principal Streamlit
├── requirements.txt                # Dependencias del proyecto
├── README.md                       # Este archivo
└── .gitignore                      # Archivos ignorados por Git
```

---

## 🌐 DESPLIEGUE EN LA NUBE

### 🔗 Aplicación Desplegada
**URL del Dashboard:** [https://actividad04proyectofinal-gdxlqr2gpgcgvhejgtgjlk.streamlit.app/](https://actividad04proyectofinal-gdxlqr2gpgcgvhejgtgjlk.streamlit.app/)

### Cómo Usar la Aplicación

1. **Acceder al dashboard** a través del enlace proporcionado
2. **Cargar datos** mediante una de las tres opciones:
   - 📁 Subir archivo CSV local
   - 📄 Subir archivo JSON local
   - 🌐 Cargar desde URL directa del dataset
3. **Configurar procesamiento** de datos en la barra lateral:
   - Eliminación de duplicados
   - Método de imputación (media, mediana, cero)
   - Tratamiento de outliers (mantener, eliminar, winsorizar)
4. **Aplicar filtros globales** para refinar el análisis:
   - Rango de fechas
   - Países y estados
   - Grupos de depósito
   - Rangos de concentración de Cu y Au
   - Clasificación de commodities
5. **Explorar las pestañas** de análisis:
   - ⚙️ **Procesamiento:** Calidad de datos y estadísticas
   - 📊 **Univariado:** Distribuciones y clasificaciones por estado
   - 🔗 **Bivariado:** Correlaciones y tendencias temporales
   - 🗺️ **Geoespacial:** Mapas interactivos de muestras georreferenciadas
   - 🤖 **Insights IA:** Análisis automatizado con modelos de lenguaje (Groq API)
   - 📄 **Reporte PDF:** Generación de informes ejecutivos

---

## 🛠️ FUNCIONALIDADES IMPLEMENTADAS

### ✅ Análisis de Datos Avanzado
- **Carga dinámica** de datos (CSV, JSON, URL) con manejo robusto de errores
- **Procesamiento completo:** eliminación de duplicados, imputación de valores faltantes, tratamiento de outliers
- **Feature Engineering avanzado:**
  - Conversión de unidades (AU_PPB → AU_PPM)
  - **Índice Polimetálico (0-2):** Suma de percentiles de Cu y Au para identificar depósitos de alto valor
  - **Z-Scores locales por estado:** Normalización robusta usando mediana y MAD que mitiga el sesgo de cantidad de muestras
  - Clasificación automática de muestras (Polimetálico Cu-Au, Cu dominante, Au dominante, Baja ley)
  - Cálculo de densidad de muestreo por estado (confianza estadística)

### 📊 Visualizaciones Interactivas
- **Gráficos Plotly** completamente interactivos
- **Mapas geoespaciales** con selector de variables (Cu vs Índice Polimetálico)
- **Análisis temporal** de evolución de concentraciones
- **Matriz de correlación** entre variables geoquímicas
- **Distribuciones univariadas** con histogramas y boxplots
- **Análisis por estado** con barras apiladas por categoría

### 🤖 Inteligencia Artificial
- **Integración con Groq API** (modelos Llama-3.3-70b y Mixtral-8x7b)
- **Insights automáticos** orientados a las 3 preguntas de negocio
- **Detección de sesgos estadísticos** advirtiendo sobre estados con baja densidad de muestreo (<30 muestras)
- Análisis crítico con recomendaciones estratégicas

### 📄 Reportes Profesionales en PDF
- **Generación automática de PDF** con ReportLab
- **Estructura orientada a decisiones:** responde directamente las 3 preguntas de negocio
- **Contenido incluido:**
  - Portada con resumen ejecutivo
  - Top 15 estados por concentración promedio de Cu
  - Top 10 estados por muestras polimetálicas
  - Comparación estadística polimetálicos vs no polimetálicos
  - Insights de IA formateados
  - Estadísticas descriptivas completas
- **Exportación de datos** filtrados en formato CSV

### 🎯 Filtros Globales Interactivos
- Rango de fechas de análisis
- Selección de países y estados
- Grupos de depósito
- Rangos de concentración de Cu y Au (sliders numéricos)
- Clasificación de commodities

---

## 📊 FUENTE DE DATOS

### Dataset Principal
**Critical Mineral Deposits Geochemistry**  
🌐 **Fuente:** [Geoscience Australia Portal](https://portal.ga.gov.au/)

**Link de Descarga Directo:**
```
https://critical-minerals.prod-geoserver.gis.ga.gov.au/geoserver/wfs?request=GetFeature&service=WFS&version=1.1.0&typeName=cmmi:CriticalMineralDepositsGeochemistry&outputFormat=excel2007&srsName=EPSG:4326
```

### Descripción del Dataset
- **Registros:** ~50,000+ muestras geoquímicas
- **Cobertura geográfica:** Global (énfasis en EE.UU., Australia, Canadá)
- **Variables principales:** 
  - Concentraciones de Cu (ppm) y Au (ppb convertido a ppm)
  - Coordenadas geográficas (WGS84)
  - Tipos de depósito y métodos de muestreo
  - Fechas de análisis geoquímico

### Columnas Requeridas del Dataset
```
DEPOSIT_GROUP, DEPOSIT_TYPE, PRIMARY_COMMODITIES, SAMPLE_UID, 
SAMPLING_METHOD, PROVINCE, AU_PPB, CU_PPM, ANALYSIS_DATETIME, 
COUNTRY, STATE, SAMPLE_LONGITUDE_WGS84, SAMPLE_LATITUDE_WGS84, 
SAMPLE_LOCATION_DESCRIPTION
```

---

## 🧮 METODOLOGÍA TÉCNICA

### Feature Engineering Avanzado

#### 1. Conversión de Unidades
```python
AU_PPM = AU_PPB / 1000  # Comparabilidad directa con Cu
```

#### 2. Índice Polimetálico (0-2)
```python
# Percentiles de Cu y Au (0-1 cada uno)
CU_PERCENTILE = rank(CU_PPM, pct=True)
AU_PERCENTILE = rank(AU_PPM, pct=True)

# Índice combinado (suma de percentiles)
CU_AU_PERCENTILE_INDEX = CU_PERCENTILE + AU_PERCENTILE
```

**Interpretación:**
- **0.0 - 0.5:** Muy baja ley (percentiles bajos en ambos)
- **0.5 - 1.0:** Baja ley
- **1.0 - 1.5:** Potencial moderado
- **1.5 - 1.8:** Alta calidad polimetálica ⭐
- **1.8 - 2.0:** Excepcional (joyas) 💎

#### 3. Z-Scores Locales por Estado
```python
# Normalización robusta usando mediana y MAD
def get_zscore_local(group):
    med = group.median()
    mad = (group - med).abs().median()
    return (group - med) / (mad if mad != 0 else 1.0)

AU_ZSCORE_LOCAL = groupby('STATE')['AU_PPM'].transform(get_zscore_local)
CU_ZSCORE_LOCAL = groupby('STATE')['CU_PPM'].transform(get_zscore_local)
```

**Ventajas:**
- Identifica anomalías relativas al contexto geológico local
- Mitiga sesgo de estados con pocas muestras
- Más robusto que Z-scores globales ante outliers

#### 4. Clasificación de Muestras
```python
# Umbrales económicos en PPM
Cu_threshold = 1000 ppm  # 0.1% Cu
Au_threshold = 0.1 ppm   # 0.1 ppm Au

if Cu >= 1000 and Au >= 0.1:
    → 'Polimetálico Cu-Au'
elif Cu >= 1000:
    → 'Cu dominante'
elif Au >= 0.1:
    → 'Au dominante'
else:
    → 'Baja ley'
```

### Manejo de Sesgos Estadísticos

El dashboard implementa **advertencias automáticas** cuando un estado presenta:
- **Menos de 30 muestras** (baja confianza estadística)
- **Concentraciones extremas con muestreo limitado**

La IA recibe información detallada sobre la densidad de muestreo por estado y advierte explícitamente sobre hallazgos que requieren confirmación.

---

## 💡 INSIGHTS CLAVE DEL PROYECTO

### Descubrimientos Principales
1. **Sesgo de cantidad de muestras:** Estados con pocas muestras pero concentraciones altas pueden ser falsos positivos
2. **Índice polimetálico superior al ratio simple:** Normaliza diferencias de magnitud entre Cu (miles de ppm) y Au (décimas de ppm)
3. **Z-scores locales revelan anomalías contextuales:** Un depósito puede ser excepcional en su región pero promedio globalmente

### Criterios de Priorización Desarrollados
- **Alta confianza:** Estados con >30 muestras + Índice Polimetálico >1.5
- **Potencial por confirmar:** Estados con pocas muestras pero concentraciones prometedoras
- **Diversificación de riesgo:** Depósitos polimetálicos reducen exposición a volatilidad del precio del cobre

---

## 🎓 TECNOLOGÍAS UTILIZADAS

### Stack Tecnológico
- **Python 3.8+**
- **Streamlit** - Framework web interactivo
- **Pandas & NumPy** - Procesamiento y análisis de datos
- **Plotly** - Visualizaciones interactivas
- **ReportLab** - Generación de PDFs profesionales
- **Groq API** - Integración de IA (Llama-3.3, Mixtral-8x7b)
- **Requests** - Carga de datos desde URLs
- **Claude Sonnet 4.5** - Asistencia en desarrollo y optimización de código

### Bibliotecas Principales
```python
streamlit        # Dashboard interactivo
pandas           # Manipulación de datos
numpy            # Cálculos numéricos
plotly           # Gráficos interactivos
groq             # API de IA
reportlab        # Generación de PDFs
openpyxl         # Manejo de archivos Excel
python-dateutil  # Procesamiento de fechas
```

---

## 🤖 PROCESO DE DESARROLLO

### Asistencia con IA
Este proyecto fue desarrollado con la **asistencia de Claude Sonnet 4.5** (Anthropic), utilizado como copiloto de programación para:

- ✅ **Arquitectura del código:** Diseño de estructura modular y funciones reutilizables
- ✅ **Optimización de algoritmos:** Implementación eficiente de Z-scores locales y cálculo de percentiles
- ✅ **Debugging y troubleshooting:** Resolución de errores en procesamiento de datos y visualizaciones
- ✅ **Mejores prácticas:** Aplicación de patrones de diseño y convenciones de código Python
- ✅ **Documentación:** Generación de docstrings y comentarios explicativos
- ✅ **Feature engineering:** Desarrollo de métricas avanzadas (Índice Polimetálico, Z-scores locales)
- ✅ **Integración de APIs:** Implementación de llamadas a Groq API para insights de IA
- ✅ **Generación de reportes:** Creación de PDFs estructurados con ReportLab

### Metodología de Desarrollo
1. **Análisis de requisitos** con enfoque en las 3 preguntas de negocio
2. **Diseño iterativo** del pipeline de datos y visualizaciones
3. **Desarrollo incremental** con pruebas continuas
4. **Validación geológica** de métricas y clasificaciones
5. **Optimización de rendimiento** para datasets grandes
6. **Despliegue en Streamlit Cloud** con documentación completa

---

## 📈 MÉTRICAS DE RENDIMIENTO

### Escalabilidad
- ✅ Probado con datasets de hasta **50,000+ registros**
- ✅ Procesamiento en tiempo real con filtros interactivos
- ✅ Generación de PDFs en <5 segundos
- ✅ Mapas interactivos con miles de puntos georreferenciados

### Optimizaciones Implementadas
- Carga selectiva de columnas (`usecols`)
- Tipos de datos optimizados
- Caching de Streamlit para funciones pesadas
- Procesamiento vectorizado con NumPy/Pandas

---

## 🔒 LIMITACIONES Y TRABAJO FUTURO

### Limitaciones Actuales
- Generación de PDFs sin gráficos embebidos (solo tablas y texto)
- Análisis limitado a datos geoquímicos (no incluye costos de extracción)
- Requiere conexión a internet para API de IA

### Mejoras Futuras
- [ ] Integración con modelos de costos de extracción
- [ ] Análisis de viabilidad económica por depósito
- [ ] Dashboard multi-idioma (inglés/español)
- [ ] Exportación a formatos adicionales (Excel, PowerPoint)
- [ ] Clustering geoespacial de depósitos
- [ ] Análisis predictivo con Machine Learning

---

## 👥 CRÉDITOS

### Autores
- **Juan Morales** - Estudiante de Maestría en Ciencia de Datos
- **Sebastian Ruiz** - Estudiante de Maestría en Ciencia de Datos
- **Daniel Pareja** - Estudiante de Maestría en Ciencia de Datos

**Institución:** Universidad EAFIT, Medellín, Colombia  
**Programa:** Maestría en Ciencia de Datos  
**Curso:** Fundamentos de Ciencia de Datos  
**Fecha:** Febrero 2026

### Fuentes de Datos
- **Geoscience Australia** - Critical Mineral Deposits Geochemistry Database
- **Groq API** - Modelos de lenguaje para generación de insights

### Herramientas de Desarrollo
- **Claude Sonnet 4.5 (Anthropic)** - Asistente de IA para desarrollo de código, optimización de algoritmos y documentación
- **GitHub Copilot** - Asistencia adicional en autocompletado de código
- **Streamlit Cloud** - Plataforma de despliegue

---

## 📝 LICENCIA

Este proyecto fue desarrollado con fines académicos para el curso de Fundamentos de Ciencia de Datos de la Universidad EAFIT.

