# Análisis de Tendencias de Ventas Minoristas

Repositorio para el proyecto **"Análisis de Tendencias de Ventas Minoristas"**, centrado en las primeras fases de ingesta y exploración de datos del dataset real de ventas (Corporación Favorita Grocery Sales Forecasting de Kaggle).

---

## 📋 Tabla de Contenidos

1. [Descripción del Proyecto](#descripción-del-proyecto)
2. [Progreso Actual](#progreso-actual)
3. [Escenario Empresarial](#escenario-empresarial)
4. [Estructura del Repositorio](#estructura-del-repositorio)
5. [Requisitos](#requisitos)
6. [Instalación y Configuración](#instalación-y-configuración)
7. [Flujo de Trabajo y Fases](#flujo-de-trabajo-y-fases)
8. [Integración Continua (CI/CD)](#integración-continua-cicd)
9. [Contribuir](#contribuir)
10. [Licencia](#licencia)

---

## Descripción del Proyecto

Este proyecto tiene como propósito realizar un análisis meticuloso de los registros de ventas minoristas de Corporación Favorita (Kaggle). A través de un proceso sistemático de ingesta y exploración de datos, se persigue:

- Identificar tendencias de largo plazo y patrones estacionales que influyen en la demanda.
- Detectar anomalías y comportamientos atípicos que puedan revelar oportunidades o riesgos.
- Generar visualizaciones de alto impacto para interpretar indicadores clave de rendimiento (KPIs).
- Sentar las bases metodológicas para la formulación de recomendaciones estratégicas en gestión de inventarios, programación de campañas promocionales y políticas de precios.

> **Visión a futuro**: Desarrollar un reporte ejecutivo y un dashboard interactivo que respalden la toma de decisiones de la alta dirección de una cadena retail.

Dataset base: [Corporación Favorita Grocery Sales Forecasting (Kaggle)](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data)

# Flujo de Trabajo Refinado para Proyecto de Análisis de Ventas Minoristas

Basado en tus diagramas y arquitectura, te propongo un flujo de trabajo más coherente y detallado que se alinea mejor con tu proyecto de análisis de ventas minoristas.

## Definition of Done (DoD) Refinado
Cada entregable se considerará completo cuando:
1. Funcionalidad verificada sin errores
2. Pasa linting (Black + Flake8)
3. Cuenta con tests críticos (pytest)
4. Incluye documentación mínima (docstrings o Markdown)
5. Es reproducible en Docker limpio
6. Se integra correctamente con la base de datos PostgreSQL

## Día 1: Setup & Infraestructura Inicial

**Entregables:**
* **Tablero Kanban** en GitHub Projects con columnas To Do, In Progress, Done y etiquetas de prioridad
* **CI Pipeline** configurado (`.github/workflows/ci.yml`) integrando Black, Flake8 y pytest
* **Docker Compose** configurado (`infra/docker-compose.yml`) con:
 - Servicio PostgreSQL
 - Volúmenes persistentes
 - Variables de entorno en `.env` 
* **Scripts de Migración** (`infra/migrations/001_initial_schema.sql`) definiendo tablas para ventas, productos, tiendas y promociones
* **Config centralizado** (`config/config.yaml`) con parámetros de conexión y rutas
* **Test de Infraestructura** (`tests/test_db.py`) validando conexión a PostgreSQL

## Día 2: Ingesta de Datos & EDA Inicial

**Entregables:**
* **Módulo de Ingesta** (`src/ingestion/ingest.py`) con:
 - Lectura chunked para datasets grandes
 - Validación de esquema inicial
 - Carga a PostgreSQL con logging
* **Notebook EDA Inicial** (`notebooks/01_EDA.ipynb`) con:
 - Conexión a PostgreSQL
 - Estadísticas descriptivas por categoría y tienda
 - Visualizaciones de distribuciones y correlaciones
 - Detección de valores faltantes y outliers
* **Test de Ingesta** (`tests/test_ingest.py`) validando:
 - Correcta lectura de chunks
 - Inserción en PostgreSQL
 - Manejo de tipos de datos

## Día 3: Limpieza & Análisis de Tendencias

**Entregables:**
* **Módulo de Limpieza** (`src/cleaning/clean.py`) con:
 - Funciones vectorizadas para imputación
 - Detección y tratamiento de outliers
 - Normalización de categorías y fechas
 - Logging detallado de transformaciones
* **Notebook de Limpieza** (`notebooks/02_Cleaning.ipynb`) demostrando el flujo completo
* **Notebook de Análisis** (`notebooks/03_Trend_Analysis.ipynb`) con:
 - Series temporales de ventas por segmento
 - Análisis de estacionalidad
 - Detección de anomalías y eventos especiales
* **Test de Limpieza** (`tests/test_clean.py`) usando pandera para validar esquema

## Día 4: Modelado & Dashboard Prototipo

**Entregables:**
* **Módulo de Modelado** (`src/modeling/forecast.py`) con:
 - Implementación de modelos de series temporales (ARIMA/Prophet)
 - Evaluación de métricas de error (MAPE, RMSE)
 - Función de predicción para próximos periodos
* **Notebook de Modelado** (`notebooks/04_Modeling.ipynb`) demostrando entrenamiento y validación
* **Módulo de Visualización** (`src/visualization/plots.py`) con funciones reutilizables para:
 - Gráficos de tendencias
 - Comparativas de categorías
 - Mapas de calor por región/tienda
* **Dashboard Prototipo** (`notebooks/05_Dashboard_Prototype.ipynb`) validando componentes visuales
* **Tests de Modelado** (`tests/test_modeling.py`) verificando funciones predictivas

## Día 5: Dashboard Interactivo & Documentación

**Entregables:**
* **App Interactiva** (`dashboards/interactive_dashboard/app.py`) implementada en Streamlit con:
 - Filtros de rango de fechas y categorías
 - KPIs principales (ventas totales, variación %, top productos)
 - Visualización geográfica de tiendas
 - Predicciones de ventas futuras
* **Script de Despliegue** (`scripts/start.sh`) automatizando el arranque del entorno completo
* **Reporte Ejecutivo** (`reports/executive_report.pdf` o `docs/executive_report.md`) con:
 - Hallazgos principales del análisis
 - Tendencias identificadas y anomalías
 - Recomendaciones estratégicas basadas en datos
 - Capturas del dashboard
* **README Actualizado** con:
 - Badges de CI y cobertura de tests
 - Instrucciones de instalación con Docker
 - Enlace al dashboard desplegado
 - Estructura del proyecto y flujo de datos

## Herramientas y Estándares Específicos

* **Base de Datos:** PostgreSQL en Docker con esquemas para ventas, productos, tiendas y promociones
* **Análisis:** pandas + NumPy para manipulación, statsmodels/Prophet para series temporales
* **Visualización:** Matplotlib/Plotly para notebooks, Streamlit para dashboard interactivo
* **Docker:** Multi-container con PostgreSQL + servicios Python
* **Versionado de Datos:** Git-LFS para CSVs grandes (>100MB)
* **Testing:** pytest + fixtures para conexión a DB de test
* **CI/CD:** GitHub Actions para tests, linting y despliegue automático
* **Documentación:** Markdown en docs/ con diagramas PlantUML renderizados

---

## Escenario Empresarial

Simulamos el rol de analista de datos en una cadena de tiendas (p.ej., Walmart). Analizamos historial de ventas (tienda, producto, fecha, promociones, ubicación) para mejorar planificación del próximo año.

---

## Estructura del Repositorio

```
project-retail-sales-analysis/
├── .github/                           # Workflows de CI/CD
├── infra/                             # Infraestructura (Docker, migraciones SQL)
├── config/                            # Configuración (YAML, .env)
├── data/                              # Datos raw y processed
├── docs/                              # Documentación (diagramas y esquema)
├── notebooks/                         # Notebooks: EDA, limpieza, etc.
├── scripts/                           # Utilidades: carga de datos, arranque
├── src/                               # Código fuente: ingestión, limpieza, EDA, utils
├── tests/                             # Pruebas unitarias e integración
├── README.md                          # Este archivo
└── environment.yml                    # Dependencias Conda
```

---

## Requisitos

- Docker & Docker Compose
- Python 3.8+ (Conda recomendado)
- Git

Dependencias definidas en `environment.yml`.

---

## Instalación y Configuración

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/<usuario>/project-retail-sales-analysis.git
   cd project-retail-sales-analysis
   ```
2. Crear y activar entorno Conda:
   ```bash
   conda env create -f environment.yml
   conda activate retail-sales-analysis
   ```
3. Copiar variables de entorno:
   ```bash
   cp config/.env.example .env
   ```
4. Levantar PostgreSQL en Docker:
   ```bash
   docker-compose -f infra/docker-compose.yml up -d
   ```
5. Cargar datos crudos:
   ```bash
   scripts/load_data.sh
   ```

---

## Flujo de Trabajo y Fases

1. **Ingesta de datos** (en desarrollo)
2. **EDA inicial** (pendiente)
3. **Limpieza de datos** (pendiente)
4. **Análisis de tendencias** (pendiente)
5. **Modelado de series temporales** (opcional)
6. **Desarrollo de dashboard** (pendiente)
7. **Reporte ejecutivo** (pendiente)

Cada fase puede ejecutarse desde los notebooks o scripts correspondientes.

---

## Integración Continua (CI/CD)

Configurado para:

- Linting (flake8, black)
- Tests unitarios (pytest)

Workflow en `.github/workflows/ci.yml`.

---

## Contribuir

1. Fork del repositorio
2. Crear rama: `git checkout -b feature/nombre`
3. Commit y push
4. Abrir Pull Request

---

## Licencia

MIT License. Consulta el archivo `LICENSE` para detalles.

