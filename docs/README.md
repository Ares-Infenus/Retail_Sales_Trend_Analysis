# Estructura de carpetas del proyecto
```
project-retail-sales-analysis/
├── .github/
│   └── workflows/
│       └── ci.yml                    # Integración continua / tests automatizados
├── infra/                            # Infraestructura y despliegue
│   ├── docker-compose.yml            # Orquestación de contenedores
│   └── migrations/                   # Scripts SQL de migraciones
│       └── 001_initial_schema.sql
├── config/                           # Archivos de configuración
│   ├── config.yaml                   # Parámetros de conexión, paths, etc.
│   └── .env.example                  # Variables de entorno de muestra
├── data/                             # Almacenamiento de datos
│   ├── raw/                          # Datos originales (no tocar)
│   └── processed/                    # Datos limpios / transformados
├── docs/                             # Documentación del proyecto
│   ├── architecture.md               # Descripción de la arquitectura (PlantUML)
│   └── requirements.txt              # Dependencias para la documentación
├── notebooks/                        # Jupyter Notebooks por fase
│   ├── 01_EDA.ipynb
│   ├── 02_Cleaning.ipynb
│   ├── 03_Trend_Analysis.ipynb
│   ├── 04_Modeling.ipynb
│   └── 05_Dashboard_Prototype.ipynb
├── reports/                          # Entregables finales
│   └── executive_report.pdf
├── dashboards/                       # Código del dashboard interactivo
│   └── interactive_dashboard/
│       ├── app.py
│       └── requirements.txt
├── scripts/                          # Scripts de utilidad
│   ├── start.sh                      # Levanta todos los servicios (DB, dashboard…)
│   └── load_data.sh                  # Carga inicial de CSV → PostgreSQL
├── src/                              # Código fuente principal
│   ├── infra/                        # Clases de infraestructura Docker/DB
│   │   ├── docker_compose.py         # Clase DockerCompose
│   │   └── postgres_container.py     # Clase PostgresContainer
│   ├── ingestion/                    # Módulo de ingestión de datos
│   │   └── ingest.py
│   ├── cleaning/                     # Módulo de limpieza y validación
│   │   └── clean.py
│   ├── analysis/                     # Módulo de análisis exploratorio
│   │   └── eda.py
│   ├── modeling/                     # Módulo de modelado de series temporales
│   │   └── forecast.py
│   ├── visualization/                # Funciones para gráficas reutilizables
│   │   └── plots.py
│   └── utils/                        # Helpers generales (logging, config, io)
│       └── helpers.py
├── tests/                            # Pruebas unitarias e integración
│   ├── test_ingest.py
│   ├── test_clean.py
│   ├── test_eda.py
│   └── test_db.py                    # Pruebas de conexión y migraciones
├── .gitignore
├── README.md                         # Visión general + cómo arrancar
├── environment.yml                   # Entorno Conda (o requirements.txt)
└── LICENSE

```

## Descripción de carpetas y archivos (versión actualizada)

- **`.github/workflows/ci.yml**  
  Configuración de integración continua: ejecución de tests automáticos, linting y chequeos de estilo.

- **`infra/`**  
  Directorio de infraestructura y despliegue:  
  - `docker-compose.yml`: orquestación del contenedor PostgreSQL y otros servicios.  
  - `migrations/`: scripts SQL de migraciones de esquema (p. ej. `001_initial_schema.sql`).

- **`config/`**  
  Centraliza la configuración del proyecto:  
  - `config.yaml`: parámetros de conexión, rutas y ajustes generales.  
  - `.env.example`: plantilla de variables de entorno (credenciales, puertos, etc.).

- **`data/`**  
  - `raw/`: datos originales (no modificar).  
  - `processed/`: datos limpios y transformados listos para análisis.

- **`docs/`**  
  Documentación general del proyecto:  
  - `diagrams/`: código fuente PlantUML de todos los diagramas.  
  - `images/`: versiones exportadas (PNG/SVG) de los diagramas.  
  - `architecture.md`: explicación detallada de la arquitectura y referencias a los diagramas.

- **`notebooks/`**  
  Jupyter Notebooks organizados por fase:  
  1. `01_EDA.ipynb`  
  2. `02_Cleaning.ipynb`  
  3. `03_Trend_Analysis.ipynb`  
  4. `04_Modeling.ipynb`  
  5. `05_Dashboard_Prototype.ipynb`

- **`reports/`**  
  Entregables finales:  
  - `executive_report.pdf`: reporte ejecutivo con hallazgos y recomendaciones.

- **`dashboards/interactive_dashboard/`**  
  Código y dependencias del dashboard interactivo (Streamlit, Dash, etc.):  
  - `app.py`  
  - `requirements.txt`

- **`scripts/`**  
  Scripts de utilidad para orquestar tareas comunes:  
  - `start.sh`: levanta servicios (DB, dashboard, etc.).  
  - `load_data.sh`: carga inicial de CSV → PostgreSQL.

- **`src/`**  
  Código fuente modularizado por dominio:  
  - **`infra/`**:  
    - `docker_compose.py`: clase de orquestación DockerCompose.  
    - `postgres_container.py`: clase PostgresContainer para gestionar el contenedor DB.  
  - **`ingestion/`**: `ingest.py` (lectura de CSV/DB y carga en DB).  
  - **`cleaning/`**: `clean.py` (limpieza y validación de datos).  
  - **`analysis/`**: `eda.py` (estadísticas descriptivas, correlaciones).  
  - **`modeling/`**: `forecast.py` (modelado de series temporales, ARIMA/Prophet).  
  - **`visualization/`**: `plots.py` (funciones reutilizables para gráficos).  
  - **`utils/`**: `helpers.py` (logging, configuración, I/O general).

- **`tests/`**  
  Pruebas unitarias y de integración:  
  - `test_ingest.py`  
  - `test_clean.py`  
  - `test_eda.py`  
  - `test_db.py` (conectividad, migraciones y esquemas en PostgreSQL)

- **`.gitignore`**  
  Archivos y carpetas a excluir del control de versiones.

- **`README.md`**  
  Visión general del proyecto, guía de instalación, comandos principales y ejemplos de uso.

- **`environment.yml`** (o `requirements.txt`)  
  Definición del entorno Conda (o pip) con todas las dependencias necesarias.

- **`LICENSE`**  
  Licencia de uso y distribución del proyecto.
