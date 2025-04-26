# Estructura de carpetas del proyecto
```
project-retail-sales-analysis/
├── .github/
│   └── workflows/
│       └── ci.yml                  # (Opcional) Integración continua / tests automatizados
├── data/
│   ├── raw/                        # Datos originales (no tocar)
│   └── processed/                  # Datos limpios / transformados
├── docs/
│   ├── architecture.md             # Descripción de la arquitectura (incluye PlantUML)
│   └── requirements.txt            # Dependencias para documentación (mkdocs, etc.)
├── notebooks/
│   ├── 01_EDA.ipynb                # Exploración inicial
│   ├── 02_Cleaning.ipynb           # Limpieza y preprocesamiento
│   ├── 03_Trend_Analysis.ipynb     # Análisis de tendencias y estacionalidad
│   ├── 04_Modeling.ipynb           # Modelado de series temporales (opcional)
│   └── 05_Dashboard_Prototype.ipynb# Prototipo de visualizaciones
├── reports/
│   └── executive_report.pdf        # Versión final del reporte ejecutivo
├── dashboards/
│   └── interactive_dashboard/      # Código / archivos del dashboard (p. ej. Streamlit)
│       ├── app.py
│       └── requirements.txt
├── src/
│   ├── ingestion/                  # Módulo de ingestión de datos
│   │   └── ingest.py
│   ├── cleaning/                   # Módulo de limpieza y validación
│   │   └── clean.py
│   ├── analysis/                   # Módulo de análisis exploratorio y tendencias
│   │   └── eda.py
│   ├── modeling/                   # Módulo de modelado de series temporales
│   │   └── forecast.py
│   ├── visualization/              # Funciones para gráficas reutilizables
│   │   └── plots.py
│   └── utils/                      # Helpers generales (logging, config, io)
│       └── helpers.py
├── tests/                          # Pruebas unitarias / de integración
│   ├── test_ingest.py
│   ├── test_clean.py
│   └── test_eda.py
├── .gitignore
├── README.md                       # Visión general + cómo arrancar
├── environment.yml                 # Entorno Conda (o `requirements.txt`)
└── LICENSE
```

## Descripción de carpetas y archivos

- **`.github/workflows/ci.yml`**  
  Configuración de integración continua (tests automáticos, linting).

- **`data/`**  
  - `raw/`: Datos originales sin modificar.  
  - `processed/`: Datos limpios y transformados listos para análisis.

- **`docs/`**  
  - `diagrams/`: Código fuente PlantUML de tus diagramas.  
  - `images/`: Versiones PNG/SVG exportadas de cada diagrama.  
  - `architecture.md`: Documento que agrupa explicaciones y muestra las imágenes.

- **`notebooks/`**  
  Notebooks para exploración, limpieza, análisis de tendencias, modelado y prototipado de dashboard.

- **`reports/`**  
  Salidas finales en PDF del reporte ejecutivo.

- **`dashboards/interactive_dashboard/`**  
  Código y dependencias para el dashboard interactivo (por ejemplo en Streamlit o Dash).

- **`src/`**  
  Código modularizado por dominio de responsabilidad: ingestión, limpieza, análisis, modelado, visualización y utilidades.

- **`tests/`**  
  Pruebas unitarias e integración para cada módulo clave.

- **`README.md`**  
  Guía rápida de instalación, listado de scripts y uso general del proyecto.

- **`environment.yml`**  
  Especifica el entorno Conda con todas las dependencias necesarias.

- **`LICENSE`**  
  Licencia abierta para el uso y distribución del proyecto.
