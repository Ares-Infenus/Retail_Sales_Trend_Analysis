"""
Módulo para cargar y validar la configuración desde un archivo YAML.

Este módulo proporciona funciones para cargar un archivo de configuración YAML (por defecto,
'config/config.yaml'),
validando la presencia de las secciones y claves necesarias para el correcto funcionamiento del
sistema.

Contiene:
- Una clase de excepción personalizada `ConfigError` para manejar errores relacionados con la
configuración.
- La función `_validate_section` para verificar que las secciones y claves necesarias estén
presentes
en el archivo YAML.
- La función `load_config` para cargar el archivo YAML y validar su contenido. Si faltan secciones o
claves, se lanza un `ConfigError`.

Excepciones:
- `ConfigError`: Se lanza cuando hay errores en el archivo de configuración, como claves o secciones
faltantes.

Requiere:
- `os`: Para manejar rutas de archivos y directorios.
- `yaml`: Para cargar y leer archivos YAML.

Autor: Ians Bastian De PinzonZ
Fecha: 28/04/2025
"""

from typing import Dict, List, Any, Optional

import os
import yaml


class ConfigError(Exception):
    """Error al cargar o validar la configuración."""


def _validate_section(
    cfg: Dict[str, Dict[str, str]], section: str, required_keys: List[str]
):
    """
    Valida que las secciones y las claves requeridas estén presentes en la configuración.
    Lanza un `ConfigError` si falta alguna sección o clave requerida.

    Parámetros:
    - cfg (Dict[str, Dict[str, str]]): Diccionario de configuración donde las claves de nivel
      superior. son nombres de secciones (por ejemplo, "database", "paths"), y cada valor es
      un diccionario con las claves y valores de esa sección.
    - section (str): Nombre de la sección que se va a validar.
    - required_keys (List[str]): Lista de claves que deben estar presentes en la sección.
    """
    if section not in cfg:
        raise ConfigError(f"Sección '{section}' faltante en config.yaml")
    for key in required_keys:
        if key not in cfg[section]:
            raise ConfigError(f"Clave '{key}' faltante en sección '{section}'")


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Carga y valida config/config.yaml.
    Lanza ConfigError si hay errores de formato o claves faltantes.
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    path = os.path.abspath(path)

    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError as exc:
        raise ConfigError(
            f"No se encontró el archivo de configuración: {path}"
        ) from exc
    except yaml.YAMLError as e:
        raise ConfigError(f"Error de sintaxis en YAML: {e}") from e

    # Validar secciones y claves mínimas
    _validate_section(cfg, "database", ["host", "port", "user", "password", "database"])
    _validate_section(
        cfg, "paths", ["log_dir", "data_raw", "data_proc"]
    )  # Nueva validación
    # 'other' puede tener claves adicionales, no exigimos todas
    if "other" not in cfg:
        cfg["other"] = {}

    return cfg


# Carga global al importar el módulo
try:
    config = load_config()
except ConfigError as e:
    # Aquí puedes loguear el error antes de manejarlo
    print(f"Error al cargar la configuración: {e}")
    raise
