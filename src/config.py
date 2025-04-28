import os
import yaml


class ConfigError(Exception):
    """Error al cargar o validar la configuración."""

    pass


def _validate_section(cfg: dict, section: str, required_keys: list):
    """
    Valida que las secciones y las claves requeridas estén presentes en la configuración.
    """
    if section not in cfg:
        raise ConfigError(f"Sección '{section}' faltante en config.yaml")
    for key in required_keys:
        if key not in cfg[section]:
            raise ConfigError(f"Clave '{key}' faltante en sección '{section}'")


def load_config(path: str = None) -> dict:
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
    except FileNotFoundError:
        raise ConfigError(f"No se encontró el archivo de configuración: {path}")
    except yaml.YAMLError as e:
        raise ConfigError(f"Error de sintaxis en YAML: {e}")

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
    # Aquí puedes loguear o simplemente volver a lanzar
    raise
