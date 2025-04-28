# -*- coding: utf-8 -*-
"""
Test de conexión a PostgreSQL usando pytest y psycopg2.
"""
import os
from typing import Any, Dict

import pytest
import yaml
from dotenv import load_dotenv  # type: ignore[reportUnknownVariableType]
from psycopg2 import OperationalError, connect


# Carga variables de entorno desde .env (si existe)
# Pylance puede reportar el tipo de load_dotenv como parcialmente desconocido,
# por lo que se añade una directiva para ignorar el warning.
_ = load_dotenv()  # type: ignore[reportUnknownVariableType]


def load_config() -> Dict[str, Any]:  # type: ignore[reportUnknownParameterType]
    """
    Carga la configuración de la base de datos desde el archivo YAML.

    Returns:
        Un diccionario con los parámetros de configuración de la base de datos.
    """
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(cfg_path, mode="r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    return config.get("database", {}) or {}


def get_db_params() -> Dict[str, Any]:  # type: ignore[reportUnknownParameterType]
    """
    Construye los parámetros de conexión a la base de datos.

    Prioriza las variables de entorno sobre la configuración en YAML.

    Returns:
        Un diccionario con host, port, user, password y dbname.
    """
    cfg = load_config()

    return {
        "host": os.getenv("DB_HOST", cfg.get("host")),
        "port": int(os.getenv("DB_PORT", cfg.get("port", 5432))),
        "user": os.getenv("DB_USER", cfg.get("user")),
        "password": os.getenv("DB_PASSWORD", cfg.get("password")),
        "dbname": os.getenv("DB_NAME", cfg.get("database") or cfg.get("dbname")),
    }


def test_postgres_connection() -> None:
    """
    Verifica la conexión a PostgreSQL y que la base seleccionada coincide.
    """
    params = get_db_params()
    connection = None

    try:
        connection = connect(**params)
        cursor = connection.cursor()
        cursor.execute("SELECT current_database();")
        current_db = cursor.fetchone()[0]

        assert (
            current_db == params["dbname"]
        ), f"Se conectó a '{current_db}' en lugar de '{params['dbname']}'"

    except OperationalError as error:
        pytest.fail(
            f"No se pudo conectar a PostgreSQL con parámetros {params!r}: {error}"
        )
    finally:
        if connection:
            connection.close()
