"""
creation_syntetic.py

Este módulo implementa un pipeline robusto y paralelizable para la generación de
muestras sintéticas a partir de archivos CSV de gran tamaño. El objetivo principal
es reducir la carga de datos, permitiendo trabajar con subconjuntos representativos
de grandes volúmenes de información manteniendo las estadísticas fundamentales de
las variables originales.

Funcionalidades principales:
----------------------------
- Descubrimiento automático de archivos `.csv` dentro de un directorio especificado.
- Lectura eficiente por fragmentos (`chunks`) para minimizar el uso de memoria.
- Muestreo aleatorio sin reemplazo con control de reproducibilidad.
- Recolección y resumen de estadísticas por tipo de columna (numéricas, categóricas, fechas).
- Detección y manejo de valores especiales como NaN e infinitos.
- Comparación estadística entre el conjunto original y el muestreado utilizando la prueba de Kolmogorov-Smirnov y otras métricas de similitud.
- Registro exhaustivo de eventos, advertencias y errores mediante `logging`.

Estructura del módulo:
----------------------
- **Funciones utilitarias**: gestión de memoria, detección de archivos, apertura segura de archivos, cálculo de núcleos disponibles.
- **ChunkSampler**: clase responsable del muestreo incremental por fragmento.
- **StatsCollector**: clase que acumula estadísticas a lo largo de los fragmentos procesados.
- **SyntheticSampler**: clase principal que coordina el proceso de lectura, muestreo, estadística y guardado.
- **compare_distributions**: función que contrasta distribuciones originales vs sintéticas para validar la calidad de la muestra.
- **main()**: punto de entrada para ejecutar el pipeline completo en un directorio.

Dependencias:
-------------
- pandas, numpy, scipy, tqdm, psutil (opcional), logging

Uso:
----
Este script está diseñado para ejecutarse como módulo principal (`__main__`) apuntando a un
directorio con archivos CSV:

    python creation_syntetic.py

Se recomienda ajustar la constante `DATA_FOLDER` en la sección final del script o integrarlo
dentro de una solución más amplia para procesamiento por lotes.

Autor: Sebastian David Pinzon Zambrano.
------
Desarrollado como parte de un sistema automatizado de muestreo sintético para análisis de grandes volúmenes de datos en entornos con recursos limitados.
"""

import glob
import json
import logging
import multiprocessing as mp  # type: ignore
import os
import shutil
import sys
import warnings
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, TypedDict, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas.api.types import (
    is_bool_dtype,  # type: ignore
    is_categorical_dtype,  # type: ignore
    is_datetime64_any_dtype,  # type: ignore
    is_numeric_dtype,  # type: ignore
    is_object_dtype,  # type: ignore
)
from scipy.stats import ks_2samp
from tqdm import tqdm


class NumericStats(TypedDict):
    """
    Diccionario tipado que almacena estadísticas numéricas acumuladas.

    Atributos:
        sum (float): Suma acumulada de los valores.
        sum2 (float): Suma acumulada de los cuadrados de los valores (útil para
        calcular la varianza).
        count (int): Cantidad de elementos acumulados.
        values (List[np.ndarray]): Lista de arrays NumPy con los valores individuales registrados.
            Puede ser un solo np.ndarray si se prefiere esa representación.
    """

    sum: float
    sum2: float
    count: int
    values: List[np.ndarray]  # o np.ndarray si es solo uno # type: ignore


try:
    import psutil
except ImportError:
    psutil = None
# Hacer que cada warning concreto solo aparezca una vez
warnings.filterwarnings("once", message="All-NaN slice encountered")
warnings.filterwarnings("once", message="Mean of empty slice")


# -------------------- Logging Setup --------------------
def setup_logger(log_path: Optional[str] = None) -> logging.Logger:
    """
    Configura un logger con salida a consola y, opcionalmente, a un archivo rotativo.

    El logger se llama "synthetic_sampler" y está configurado con los siguientes handlers:
    - Consola (`stdout`): Nivel WARNING o superior.
    - Archivo rotativo (si se proporciona `log_path`): Nivel INFO o superior, incluyendo DEBUG.

    Args:
        log_path (Optional[str]): Ruta al archivo de log. Si se proporciona, se escribe allí
                                con rotación de hasta 3 archivos de 5MB cada uno.

    Returns:
        logging.Logger: Instancia del logger configurado.
    """
    logger = logging.getLogger("synthetic_sampler")  # type: ignore
    logger.setLevel(logging.DEBUG)  # Captura todo, los handlers filtran por nivel

    # Handler de consola: muestra WARNING o superior
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    # Handler de archivo rotativo: guarda INFO y DEBUG si se proporciona log_path
    if log_path:
        fh = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3)
        fh.setLevel(logging.INFO)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d %(message)s"
            )
        )
        logger.addHandler(fh)

    return logger


logger = setup_logger("synthetic_sampler.log")


# -------------------- Utility Functions --------------------
def get_free_memory_mb() -> float:
    """
    Returns the amount of available memory in megabytes (MB).

    Tries to use `psutil` to get the available virtual memory. If `psutil` is not available,
    falls back to checking the free disk space in the current working directory.

    Returns:
        float: The amount of free memory (or disk space) in megabytes.

    Raises:
        RuntimeError: If the available memory or disk space cannot be determined.
    """
    try:
        if psutil:
            mem = psutil.virtual_memory()
            return mem.available / (1024 * 1024)
        else:
            _, _, free = shutil.disk_usage(os.getcwd())
            return free / (1024 * 1024)
    except Exception as e:
        logger.error("Error getting free memory: %s", e, exc_info=True)
        raise RuntimeError("Could not determine free memory.") from e


def get_file_size_mb(path: str) -> float:
    """
    Returns the size of a file in megabytes (MB).

    Args:
        path (str): The full path to the file.

    Returns:
        float: The size of the file in megabytes.

    Raises:
        Exception: If there is an error accessing the file size.
    """
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except Exception as e:
        logger.error("Error getting file size for %s: %s", path, e, exc_info=True)
        raise


def discover_csv_files(folder: str) -> List[str]:
    """
    Finds all `.csv` files in a given folder.

    Args:
        folder (str): The path to the directory to search.

    Returns:
        List[str]: A list of paths to `.csv` files found in the folder.
    """
    pattern = os.path.join(folder, "*.csv")
    files = glob.glob(pattern)
    return files


def safe_open_file(path: str, mode: str = "r") -> None:
    """
    Safely attempts to open a file to check if it is accessible.

    Args:
        path (str): The file path to open.
        mode (str): The mode in which to open the file (default is 'r').

    Raises:
        FileNotFoundError: If the file does not exist.
        PermissionError: If there are insufficient permissions to open the file.
        OSError: For other I/O-related errors.
    """
    try:
        with open(path, mode):
            pass
    except FileNotFoundError:
        logger.error("File not found: %s", path, exc_info=True)
        raise
    except PermissionError:
        logger.error("Permission denied: %s", path, exc_info=True)
        raise
    except OSError:
        logger.error("I/O error on file: %s", path, exc_info=True)
        raise


def get_num_workers() -> int:
    """
    Calcula el número óptimo de procesos a usar para paralelismo.

    Intenta obtener el número de CPUs disponibles y devuelve uno menos
    (dejando un CPU libre para el sistema). En caso de error, devuelve 1.

    Returns:
        int: Número de procesos recomendados para el pool de multiprocessing.
    """
    try:
        cpu_count = mp.cpu_count()
        return max(1, cpu_count - 1)
    except NotImplementedError as e:
        logger.error("CPU count not implemented on this platform: %s", e, exc_info=True)
        return 1


def print_startup_info(num_files: int, free_mem_mb: float) -> None:
    """
    Muestra por consola información inicial del pipeline.

    Imprime el número de archivos CSV detectados, la memoria libre disponible
    y un recordatorio de cómo interrumpir el proceso.

    Args:
        num_files (int): Cantidad de archivos CSV encontrados.
        free_mem_mb (float): Memoria libre disponible en MB.
    """
    print(
        f"Archivos CSV detectados: {num_files}\n"
        f"Memoria libre total: {free_mem_mb:.2f} MB\n"
        "Presione Ctrl+C para cancelar en cualquier momento.\n"
        "Ante interrupción, el archivo actual se abortará y continuará con el siguiente."
    )


# -------------------- Sampling Logic --------------------
class ChunkSampler:
    """
    Clase para muestrear filas de múltiples fragmentos (chunks) de un DataFrame de pandas
    hasta alcanzar un número total de filas deseado. Ideal para leer archivos CSV grandes
    por partes y obtener una muestra representativa sin cargar todo el archivo en memoria.

    Atributos:
        target_rows (int): Número total de filas que se desea muestrear.
        logger (logging.Logger): Logger para registrar errores y eventos.
        random_state (Optional[int]): Semilla para reproducibilidad en el muestreo aleatorio.
        sampled_chunks (List[pd.DataFrame]): Lista acumulada de fragmentos ya muestreados.
        rows_sampled (int): Total de filas acumuladas hasta el momento.
    """

    def __init__(
        self,
        target_rows: int,
        local_logger: logging.Logger,
        random_state: Optional[int] = None,
    ):
        """
        Inicializa el muestreador de fragmentos.

        Args:
            target_rows (int): Número total de filas a muestrear.
            local_logger (logging.Logger): Logger para errores y seguimiento.
            random_state (Optional[int]): Semilla para reproducibilidad (por defecto 42).
        """
        self.target_rows = target_rows
        self.logger = local_logger
        self.random_state = random_state or 42
        self.sampled_chunks: List[pd.DataFrame] = []
        self.rows_sampled = 0

    def sample_chunk(self, chunk: pd.DataFrame, remaining: int) -> pd.DataFrame:
        """
        Muestra un subconjunto del fragmento dado, sin reemplazo, respetando el número
        de filas restantes necesarias.

        Args:
            chunk (pd.DataFrame): Fragmento de datos a muestrear.
            remaining (int): Número de filas que aún se necesitan.

        Returns:
            pd.DataFrame: Subconjunto muestreado del fragmento original.
        """
        try:
            n = min(len(chunk), remaining)
            if n <= 0:
                return pd.DataFrame()
            sampled = chunk.sample(  # type: ignore
                n=n, replace=False, random_state=self.random_state  # type: ignore
            )  # type: ignore
            return sampled
        except Exception as e:
            self.logger.error(
                "Error sampling chunk: %s", e, exc_info=True, extra={"chunk": chunk}
            )
            raise

    def accumulate(self, chunk: pd.DataFrame):
        """
        Muestra un fragmento parcial del chunk y lo acumula, si aún no se ha
        alcanzado el total deseado de filas.

        Args:
            chunk (pd.DataFrame): Fragmento de datos a procesar.
        """
        remaining = self.target_rows - self.rows_sampled
        if remaining <= 0:
            return
        sampled = self.sample_chunk(chunk, remaining)
        self.sampled_chunks.append(sampled)
        self.rows_sampled += len(sampled)

    def get_sample(self) -> pd.DataFrame:
        """
        Devuelve el DataFrame combinado de todos los fragmentos muestreados.

        Returns:
            pd.DataFrame: Muestra combinada de los fragmentos procesados.
        """
        if self.sampled_chunks:
            return pd.concat(self.sampled_chunks, ignore_index=True)
        return pd.DataFrame()


class StatsCollector:
    """
    Clase para recolectar estadísticas de columnas en bloques de datos tipo pandas.DataFrame.

    Guarda estadísticas numéricas (incl. fechas y booleanos), categóricas y valores especiales
    como NaNs e infinitos. Permite resumirlas posteriormente.

    Atributos:
        logger (logging.Logger): Logger para registrar errores o advertencias.
        numeric_stats (Dict[str, Dict[str, Any]]): Estadísticas numéricas por columna.
        categorical_stats (Dict[str, Dict[str, Any]]): Frecuencias relativas por categoría.
        special_values (Dict[str, Dict[str, Any]]): Conteo de valores especiales (NaN, Inf, total).
    """

    def __init__(self, log: logging.Logger) -> None:
        """
        Inicializa la clase StatsCollector con un logger.

        Args:
            logger (logging.Logger): Objeto logger para registrar errores o advertencias.
        """
        self.logger = log
        self.numeric_stats: Dict[
            str, Dict[str, Union[int, float, List[NDArray[Any]]]]
        ] = {}
        self.categorical_stats: Dict[str, Dict[Any, float]] = {}
        self.special_values: Dict[str, Dict[str, int]] = {}

    def update(self, chunk: pd.DataFrame) -> None:
        """
        Actualiza las estadísticas acumuladas con un nuevo bloque de datos.

        Detecta el tipo de cada columna (numérico, categórico, datetime) y actualiza
        los contadores, sumas, percentiles o frecuencias relativas.

        También actualiza los contadores de valores especiales: NaN e infinito.

        Args:
            chunk (pd.DataFrame): Un DataFrame que contiene las columnas a analizar.
        """
        for col in chunk.columns:  # type: ignore
            # anotar tipo de col_raw para Pylance
            # pylint: disable=unsubscriptable-object
            col_raw: pd.Series[Any] = chunk[col]
            # pylint: enable=unsubscriptable-object
            try:
                # —————————— DATETIME ——————————
                if is_datetime64_any_dtype(col_raw):
                    col_data: pd.Series[np.int64] = col_raw.astype("int64") // 10**9
                    stats = self.numeric_stats.setdefault(
                        col,
                        {"count": 0, "sum": 0.0, "sum2": 0.0, "values": []},
                    )
                    stats["count"] += int(col_data.count())
                    stats["sum"] += float(col_data.sum())
                    stats["sum2"] += float((col_data**2).sum())
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            percentiles = np.nanpercentile(col_data, [10, 50, 90])
                    except Exception as e:
                        percentiles = np.array([np.nan, np.nan, np.nan])
                        self.logger.warning(
                            f"Percentiles datetime fallaron en {col}: {e}"
                        )
                    stats["values"].append(percentiles)

                    specials = self.special_values.setdefault(
                        col, {"nan": 0, "inf": 0, "total": 0}
                    )
                    specials["nan"] += int(col_data.isna().sum())
                    specials["inf"] += 0
                    specials["total"] += len(col_data)
                    continue

                # —————————— NUMERIC & BOOL ——————————
                if is_numeric_dtype(col_raw) or is_bool_dtype(col_raw):
                    if is_bool_dtype(col_raw):
                        col_data: pd.Series[float] = col_raw.astype(float)
                    else:
                        col_data = pd.to_numeric(col_raw, errors="coerce")

                    stats = self.numeric_stats.setdefault(
                        col,
                        {"count": 0.0, "sum": 0.0, "sum2": 0.0, "values": []},
                    )
                    stats["count"] += int(col_data.count())
                    stats["sum"] += float(col_data.sum(skipna=True))
                    stats["sum2"] += float((col_data**2).sum(skipna=True))

                    # <<< REEMPLAZA ESTE BLOQUE POR EL SIGUI >>>
                    # Solución robusta: solo calcular si hay datos no-NaN
                    clean = col_data.dropna()
                    if clean.size > 0:
                        try:
                            p = np.nanpercentile(clean, [10, 50, 90])
                            p = np.atleast_1d(p).flatten()
                            if p.shape != (3,):
                                raise ValueError(f"Percentiles inesperados: {p.shape}")
                        except Exception:
                            p = np.array([np.nan, np.nan, np.nan])
                    else:
                        # no hay datos válidos
                        p = np.array([np.nan, np.nan, np.nan])
                    stats["values"].append(p)

                # —————————— CATEGÓRICO ——————————
                elif is_categorical_dtype(col_raw) or is_object_dtype(col_raw):
                    freq: pd.Series = col_raw.value_counts(normalize=True, dropna=False)
                    cat_stats = self.categorical_stats.setdefault(col, {})
                    for category, rel_freq in freq.items():
                        cat_stats[category] = cat_stats.get(category, 0.0) + float(
                            rel_freq
                        )

                # —————————— VALORES ESPECIALES ——————————
                specials = self.special_values.setdefault(
                    col, {"nan": 0, "inf": 0, "total": 0}
                )
                specials["nan"] += int(col_raw.isna().sum())
                specials["inf"] += int(np.isinf(col_raw).sum()) if is_numeric_dtype(col_raw) else 0  # type: ignore
                specials["total"] += len(col_raw)

            except (TypeError, ValueError, KeyError) as e:
                # Capturamos sólo excepciones esperables de datos mal formados
                self.logger.error(
                    "Error collecting stats for column %s: %s", col, e, exc_info=True
                )
            # No capturamos Exception amplio para no esconder otros errores

    def summarize(self) -> Dict[str, Dict[str, Any]]:
        """
        Devuelve un resumen de las estadísticas recolectadas.

        Calcula la media, varianza y percentiles (10, 50, 90) para columnas numéricas.
        Calcula frecuencias relativas para columnas categóricas.
        Calcula proporciones de valores NaN e Inf para todas las columnas.

        Returns:
            Dict[str, Dict[str, Any]]: Diccionario con resumen de estadísticas por columna.
        """
        summary: Dict[str, Dict[str, Any]] = {}

        for col, stats in self.numeric_stats.items():
            try:
                mean = stats["sum"] / stats["count"] if stats["count"] else np.nan
                var = (
                    (stats["sum2"] / stats["count"] - mean**2)
                    if stats["count"]
                    else np.nan
                )

                # ———————— BLOQUE CORREGIDO DE PERCENTILES ————————
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    if stats["values"]:
                        arr = np.stack(stats["values"], axis=0)
                        # Solo promediar si hay al menos un valor no-NaN
                        if not np.all(np.isnan(arr)):
                            percentiles = np.nanmean(arr, axis=0)
                        else:
                            percentiles = np.array([np.nan, np.nan, np.nan])
                    else:
                        percentiles = np.array([np.nan, np.nan, np.nan])
                # ————— FIN BLOQUE PERCENTILES —————

                summary[col] = {
                    "mean": mean,
                    "var": var,
                    "percentiles": percentiles.tolist(),
                }
            except (TypeError, ValueError, KeyError) as e:
                self.logger.error(
                    "Error summarizing numeric stats for %s: %s", col, e, exc_info=True
                )

        for col, cats in self.categorical_stats.items():
            try:
                total = sum(cats.values())
                rel_freqs = {k: v / total for k, v in cats.items()} if total else {}
                summary[col] = {"rel_freqs": rel_freqs}
            except (TypeError, ValueError) as e:
                self.logger.error(
                    "Error summarizing categorical stats for %s: %s",
                    col,
                    e,
                    exc_info=True,
                )

        for col, specials in self.special_values.items():
            try:
                total = specials["total"]
                nan_prop = specials["nan"] / total if total else np.nan
                inf_prop = specials["inf"] / total if total else np.nan

                col_summary = summary.setdefault(col, {})
                col_summary["nan_prop"] = nan_prop
                col_summary["inf_prop"] = inf_prop
            except (TypeError, ZeroDivisionError) as e:
                self.logger.error(
                    "Error summarizing special values for %s: %s", col, e, exc_info=True
                )

        return summary


# -------------------- File Processing --------------------
class SyntheticSampler:
    """
    Clase encargada de generar una muestra sintética desde un archivo CSV de entrada.
    Utiliza muestreo por chunks, recolecta estadísticas y compara distribuciones.

    Atributos:
        input_path (str): Ruta al archivo CSV de entrada.
        output_path (str): Ruta donde se guardará la muestra generada.
        target_rows (int): Número objetivo de filas para la muestra.
        logger (logging.Logger): Logger para registrar eventos y errores.
        chunk_size_mb (float): Tamaño del chunk en megabytes.
        random_state (Optional[int]): Semilla aleatoria para reproducibilidad.
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        target_rows: int,
        logger: logging.Logger,
        chunk_size_mb: float,
        random_state: Optional[int] = None,
    ):
        """
        Inicializa la clase SyntheticSampler.

        Args:
            input_path (str): Ruta al archivo CSV de entrada.
            output_path (str): Ruta al archivo CSV de salida.
            target_rows (int): Cantidad de filas deseadas en la muestra sintética.
            logger (logging.Logger): Instancia de logger para registrar eventos.
            chunk_size_mb (float): Tamaño del chunk de lectura en megabytes.
            random_state (Optional[int], opcional): Semilla para la aleatoriedad.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.target_rows = target_rows
        self.logger = logger
        self.chunk_size_mb = chunk_size_mb
        self.random_state = random_state or 42

    def process(self) -> bool:
        """
        Procesa el archivo de entrada por chunks, genera una muestra aleatoria
        con estadísticas asociadas y guarda los resultados.

        Realiza los siguientes pasos:
        1. Estima tamaño de chunk según memoria disponible.
        2. Lee el archivo en chunks.
        3. Recolecta estadísticas con StatsCollector.
        4. Aplica muestreo con ChunkSampler.
        5. Guarda el CSV de salida.
        6. Escribe archivo JSON con estadísticas.
        7. Compara la distribución original vs la muestra y guarda el resultado.

        Returns:
            bool: True si el proceso fue exitoso, False si hubo errores críticos.
        """
        try:
            safe_open_file(self.input_path, "r")
            file_size = get_file_size_mb(self.input_path)
            free_mem = get_free_memory_mb()
        except (OSError, IOError) as e:
            self.logger.error(
                "Skipping file due to I/O error: %s (%s)",
                self.input_path,
                str(e),
                exc_info=True,
            )
            return False

        try:
            buffer_mb = 200  # MB buffer for safety
            usable_mem = max(1, free_mem - buffer_mb)
            est_chunk_size = min(
                int(usable_mem * 0.5 * 1024 * 1024), int(file_size * 1024 * 1024)
            )
            chunk_bytes = max(est_chunk_size, 10 * 1024 * 1024)
            chunk_rows = 100_000  # Ajustable dinámicamente si quieres
        except Exception as e:
            self.logger.error(
                "Error calculating chunk size for %s: %s",
                self.input_path,
                e,
                exc_info=True,
            )
            chunk_rows = 100_000

        sampler = ChunkSampler(self.target_rows, self.logger, self.random_state)
        stats_collector = StatsCollector(self.logger)

        try:
            try:
                total_rows = sum(1 for _ in open(self.input_path, encoding="utf-8")) - 1
            except Exception:
                total_rows = None

            with pd.read_csv(
                self.input_path,
                chunksize=chunk_rows,
                iterator=True,
                low_memory=False,
                encoding="utf-8",
            ) as reader:
                pbar = tqdm(
                    desc=f"Procesando {os.path.basename(self.input_path)}",
                    total=total_rows,
                    unit="rows",
                    leave=False,
                )
                for chunk in reader:
                    try:
                        stats_collector.update(chunk)
                        sampler.accumulate(chunk)
                        pbar.update(len(chunk))
                        if sampler.rows_sampled >= self.target_rows:
                            break
                    except (RuntimeWarning, pd.errors.DtypeWarning) as e:
                        self.logger.warning(
                            "Warning during chunk processing: %s", e, exc_info=True
                        )
                    except (IndexError, KeyError, ValueError) as e:
                        self.logger.error(
                            "Indexing error in chunk: %s", e, exc_info=True
                        )
                    except Exception:
                        self.logger.exception("Critical error processing chunk")
                pbar.close()
        except KeyboardInterrupt:
            self.logger.warning(
                "Interrupción detectada. Abortando archivo: %s", self.input_path
            )
            return False
        except Exception as e:
            self.logger.error(
                "Error reading file %s: %s", self.input_path, e, exc_info=True
            )
            return False

        sample_df = sampler.get_sample()
        if len(sample_df) > self.target_rows:
            sample_df = sample_df.iloc[: self.target_rows]

        try:
            sample_df.to_csv(self.output_path, index=False, encoding="utf-8")
        except (IOError, OSError) as e:
            self.logger.error(
                "Error writing output file %s: %s", self.output_path, e, exc_info=True
            )
            return False
        except Exception as e:
            self.logger.error(
                "Unknown error writing output file %s: %s",
                self.output_path,
                e,
                exc_info=True,
            )
            return False

        stats = stats_collector.summarize()
        stats_path = self.output_path.replace(".csv", "_stats.json")
        try:
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, default=str)
        except Exception as e:
            self.logger.warning(
                "Could not write stats file for %s: %s", self.input_path, e
            )

        try:
            compare_results = compare_distributions(self.input_path, self.output_path)
            compare_path = self.output_path.replace(".csv", "_compare.json")
            with open(compare_path, "w", encoding="utf-8") as f:
                json.dump(compare_results, f, indent=2, default=str)
            self.logger.info("Comparación guardada en: %s", compare_path)
        except Exception as e:
            self.logger.warning(
                "No se pudo comparar distribuciones para %s: %s",
                self.input_path,
                e,
            )

        self.logger.info(
            "Archivo procesado: %s. Muestra sintética guardada en: %s",
            self.input_path,
            self.output_path,
        )
        return True


def process_file_worker(args):
    """
    Worker para procesar un archivo utilizando un objeto SyntheticSampler.

    Esta función se diseña para ejecutarse en un pool de procesos. Recibe como argumento
    una instancia de SyntheticSampler, invoca su método `process` y captura cualquier
    excepción que ocurra durante la ejecución.

    Args:
        args (SyntheticSampler): Instancia de SyntheticSampler que contiene la lógica
            y los parámetros necesarios para procesar un archivo CSV.

    Returns:
        bool: Devuelve True si `sampler.process()` finaliza correctamente, o False
            si ocurre cualquier error durante el procesamiento.
    """
    sampler: SyntheticSampler = args
    try:
        return sampler.process()
    except Exception as e:
        logger.error("Worker error: %s", e, exc_info=True)
        return False


# -------------------- Main Pipeline --------------------
def main(
    data_folder: str,
    target_rows: int = 50_000,
    logger: logging.Logger = logger,
):
    """
    Punto de entrada principal del pipeline de muestreo sintético.

    Este método realiza los siguientes pasos:
      1. Descubre archivos CSV en el directorio proporcionado.
      2. Calcula la memoria libre disponible y muestra información de inicio.
      3. Crea un pool de procesos para paralelizar el procesamiento (si es posible).
      4. Inicializa instancias de `SyntheticSampler` para cada archivo.
      5. Ejecuta el muestreo en paralelo (o secuencialmente si el pool falla).
      6. Muestra una barra de progreso y acumula resultados.
      7. Registra el resumen final de archivos procesados exitosamente.

    Args:
        data_folder (str): Ruta al directorio donde buscar archivos CSV.
        target_rows (int, opcional): Número de filas objetivo para cada muestra sintética.
            Por defecto 50_000.
        logger (logging.Logger, opcional): Logger para registrar eventos, advertencias y errores.
            Por defecto, el logger configurado globalmente.

    Returns:
        None: Este método no retorna valor; en caso de error crítico, lo registra en el logger.
    """
    try:
        files = discover_csv_files(data_folder)
        free_mem = get_free_memory_mb()
        print_startup_info(len(files), free_mem)
        if not files:
            logger.warning("No CSV files found in %s", data_folder)
            return

        num_workers = get_num_workers()
        logger.info("Usando %d procesos en paralelo.", num_workers)
        pool = None
        try:
            pool = mp.get_context("spawn").Pool(num_workers)
        except (OSError, ValueError) as e:
            logger.error("Error starting multiprocessing pool: %s", e, exc_info=True)
            pool = None

        tasks = []
        for file_path in files:
            base, ext = os.path.splitext(file_path)
            out_path = f"{base}_syn{ext}"
            sampler = SyntheticSampler(
                input_path=file_path,
                output_path=out_path,
                target_rows=target_rows,
                logger=logger,
                chunk_size_mb=50,
            )
            tasks.append(sampler)

        results = []
        with tqdm(total=len(tasks), desc="Archivos procesados", unit="file") as pbar:
            if pool:
                for ok in pool.imap_unordered(process_file_worker, tasks):
                    pbar.update(1)
                    results.append(ok)
            else:
                for sampler in tasks:
                    ok = sampler.process()
                    pbar.update(1)
                    results.append(ok)
        logger.info(
            "Procesamiento completado. %d/%d archivos procesados exitosamente.",
            sum(results),
            len(results),
        )
    except KeyboardInterrupt:
        logger.warning("Interrupción detectada por el usuario. Saliendo.")
    except Exception as e:
        logger.error("Error crítico en el pipeline: %s", e, exc_info=True)


def compare_distributions(
    orig_path: str, syn_path: str, chunk_rows: int = 100_000
) -> Dict[str, Any]:
    """
    Compara distribuciones en streaming, procesando CSV por chunks
    para evitar cargar todo en memoria.
    """
    # Detectar si existe columna 'date' (para no forzar parse_dates si no existe)
    cols = pd.read_csv(orig_path, nrows=0).columns.tolist()
    has_date = "date" in cols

    # Inferimos tipos de columna antes de procesar
    data_types = infer_column_types(orig_path)
    stats_o = StatsCollector(logger)
    stats_s = StatsCollector(logger)

    if has_date:
        dates_o = []
        for chunk in pd.read_csv(
            orig_path,
            chunksize=chunk_rows,
            iterator=True,
            low_memory=False,
            encoding="utf-8",
            parse_dates=["date"],
        ):
            stats_o.update(chunk)
            dates_o.append(chunk["date"].astype("int64") // 10**9)

        dates_s = []
        for chunk in pd.read_csv(
            syn_path,
            chunksize=chunk_rows,
            iterator=True,
            low_memory=False,
            encoding="utf-8",
            parse_dates=["date"],
        ):
            stats_s.update(chunk)
            dates_s.append(chunk["date"].astype("int64") // 10**9)

        # KS test sobre timestamps

        stat, _ = ks_2samp(np.concatenate(dates_o), np.concatenate(dates_s))
        results: Dict[str, Any] = {
            "date": {"sim_ks": 1.0 - stat, "sim_nan": 1.0, "sim_inf": 1.0}
        }
    else:
        # — Si no hay 'date', actualizamos stats para todas las columnas —
        for chunk in pd.read_csv(
            orig_path,
            chunksize=chunk_rows,
            iterator=True,
            low_memory=False,
            encoding="utf-8",
        ):
            stats_o.update(chunk)
        for chunk in pd.read_csv(
            syn_path,
            chunksize=chunk_rows,
            iterator=True,
            low_memory=False,
            encoding="utf-8",
        ):
            stats_s.update(chunk)
        # Inicializamos el dict de resultados vacío; luego vendrá el loop sobre sum_o
        results: Dict[str, Any] = {}

    # Resto del procesamiento...
    sum_o = stats_o.summarize()
    sum_s = stats_s.summarize()
    for col, so in sum_o.items():
        if col == "date":
            continue
        ss = sum_s.get(col, {})
        col_res: Dict[str, float] = {}
        # Similitud de NaN
        nan_o = so.get("nan_prop", np.nan)
        nan_s = ss.get("nan_prop", np.nan)
        col_res["sim_nan"] = 1.0 - abs(nan_o - nan_s)
        # Similitud de Inf
        inf_o = so.get("inf_prop", np.nan)
        inf_s = ss.get("inf_prop", np.nan)
        col_res["sim_inf"] = 1.0 - abs(inf_o - inf_s)
        # Similitud de percentiles
        pct_o = np.array(so.get("percentiles", [np.nan, np.nan, np.nan]), dtype=float)
        pct_s = np.array(ss.get("percentiles", [np.nan, np.nan, np.nan]), dtype=float)
        denom = np.where(pct_o != 0, np.abs(pct_o), 1.0)
        col_res["sim_pct"] = 1.0 - np.mean(np.abs(pct_o - pct_s) / denom)
        results[col] = col_res

    # ————————————————————————————————————————————————————————————————————————————————
    # Añadir metadato data_type y forzar sim_pct=null en strings
    for col, metrics in results.items():
        # 1) Etiquetamos el tipo detectado
        metrics["data_type"] = data_types.get(col, "string")
        # 2) Si es string, dejamos sim_pct explícitamente null
        if metrics["data_type"] == "string":
            metrics["sim_pct"] = None
    # ————————————————————————————————————————————————————————————————————————————————
    return results


# ————————————————————————————————————————————————————————————————————————————————
# Función de inferencia de tipos por columna (solo primeras n filas)
def infer_column_types(orig_path: str, sample_rows: int = 1000) -> Dict[str, str]:
    """Detecta data_type en {datetime, boolean, numeric, string} leyendo un muestreo."""
    df_sample: pd.DataFrame = pd.read_csv(  # type: ignore
        orig_path,
        nrows=sample_rows,
        low_memory=False,
    )
    types: Dict[str, str] = {}
    for col in df_sample.columns:
        serie = df_sample[col]
        if is_datetime64_any_dtype(serie):
            types[col] = "datetime"
        elif is_bool_dtype(serie):
            types[col] = "boolean"
        elif is_numeric_dtype(serie):
            types[col] = "numeric"
        else:
            types[col] = "string"
    return types


# ————————————————————————————————————————————————————————————————————————————————
if __name__ == "__main__":
    DATA_FOLDER = r"D:\Portafolio oficial\Retail Sales Trend Analysis\data\data\raw"
    main(DATA_FOLDER)
