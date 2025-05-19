import os
import sys
import glob
import warnings
from logging.handlers import RotatingFileHandler
import math
import logging
import traceback
import multiprocessing as mp  # type: ignore
from typing import List, Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np
from tqdm import tqdm

# Hacer que cada warning concreto solo aparezca una vez
warnings.filterwarnings("once", message="All-NaN slice encountered")
warnings.filterwarnings("once", message="Mean of empty slice")
import shutil
import json
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from scipy.spatial import distance
from pandas.api.types import (
    is_bool_dtype,
    is_numeric_dtype,
    is_categorical_dtype,
    is_object_dtype,
    is_datetime64_any_dtype,
)

try:
    import psutil
except ImportError:
    psutil = None


# -------------------- Logging Setup --------------------
def setup_logger(log_path: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("synthetic_sampler")
    logger.setLevel(logging.DEBUG)  # Capturamos todo, filtramos por handler

    # 1) Consola: WARNING en adelante
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    # 2) Archivo rotativo: INFO y DEBUG
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
    try:
        if psutil:
            mem = psutil.virtual_memory()
            return mem.available / (1024 * 1024)
        else:

            total, used, free = shutil.disk_usage(os.getcwd())
            return free / (1024 * 1024)
    except Exception as e:
        logger.error("Error getting free memory: %s", e, exc_info=True)
        raise RuntimeError("Could not determine free memory.") from e


def get_file_size_mb(path: str) -> float:
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except Exception as e:
        logger.error("Error getting file size for %s: %s", path, e, exc_info=True)
        raise


def discover_csv_files(folder: str) -> List[str]:
    pattern = os.path.join(folder, "*.csv")
    files = glob.glob(pattern)
    return files


def safe_open_file(path: str, mode: str = "r"):
    try:
        with open(path, mode):
            pass
    except FileNotFoundError as e:
        logger.error("File not found: %s", path, exc_info=True)
        raise
    except PermissionError as e:
        logger.error("Permission denied: %s", path, exc_info=True)
        raise
    except OSError as e:
        logger.error("I/O error on file: %s", path, exc_info=True)
        raise


def get_num_workers() -> int:
    try:
        cpu_count = mp.cpu_count()
        return max(1, cpu_count - 1)
    except Exception as e:
        logger.error("Error getting CPU count: %s", e, exc_info=True)
        return 1


def print_startup_info(num_files: int, free_mem_mb: float):
    print(
        f"Archivos CSV detectados: {num_files}\n"
        f"Memoria libre total: {free_mem_mb:.2f} MB\n"
        "Presione Ctrl+C para cancelar en cualquier momento.\n"
        "Ante interrupción, el archivo actual se abortará y continuará con el siguiente."
    )


# -------------------- Sampling Logic --------------------
class ChunkSampler:
    def __init__(
        self,
        target_rows: int,
        logger: logging.Logger,
        random_state: Optional[int] = None,
    ):
        self.target_rows = target_rows
        self.logger = logger
        self.random_state = random_state or 42
        self.sampled_chunks: List[pd.DataFrame] = []
        self.rows_sampled = 0

    def sample_chunk(self, chunk: pd.DataFrame, remaining: int) -> pd.DataFrame:
        try:
            n = min(len(chunk), remaining)
            if n <= 0:
                return pd.DataFrame()
            # Sample without replacement
            sampled = chunk.sample(n=n, replace=False, random_state=self.random_state)
            return sampled
        except Exception as e:
            self.logger.error(
                "Error sampling chunk: %s", e, exc_info=True, extra={"chunk": chunk}
            )
            raise

    def accumulate(self, chunk: pd.DataFrame):
        remaining = self.target_rows - self.rows_sampled
        if remaining <= 0:
            return
        sampled = self.sample_chunk(chunk, remaining)
        self.sampled_chunks.append(sampled)
        self.rows_sampled += len(sampled)

    def get_sample(self) -> pd.DataFrame:
        if self.sampled_chunks:
            return pd.concat(self.sampled_chunks, ignore_index=True)
        return pd.DataFrame()


class StatsCollector:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.numeric_stats: Dict[str, Dict[str, Any]] = {}
        self.categorical_stats: Dict[str, Dict[str, Any]] = {}
        self.special_values: Dict[str, Dict[str, Any]] = {}

    def update(self, chunk: pd.DataFrame):
        for col in chunk.columns:
            try:
                col_raw = chunk[col]
                # —————————— NUEVA RAMA: DATETIME ——————————
                # Si es datetime, lo convertimos a segundos desde epoch
                from pandas.api.types import is_datetime64_any_dtype

                if is_datetime64_any_dtype(col_raw):
                    # Los nanosegundos / 10**9 → segundos
                    col_data = col_raw.astype("int64") // 10**9
                    stats = self.numeric_stats.setdefault(
                        col, {"count": 0, "sum": 0.0, "sum2": 0.0, "values": []}
                    )
                    stats["count"] += int(col_data.count())
                    stats["sum"] += float(col_data.sum())
                    stats["sum2"] += float((col_data**2).sum())
                    # Calcular percentiles suprimiendo warnings de “All-NaN slice”
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            percentiles = np.nanpercentile(col_data, [10, 50, 90])
                    except Exception as e:
                        percentiles = [np.nan] * 3
                        self.logger.warning(f"Percentiles fecha fallaron en {col}: {e}")
                    stats["values"].append(percentiles)
                    # Luego procesamos specials (NaN no aplica para datetime)
                    specials = self.special_values.setdefault(
                        col, {"nan": 0, "inf": 0, "total": 0}
                    )
                    specials["nan"] += int(col_data.isna().sum())
                    specials["inf"] += 0
                    specials["total"] += len(col_data)
                    # Saltar al siguiente col
                    continue
                # ————————————————————————————————————————————————
                # ================= Numeric (incluye bool) =================
                if is_numeric_dtype(col_raw) or is_bool_dtype(col_raw):
                    # Convertir booleanos a float (True→1.0, False→0.0)
                    if is_bool_dtype(col_raw):
                        col_data = col_raw.astype(float)
                    else:
                        col_data = pd.to_numeric(col_raw, errors="coerce")

                    stats = self.numeric_stats.setdefault(
                        col, {"count": 0, "sum": 0.0, "sum2": 0.0, "values": []}
                    )
                    stats["count"] += int(col_data.count())
                    stats["sum"] += float(col_data.sum(skipna=True))
                    stats["sum2"] += float((col_data**2).sum(skipna=True))

                    # Calcular percentiles suprimiendo warnings de “All-NaN slice”
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            percentiles = np.nanpercentile(col_data, [10, 50, 90])
                    except Exception as e:
                        percentiles = [np.nan] * 3
                        self.logger.warning(f"Percentiles fecha fallaron en {col}: {e}")
                    stats["values"].append(percentiles)

                # ============== Categorical / Object =====================
                elif is_categorical_dtype(col_raw) or is_object_dtype(col_raw):
                    freq = col_raw.value_counts(normalize=True, dropna=False)
                    cat_stats = self.categorical_stats.setdefault(col, {})
                    for cat, rel_freq in freq.items():
                        cat_stats[cat] = cat_stats.get(cat, 0.0) + float(rel_freq)

                # ============== Special Values (NaN / Inf) ===============
                # Esto se aplica a todas las columnas
                specials = self.special_values.setdefault(
                    col, {"nan": 0, "inf": 0, "total": 0}
                )
                specials["nan"] += int(col_raw.isna().sum())
                specials["inf"] += int(
                    np.isinf(col_raw).sum() if is_numeric_dtype(col_raw) else 0
                )
                specials["total"] += int(len(col_raw))

            except Exception as e:
                self.logger.error(
                    "Error collecting stats for column %s: %s", col, e, exc_info=True
                )

    def summarize(self) -> Dict[str, Any]:
        summary = {}
        for col, stats in self.numeric_stats.items():
            try:
                mean = stats["sum"] / stats["count"] if stats["count"] else np.nan
                var = (
                    (stats["sum2"] / stats["count"] - mean**2)
                    if stats["count"]
                    else np.nan
                )

                # Suprimir warnings aquí sólo para la media de percentiles
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    percentiles = (
                        np.nanmean(stats["values"], axis=0)
                        if stats["values"]
                        else [np.nan, np.nan, np.nan]
                    )
                summary[col] = {
                    "mean": mean,
                    "var": var,
                    "percentiles": percentiles,
                }
            except Exception as e:
                self.logger.error(
                    "Error summarizing numeric stats for %s: %s",
                    col,
                    e,
                    exc_info=True,
                )
        for col, cats in self.categorical_stats.items():
            try:
                total = sum(cats.values())
                rel_freqs = {k: v / total for k, v in cats.items()} if total else {}
                summary[col] = {"rel_freqs": rel_freqs}
            except Exception as e:
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
                summary.setdefault(col, {}).update(
                    {"nan_prop": nan_prop, "inf_prop": inf_prop}
                )
            except Exception as e:
                self.logger.error(
                    "Error summarizing special values for %s: %s",
                    col,
                    e,
                    exc_info=True,
                )
        return summary


# -------------------- File Processing --------------------
class SyntheticSampler:
    def __init__(
        self,
        input_path: str,
        output_path: str,
        target_rows: int,
        logger: logging.Logger,
        chunk_size_mb: float,
        random_state: Optional[int] = None,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.target_rows = target_rows
        self.logger = logger
        self.chunk_size_mb = chunk_size_mb
        self.random_state = random_state or 42

    def process(self) -> bool:
        try:
            safe_open_file(self.input_path, "r")
            file_size = get_file_size_mb(self.input_path)
            free_mem = get_free_memory_mb()
        except Exception as e:
            self.logger.error(
                "Skipping file due to I/O error: %s", self.input_path, exc_info=True
            )
            return False

        # Estimate chunk size
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

        # Lectura por chunks con barra de progreso
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
                        # Captura full traceback en el log de archivo
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

        # Concatenar y recortar la muestra
        sample_df = sampler.get_sample()
        if len(sample_df) > self.target_rows:
            sample_df = sample_df.iloc[: self.target_rows]

        # Guardar CSV sintético
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

        # Guardar estadísticas de chunking
        stats = stats_collector.summarize()
        stats_path = self.output_path.replace(".csv", "_stats.json")
        try:
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, default=str)
        except Exception as e:
            self.logger.warning(
                "Could not write stats file for %s: %s", self.input_path, e
            )

        # ——— NUEVO BLOQUE: comparar distribuciones y guardar resultado ———
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
        from scipy.stats import ks_2samp

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
    df_sample = pd.read_csv(
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
