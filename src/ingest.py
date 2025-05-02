"""
Module: ingestion_pipeline.py
Description: Chunked ingestion of CSV/Parquet data into a PostgreSQL database with
Pandera validation using credentials from config.py.
"""

import sys
import logging
from typing import Iterator, Optional

import pandas as pd
import pandera as pa
from pandera.typing import Series
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from config import config

# Load database configuration
try:
    db_cfg = config["database"]
    host = db_cfg["host"]
    port = db_cfg["port"]
    user = db_cfg["user"]
    password = db_cfg["password"]
    database = db_cfg["database"]
except KeyError as exc:
    logging.error("Missing database configuration key: %s", exc)
    sys.exit(1)

# Construct the database URL
DB_URL: str = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def get_database_engine(db_url: Optional[str] = None) -> Engine:
    """
    Create and return a SQLAlchemy engine, verifying the connection.

    :param db_url: Optional database URL override.
    :return: Connected SQLAlchemy Engine.
    """
    url = db_url or DB_URL
    try:
        engine = create_engine(
            url,
            pool_pre_ping=True,
            connect_args={"client_encoding": "UTF8"},
        )
        # Test connection
        with engine.connect():  # type: ignore
            logger.info("Database connection successful.")
        return engine
    except SQLAlchemyError as exc:
        logger.error("Failed to connect to the database.", exc_info=exc)
        sys.exit(1)


def read_in_chunks(
    file_path: str, chunk_size: int
) -> Iterator[pd.DataFrame]:  # pylint: disable=too-many-branches
    """
    Yield chunks of data from a CSV or Parquet file.

    :param file_path: Path to the CSV or Parquet file.
    :param chunk_size: Number of rows per chunk.
    :return: Iterator over DataFrame chunks.
    :raises ValueError: If file extension is unsupported.
    """
    lower_path = file_path.lower()
    if lower_path.endswith(".csv"):
        with open(file_path, mode="r", encoding="utf-8", errors="ignore") as file_obj:
            for chunk in pd.read_csv(file_obj, chunksize=chunk_size):
                yield chunk
    elif lower_path.endswith((".parquet", ".parq", ".pqt")):
        df = pd.read_parquet(file_path)
        total_rows = len(df)
        for start in range(0, total_rows, chunk_size):
            yield df.iloc[start : start + chunk_size]
    else:
        raise ValueError(f"Unsupported file type: {file_path}")


class StoresSchema(pa.DataFrameModel):  # pylint: disable=too-few-public-methods
    """Pandera schema for the 'stores' table."""

    store_nbr: Series[int] = pa.Field(ge=0)
    city: Series[str]
    state: Series[str]
    type: Series[str]
    cluster: Series[int]

    class Config:  # pylint: disable=all
        coerce = True


class ItemsSchema(pa.DataFrameModel):  # pylint: disable=too-few-public-methods
    """Pandera schema for the 'items' table."""

    item_nbr: Series[int] = pa.Field(ge=0)
    family: Series[str]
    class_: Series[int] = pa.Field(alias="class")
    perishable: Series[bool]

    class Config:  # pylint: disable=all
        coerce = True
        strict = True


class TransactionsSchema(pa.DataFrameModel):  # pylint: disable=too-few-public-methods
    """Pandera schema for the 'transactions' table."""

    date: Series[pd.Timestamp]
    store_nbr: Series[int]
    transactions: Series[int]

    class Config:  # pylint: disable=all
        coerce = True


class OilSchema(pa.DataFrameModel):  # pylint: disable=too-few-public-methods
    """Pandera schema for the 'oil' table."""

    date: Series[pd.Timestamp]
    dcoilwtico: Series[float]

    class Config:  # pylint: disable=all
        coerce = True


class HolidaysSchema(pa.DataFrameModel):  # pylint: disable=too-few-public-methods
    """Pandera schema for the 'holidays_events' table."""

    date: Series[pd.Timestamp]
    type: Series[str]
    locale: Series[str]
    locale_name: Series[str]
    description: Series[str]
    transferred: Series[bool]

    class Config:  # pylint: disable=all
        coerce = True


class SampleSubmissionSchema(
    pa.DataFrameModel
):  # pylint: disable=too-few-public-methods
    """Pandera schema for the 'sample_submission' table."""

    id_: Series[int] = pa.Field(alias="id")
    unit_sales: Series[float]

    class Config:  # pylint: disable=all
        coerce = True
        strict = True


class TrainSchema(pa.DataFrameModel):  # pylint: disable=too-few-public-methods
    """Pandera schema for the 'train' table."""

    date: Series[pd.Timestamp]
    store_nbr: Series[int]
    item_nbr: Series[int]
    unit_sales: Series[float]
    onpromotion: Series[bool]

    class Config:  # pylint: disable=all
        coerce = True
        strict = True


class TestSchema(pa.DataFrameModel):  # pylint: disable=too-few-public-methods
    """Pandera schema for the 'test' table."""

    date: Series[pd.Timestamp]
    store_nbr: Series[int]
    item_nbr: Series[int]
    onpromotion: Series[bool]

    class Config:  # pylint: disable=all
        coerce = True
        strict = True


SCHEMA_MAP: dict[str, type[pa.DataFrameModel]] = {
    "stores": StoresSchema,
    "items": ItemsSchema,
    "transactions": TransactionsSchema,
    "oil": OilSchema,
    "holidays_events": HolidaysSchema,
    "sample_submission": SampleSubmissionSchema,
    "train": TrainSchema,
    "test": TestSchema,
}


def validate_schema(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """
    Validate a DataFrame against its Pandera schema.

    :param df: DataFrame to validate.
    :param table_name: Key in SCHEMA_MAP for schema lookup.
    :return: Validated (and coerced) DataFrame.
    :raises ValueError: If no schema is defined.
    """
    schema_model = SCHEMA_MAP.get(table_name)
    if schema_model is None:
        raise ValueError(f"No schema defined for table '{table_name}'")

    # Rename columns based on field aliases
    alias_map: dict[str, str] = {
        field[1].alias: name
        for name, field in schema_model.__fields__.items()
        if field[1].alias
    }
    df_renamed = df.rename(columns=alias_map)
    return schema_model.validate(df_renamed)


def insert_chunk(
    engine: Engine, df: pd.DataFrame, table_name: str, schema_name: str = "favorita"
) -> None:
    """
    Insert a DataFrame chunk into the specified database table.

    :param engine: SQLAlchemy Engine.
    :param df: DataFrame to insert.
    :param table_name: Table name in the database.
    :param schema_name: Database schema name.
    """
    try:
        with engine.begin() as conn:
            conn.execute(text(f"SET search_path TO {schema_name}, public;"))
            df.to_sql(
                name=table_name,
                con=conn,
                schema=schema_name,
                if_exists="append",
                index=False,
            )
    except (UnicodeDecodeError, SQLAlchemyError) as exc:
        logger.error(
            "Error inserting into '%s.%s': %s",
            schema_name,
            table_name,
            exc,
            exc_info=exc,
        )
        raise


def run(file_path: str, table_name: str, chunk_size: int = 10_000) -> None:
    """
    Execute the ingestion pipeline: read, validate, and insert data in chunks.

    :param file_path: Path to the data file.
    :param table_name: Target table name in the database.
    :param chunk_size: Number of rows per chunk.
    """
    engine = get_database_engine()
    logger.info(
        "Starting ingestion of '%s' into table '%s' with chunk size %d.",
        file_path,
        table_name,
        chunk_size,
    )
    total_rows = 0

    for index, chunk in enumerate(read_in_chunks(file_path, chunk_size), start=1):
        try:
            validated_df = validate_schema(chunk, table_name)
            insert_chunk(engine, validated_df, table_name)
            rows = len(validated_df)
            total_rows += rows
            logger.info("Chunk %d: inserted %d rows.", index, rows)
        except (ValueError, SQLAlchemyError, UnicodeDecodeError) as exc:
            logger.error("Chunk %d failed: %s", index, exc, exc_info=exc)

    logger.info("Ingestion completed: total %d rows processed.", total_rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest CSV or Parquet into PostgreSQL with validation."
    )
    parser.add_argument("file_path", help="Path to the CSV or Parquet file.")
    parser.add_argument(
        "table_name", help="Destination table name in schema 'favorita'."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10_000,
        help="Number of rows to process per chunk.",
    )
    args = parser.parse_args()
    run(args.file_path, args.table_name, args.chunk_size)
