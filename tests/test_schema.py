import os
import pytest
import psycopg2

DB_DSN = os.getenv("DB_DSN", "postgresql://postgres:postgres@localhost:5432/postgres")


@pytest.fixture(scope="module")
def conn():
    """
    Context manager-style generator that yields a PostgreSQL database connection.

    Establishes a connection using the DSN provided in `DB_DSN`, sets autocommit mode
    to True, and yields the connection object. Once the context is exited, the
    connection is automatically closed.

    Yields:
        psycopg2.extensions.connection: An active PostgreSQL database connection.
    """
    cn = psycopg2.connect(DB_DSN)
    cn.autocommit = True
    yield cn
    cn.close()


def table_exists(cur, schema, table):
    """
    Check if a table exists in a specific schema within the connected PostgreSQL database.

    Executes a query against the `information_schema.tables` to determine whether
    the specified table exists in the given schema.

    Args:
        cur (psycopg2.extensions.cursor): An active database cursor.
        schema (str): The name of the schema to search within.
        table (str): The name of the table to check for existence.

    Returns:
        bool: True if the table exists, False otherwise.
    """
    cur.execute(
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s
        );
        """,
        (schema, table),
    )
    return cur.fetchone()[0]


def index_exists(cur, index_name):
    """
    Check if an index exists in the connected PostgreSQL database.

    Queries the PostgreSQL system catalogs (`pg_class` and `pg_namespace`) to
    determine whether an index with the specified name exists.

    Args:
        cur (psycopg2.extensions.cursor): An active database cursor.
        index_name (str): The name of the index to check for existence.

    Returns:
        bool: True if the index exists, False otherwise.
    """
    cur.execute(
        """
        SELECT EXISTS (
            SELECT 1 FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relkind = 'i' AND c.relname = %s
        );
        """,
        (index_name,),
    )
    return cur.fetchone()[0]


def test_extension_timescaledb(db_conn):
    """
    Verifies that the TimescaleDB extension is enabled in the connected PostgreSQL database.

    Executes a query against `pg_extension` to check for the presence of the
    'timescaledb' extension. Raises an assertion error if the extension is not found.

    Args:
        db_conn (psycopg2.extensions.connection): An active database connection.

    Raises:
        AssertionError: If the TimescaleDB extension is not enabled.
    """
    cur = db_conn.cursor()
    try:
        cur.execute("SELECT extname FROM pg_extension WHERE extname = 'timescaledb';")
        assert cur.fetchone() is not None, "La extensión timescaledb no está habilitada"
    finally:
        cur.close()


def test_schema_favorita(db_conn):
    """
    Verifies that the 'favorita' schema exists in the connected PostgreSQL database.

    Executes a query against `information_schema.schemata` to check whether
    the schema named 'favorita' exists. Raises an assertion error if it does not.

    Args:
        db_conn (psycopg2.extensions.connection): An active database connection.

    Raises:
        AssertionError: If the 'favorita' schema does not exist.
    """
    cur = db_conn.cursor()
    try:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.schemata
                WHERE schema_name = 'favorita'
            );
            """
        )
        assert cur.fetchone()[0], "El schema 'favorita' no existe"
    finally:
        cur.close()


@pytest.mark.parametrize(
    "table",
    [
        "stores",
        "items",
        "transactions",
        "oil",
        "holidays_events",
        "sample_submission",
        "train",
        "test",
    ],
)
def test_tables_exist(db_conn, table):
    """
    Verifies that a specific table exists in the 'favorita' schema of the PostgreSQL database.

    Uses the `table_exists` helper function to determine if the table is present.
    Raises an assertion error if the table does not exist.

    Args:
        db_conn (psycopg2.extensions.connection): An active database connection.
        table (str): The name of the table to check within the 'favorita' schema.

    Raises:
        AssertionError: If the specified table does not exist.
    """
    cur = db_conn.cursor()
    try:
        assert table_exists(cur, "favorita", table), f"Falta tabla favorita.{table}"
    finally:
        cur.close()


def test_hypertable_train(db_conn):
    """
    Verifies that the table 'favorita.train' is registered as a hypertable in TimescaleDB.

    Queries the TimescaleDB internal catalog to determine whether the table
    'favorita.train' is a hypertable. Raises an assertion error if it is not.

    Args:
        db_conn (psycopg2.extensions.connection): An active database connection.

    Raises:
        AssertionError: If 'favorita.train' is not a hypertable.
    """
    cur = db_conn.cursor()
    try:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM _timescaledb_catalog.hypertable h
                WHERE h.schema_name = 'favorita'
                  AND h.table_name  = 'train'
            );
            """
        )
        assert cur.fetchone()[0], "La tabla favorita.train no es hypertable"
    finally:
        cur.close()


def test_hypertable_test(db_conn):
    """
    Verifies that the table 'favorita.test' is registered as a hypertable in TimescaleDB.

    Queries the internal TimescaleDB catalog to check if the table is a hypertable.
    Raises an assertion error if the hypertable is not found.

    Args:
        db_conn (psycopg2.extensions.connection): An active database connection.

    Raises:
        AssertionError: If 'favorita.test' is not a hypertable.
    """
    cur = db_conn.cursor()
    try:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM _timescaledb_catalog.hypertable h
                WHERE h.schema_name = 'favorita'
                  AND h.table_name  = 'test'
            );
            """
        )
        assert cur.fetchone()[0], "La tabla favorita.test no es hypertable"
    finally:
        cur.close()


@pytest.mark.parametrize(
    "idx",
    [
        "idx_train_store_date",
        "idx_train_item",
        "idx_test_store_date",
        "idx_test_item",
        "idx_txn_store_date",
        "idx_items_family",
        "idx_holidays_locale",
    ],
)
def test_indices_exist(db_conn, idx):
    """
    Verifies that a specific index exists in the connected PostgreSQL database.

    Uses the `index_exists` helper function to check whether the given index name exists.
    Raises an assertion error if the index is missing.

    Args:
        db_conn (psycopg2.extensions.connection): An active database connection.
        idx (str): The name of the index to verify.

    Raises:
        AssertionError: If the specified index does not exist.
    """
    cur = db_conn.cursor()
    try:
        assert index_exists(cur, idx), f"Falta índice {idx}"
    finally:
        cur.close()
