import os
import pytest
import psycopg2

DB_DSN = os.getenv("DB_DSN", "postgresql://postgres:postgres@localhost:5432/postgres")


@pytest.fixture(scope="module")
def conn():
    cn = psycopg2.connect(DB_DSN)
    cn.autocommit = True
    yield cn
    cn.close()


def table_exists(cur, schema, table):
    cur.execute(
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
             WHERE table_schema=%s AND table_name=%s
        );
    """,
        (schema, table),
    )
    return cur.fetchone()[0]


def index_exists(cur, index_name):
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


def test_extension_timescaledb(conn):
    cur = conn.cursor()
    cur.execute("SELECT extname FROM pg_extension WHERE extname='timescaledb';")
    assert cur.fetchone() is not None, "La extensión timescaledb no está habilitada"
    cur.close()


def test_schema_favorita(conn):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT EXISTS (
          SELECT 1 FROM information_schema.schemata
           WHERE schema_name='favorita'
        );
    """
    )
    assert cur.fetchone()[0], "El schema 'favorita' no existe"
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
def test_tables_exist(conn, table):
    cur = conn.cursor()
    assert table_exists(cur, "favorita", table), f"Falta tabla favorita.{table}"
    cur.close()


def test_hypertable_train(conn):
    cur = conn.cursor()
    # Timescale guarda hypertables en _timescaledb_catalog.hypertable
    cur.execute(
        """
        SELECT EXISTS (
          SELECT 1 FROM _timescaledb_catalog.hypertable h
           JOIN pg_class c ON c.oid=h.owner_table
           JOIN pg_namespace n ON n.oid=c.relnamespace
           WHERE n.nspname='favorita' AND c.relname='train'
        );
    """
    )
    assert cur.fetchone()[0], "La tabla favorita.train no es hypertable"
    cur.close()


def test_hypertable_test(conn):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT EXISTS (
          SELECT 1 FROM _timescaledb_catalog.hypertable h
           JOIN pg_class c ON c.oid=h.owner_table
           JOIN pg_namespace n ON n.oid=c.relnamespace
           WHERE n.nspname='favorita' AND c.relname='test'
        );
    """
    )
    assert cur.fetchone()[0], "La tabla favorita.test no es hypertable"
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
def test_indices_exist(conn, idx):
    cur = conn.cursor()
    assert index_exists(cur, idx), f"Falta índice {idx}"
    cur.close()
