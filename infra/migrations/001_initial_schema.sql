BEGIN;

-- Habilitar TimescaleDB
\echo '-> Habilitando extensión TimescaleDB'
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Crear schema dedicada
\echo '-> Creando schema favorita'
CREATE SCHEMA IF NOT EXISTS favorita;

-- Ajustar search_path para crear objetos en el schema correcta
SET search_path = favorita, public;

----------------------------
-- TABLAS DIMENSIONALES
----------------------------

\echo '-> Creando tabla stores'
CREATE TABLE IF NOT EXISTS favorita.stores (
  store_nbr  INT     PRIMARY KEY,
  city       VARCHAR,
  state      VARCHAR,
  type       VARCHAR,
  cluster    INT
);

\echo '-> Creando tabla items'
CREATE TABLE IF NOT EXISTS favorita.items (
  item_nbr   INT     PRIMARY KEY,
  family     VARCHAR,
  class      INT,
  perishable BOOLEAN
);

\echo '-> Creando tabla transactions'
CREATE TABLE IF NOT EXISTS favorita.transactions (
  date         DATE    NOT NULL,
  store_nbr    INT     NOT NULL,
  transactions INT,
  PRIMARY KEY (date, store_nbr),
  FOREIGN KEY (store_nbr) REFERENCES favorita.stores(store_nbr)
);

\echo '-> Creando tabla oil'
CREATE TABLE IF NOT EXISTS favorita.oil (
  date      DATE    PRIMARY KEY,
  dcoilwtico FLOAT
);

\echo '-> Creando tabla holidays_events'
CREATE TABLE IF NOT EXISTS favorita.holidays_events (
  date        DATE    NOT NULL,
  type        VARCHAR NOT NULL,
  locale      VARCHAR NOT NULL,
  description VARCHAR,
  transferred BOOLEAN,
  PRIMARY KEY (date, type, locale)
);

\echo '-> Creando tabla sample_submission'
CREATE TABLE IF NOT EXISTS favorita.sample_submission (
  id         INT     PRIMARY KEY,
  unit_sales FLOAT
);

----------------------------
-- TABLAS DE VENTAS (HYPERTABLES)
----------------------------

\echo '-> Creando tabla train'
CREATE TABLE IF NOT EXISTS favorita.train (
  date         DATE        NOT NULL,
  store_nbr    INT         NOT NULL,
  item_nbr     INT         NOT NULL,
  unit_sales   FLOAT,
  onpromotion  BOOLEAN,
  id           BIGSERIAL,
  PRIMARY KEY (date, store_nbr, item_nbr)
);

\echo '-> Convirtiendo train en hypertable (tiempo + espacio, chunks 30 días, 50 particiones)'
SELECT create_hypertable(
  'favorita.train',
  'date',
  'store_nbr',
  chunk_time_interval => INTERVAL '30 days',
  number_partitions   => 50,
  if_not_exists       => TRUE
);

\echo '-> Creando tabla test'
CREATE TABLE IF NOT EXISTS favorita.test (
  date         DATE        NOT NULL,
  store_nbr    INT         NOT NULL,
  item_nbr     INT         NOT NULL,
  onpromotion  BOOLEAN,
  id           BIGSERIAL,
  PRIMARY KEY (date, store_nbr, item_nbr)
);

\echo '-> Convirtiendo test en hypertable (tiempo + espacio, chunks 30 días, 50 particiones)'
SELECT create_hypertable(
  'favorita.test',
  'date',
  'store_nbr',
  chunk_time_interval => INTERVAL '30 days',
  number_partitions   => 50,
  if_not_exists       => TRUE
);

----------------------------
-- ÍNDICES ADICIONALES
----------------------------

\echo '-> Creando índices adicionales para acelerar consultas'

-- Índices sobre train
CREATE INDEX IF NOT EXISTS idx_train_store_date 
  ON favorita.train (store_nbr, date DESC);
CREATE INDEX IF NOT EXISTS idx_train_item 
  ON favorita.train (item_nbr);

-- Índices sobre test
CREATE INDEX IF NOT EXISTS idx_test_store_date 
  ON favorita.test (store_nbr, date DESC);
CREATE INDEX IF NOT EXISTS idx_test_item 
  ON favorita.test (item_nbr);

-- Índice sobre transactions
CREATE INDEX IF NOT EXISTS idx_txn_store_date
  ON favorita.transactions (store_nbr, date);

-- Índice sobre items.family
CREATE INDEX IF NOT EXISTS idx_items_family
  ON favorita.items (family);

-- Índice sobre holidays_events.locale
CREATE INDEX IF NOT EXISTS idx_holidays_locale
  ON favorita.holidays_events (locale);

\echo '-> Script completado exitosamente'

ROLLBACK; -- Cambiar a COMMIT; para aplicar cambios
