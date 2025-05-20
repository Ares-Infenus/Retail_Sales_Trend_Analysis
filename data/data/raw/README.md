# Favorita Grocery Sales Forecasting – Descripción de los archivos

El dataset de la competición **Corporación Favorita Grocery Sales Forecasting** se compone de los siguientes archivos CSV:

---

## `train.csv`
- Datos de entrenamiento: ventas diarias (`unit_sales`) por combinación **fecha – tienda – artículo**.
- Columnas principales: 
  - `id`
  - `date`
  - `store_nbr`
  - `item_nbr`
  - `unit_sales`
  - `onpromotion`
- **Notas**:
  - Valores negativos en `unit_sales` representan devoluciones.

---

## `test.csv`
- Datos para hacer predicciones (mismo formato que `train.csv`, pero **sin `unit_sales`**).
- Columnas principales:
  - `id`
  - `date`
  - `store_nbr`
  - `item_nbr`
  - `onpromotion`
- Corresponde a los 15 días posteriores al final del período de entrenamiento.

---

## `sample_submission.csv`
- Ejemplo del formato de entrega.
- Columnas:
  - `id`
  - `unit_sales` (vacío para que rellenes con tus predicciones).

---

## `stores.csv`
- Metadatos de las tiendas.
- Columnas:
  - `store_nbr`
  - `city`
  - `state`
  - `type`
  - `cluster`

---

## `items.csv`
- Metadatos de los artículos.
- Columnas:
  - `item_nbr`
  - `family`
  - `class`
  - `perishable`
- **Notas**:
  - `perishable` indica si es un producto perecedero (afecta el peso en la métrica de evaluación).

---

## `transactions.csv`
- Número de transacciones diarias por tienda.
- Columnas:
  - `date`
  - `store_nbr`
  - `transactions`
- Útil para capturar la actividad de tráfico de clientes.

---

## `oil.csv`
- Precio diario del petróleo.
- Columnas:
  - `date`
  - `dcoilwtico`
- Importante porque la economía de Ecuador está ligada al precio del crudo.

---

## `holidays_events.csv`
- Información de días festivos y eventos en Ecuador.
- Columnas:
  - `date`
  - `type` (Event, Holiday, Bridge, Transfer)
  - `locale` (National, Regional, Local)
  - `description`
  - `transferred`
- Permite analizar el impacto de días especiales en las ventas.

---

## Resumen
Con estos ocho archivos podrás llevar a cabo todas las fases del proyecto: 
- EDA
- Limpieza
- Modelado de series temporales
- Generación del dashboard interactivo
