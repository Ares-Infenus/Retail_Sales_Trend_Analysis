<#
.SYNOPSIS
  Carga datos ejecutando src/ingest.py sin rutas absolutas, para varios CSV→tabla.
#>

Param(
    # Hashtable que mapea cada CSV (relativo al repo) a su nombre de tabla
    [hashtable]$CsvTableMap = @{
        'data\raw\items.csv'           = 'items'
        'data\raw\holidays_events.csv' = 'holidays_events'
        'data\raw\stores.csv'          = 'stores'
        'data\raw\test.csv'          = 'test'
        'data\raw\transactions.csv'         = 'transactions'
        'data\processed\clear_train.csv'          = 'train'
        'data\processed\clear_oil_raw_Dukascopy.csv'          = 'oil'
    }
)

# 1) Detecta la raíz del proyecto
$RepoRoot = Split-Path -Parent $PSScriptRoot
Write-Host "📁 Repo root: $RepoRoot"

# 2) Construye ruta al script Python
$ScriptPath = Join-Path $RepoRoot 'src\ingest.py'
if (-not (Test-Path $ScriptPath)) {
    Write-Error "❌ No encuentro el script: $ScriptPath"
    exit 1
}

# 3) Recorre cada par CSV→tabla
foreach ($relativePath in $CsvTableMap.Keys) {
    $tableName = $CsvTableMap[$relativePath]
    $CsvPath   = Join-Path $RepoRoot $relativePath

    # 3a) Comprueba existencia
    if (-not (Test-Path $CsvPath)) {
        Write-Error "❌ No encuentro el CSV: $CsvPath"
        continue
    }

    # 4) Ejecuta Python para este CSV y tabla
    Write-Host "▶️ Ejecutando: python $ScriptPath $CsvPath $tableName"
    & python $ScriptPath $CsvPath $tableName
}
