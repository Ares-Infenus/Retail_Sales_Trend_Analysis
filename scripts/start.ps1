#!/usr/bin/env pwsh
<#
    start-container.ps1: Inicia los contenedores definidos en infra/docker-compose.yml
    y avisa si no están creados.
    Compatible con PowerShell Core (Windows & Linux) y Windows PowerShell.
#>

# Función para abortar con mensaje de error
function Abort([string]$msg) {
    Write-Error $msg
    exit 1
}

# Verificar Docker
try {
    docker version > $null 2>&1
} catch {
    Abort "Docker no encontrado o no está en ejecución."
}

# Ubicar docker-compose.yml y verificar que los contenedores existan
$scriptDir = Split-Path -Path $MyInvocation.MyCommand.Path -Parent
$projectRoot = Resolve-Path -Path (Join-Path $scriptDir '..')
$composeFile = Join-Path $projectRoot 'infra\docker-compose.yml'
if (-not (Test-Path $composeFile)) {
    Abort "No se encontró docker-compose.yml en infra. Ejecuta antes la preparación."
}

# Seleccionar comando de Compose
if (docker compose version -q 2>$null) {
    $composeCmd = 'docker compose'
} elseif (Get-Command docker-compose -ErrorAction SilentlyContinue) {
    $composeCmd = 'docker-compose'
} else {
    Abort "Ni 'docker compose' ni 'docker-compose' disponibles.";
}

# Intentar iniciar contenedores
Write-Host "→ Iniciando contenedores..."
& $composeCmd -f $composeFile start
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Contenedores iniciados correctamente."
} else {
    Abort "❌ Error al iniciar contenedores (exit code: $LASTEXITCODE). Verifica que estén creados."  
}