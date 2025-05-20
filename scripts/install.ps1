#!/usr/bin/env pwsh
<#
    install.ps1: Descarga imágenes y crea (sin arrancar) los contenedores definidos en infra/docker-compose.yml
    Compatible con PowerShell Core (Windows & Linux) y Windows PowerShell.
#>

# Función para abortar con mensaje de error
function Abort([string]$msg) {
    Write-Error $msg
    exit 1
}

# Detectar plataforma
$osEnv = $env:OS
if ($PSVersionTable.PSEdition -eq 'Core') {
    if ($IsWindows) { $platform = 'Windows' }
    elseif ($IsLinux) { $platform = 'Linux' }
    else { Abort "SO no soportado (Core): $osEnv" }
} else {
    if ($osEnv -eq 'Windows_NT') { $platform = 'Windows' }
    else { Abort "SO no soportado (Desktop): $osEnv" }
}
Write-Host "→ Plataforma detectada: $platform"

# Verificar Docker
try {
    docker version > $null 2>&1
} catch {
    Abort "Docker no encontrado o no está en ejecución."
}

# En Linux, asegurar docker compose plugin
if ($platform -eq 'Linux') {
    # Intentar instalación de docker compose plugin si no existe
    if (-not (docker compose version -q 2>$null) -and -not (Get-Command docker-compose -ErrorAction SilentlyContinue)) {
        Write-Host "→ Instalando docker compose plugin en Linux..."
        try {
            sudo apt-get update
            sudo apt-get install -y docker-compose
        } catch {
            Abort "Error al instalar docker compose en Linux."
        }
    }
}

# Ubicar docker-compose.yml
$scriptDir = Split-Path -Path $MyInvocation.MyCommand.Path -Parent
$projectRoot = Resolve-Path -Path (Join-Path $scriptDir '..')
$composeDir = Join-Path $projectRoot 'infra'
if (-not (Test-Path (Join-Path $composeDir 'docker-compose.yml'))) {
    Abort "No se encontró docker-compose.yml en: $composeDir"
}
Set-Location $composeDir
Write-Host "→ Directorio de Compose: $PWD"

# Seleccionar comando de Compose
if (Get-Command docker-compose -ErrorAction SilentlyContinue) { $composeCmd = 'docker-compose' }
elseif (docker compose version -q 2>$null) { $composeCmd = 'docker compose' }
else { Abort "Ni 'docker compose' ni 'docker-compose' disponibles." }
Write-Host "→ Usando: $composeCmd"

# Paso 1: Descargar imágenes
Write-Host "→ Descargando imágenes..."
& $composeCmd pull
if ($LASTEXITCODE -ne 0) { Abort "Fallo al descargar imágenes (exit code: $LASTEXITCODE)." }

# Paso 2: Crear contenedores sin ejecutarlos
Write-Host "→ Creando contenedores (sin arrancar)..."
& $composeCmd create
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Contenedores preparados correctamente pero detenidos."
} else {
    Abort "❌ Error al crear contenedores (exit code: $LASTEXITCODE)."
}
