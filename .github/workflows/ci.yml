name: CI Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  base-setup:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Python 3.12.7
        uses: actions/setup-python@v4
        with:
          python-version: '3.12.7'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          if [ -f environment.txt ]; then
            pip install -r environment.txt
          else
            echo "❌ No se encontró environment.txt"
            exit 1
          fi

      - name: Run install.ps1
        shell: pwsh
        run: |
          if (Test-Path -Path "./scripts/install.ps1") {
            Write-Host "🔧 Ejecutando scripts/install.ps1 para crear Docker Compose..."
            ./scripts/install.ps1
          } else {
            Write-Error "❌ No se encontró scripts/install.ps1"
            exit 1
          }

      - name: Start and initialize containers
        shell: pwsh
        run: |
          if (Test-Path -Path "./scripts/start.ps1") {
            Write-Host "🚀 Iniciando contenedores y creando arquitectura de base de datos..."
            ./scripts/start.ps1
          } else {
            Write-Error "❌ No se encontró scripts/start.ps1"
            exit 1
          }