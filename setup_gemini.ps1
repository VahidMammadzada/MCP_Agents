#!/usr/bin/env powershell
# Setup script for Financial AI System with Gemini API

Write-Host "Setting up Financial AI System with Gemini API..." -ForegroundColor Green

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: Python is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

# Create virtual environment if it doesn't exist
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install requirements
Write-Host "Installing Python packages..." -ForegroundColor Yellow
pip install -r requirements.txt

# Create .env file from template if it doesn't exist
if (-not (Test-Path ".env")) {
    Write-Host "Creating .env file from template..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host "Please edit .env file and add your Gemini API key" -ForegroundColor Red
} else {
    Write-Host ".env file already exists" -ForegroundColor Green
}

# Create necessary directories
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs"
    Write-Host "Created logs directory" -ForegroundColor Green
}

if (-not (Test-Path "data")) {
    New-Item -ItemType Directory -Path "data"
    Write-Host "Created data directory" -ForegroundColor Green
}

Write-Host "`nSetup completed!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "1. Edit .env file and add your Gemini API key" -ForegroundColor White
Write-Host "2. Get your Gemini API key from: https://makersuite.google.com/app/apikey" -ForegroundColor White
Write-Host "3. Run: python specialized_agents.py to test the system" -ForegroundColor White

Write-Host "`nTo run the system:" -ForegroundColor Yellow
Write-Host "python launch_financial_ai.py" -ForegroundColor White
