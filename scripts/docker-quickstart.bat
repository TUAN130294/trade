@echo off
REM VN-Quant Docker Quick Start Script (Windows)
REM Run this script to start the complete system in Docker

setlocal enabledelayedexpansion
cd /d "%~dp0\.."

echo ================================
echo.Docker Quick Start for VN-Quant
echo ================================
echo.

REM Check prerequisites
echo Checking prerequisites...

docker --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not installed
    echo Install from: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker Compose is not installed
    pause
    exit /b 1
)

echo [OK] Docker and Docker Compose found
echo.

REM Check if Docker daemon is running
docker ps >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker daemon is not running
    echo Please start Docker Desktop and try again
    pause
    exit /b 1
)

echo [OK] Docker daemon is running
echo.

REM Configuration
echo Configuring environment...

if not exist .env (
    echo Creating .env from .env.example...
    copy .env.example .env >nul
    echo [OK] .env created
    echo.
    echo WARNING: Please edit .env with your API keys:
    echo   - GEMINI_API_KEY
    echo   - TELEGRAM_BOT_TOKEN
    echo.
    set /p continue="Continue with defaults? (y/n): "
    if /i not "!continue!"=="y" exit /b 0
) else (
    echo [OK] .env already configured
)

echo.

REM Build images
echo Building Docker images (this may take 5-10 minutes)...
docker-compose build
if errorlevel 1 (
    echo ERROR: Docker build failed
    pause
    exit /b 1
)

echo [OK] Docker images built successfully
echo.

REM Start services
echo Starting services...
echo   - PostgreSQL database
echo   - Redis cache
echo   - Autonomous trading system
echo   - REST API server
echo   - Model training service
echo.

docker-compose up -d
if errorlevel 1 (
    echo ERROR: Failed to start services
    pause
    exit /b 1
)

echo.

REM Wait for services
echo Waiting for services to be ready...
timeout /t 5 /nobreak

REM Show status
echo.
echo Service Status:
echo ===============
docker-compose ps

echo.
echo ================================
echo [OK] VN-Quant is ready!
echo ================================
echo.
echo Dashboard:
echo   http://localhost:8001/autonomous
echo.
echo API Documentation:
echo   http://localhost:8003/docs
echo.
echo View Logs:
echo   docker-compose logs -f autonomous
echo   docker-compose logs -f api
echo   docker-compose logs -f model-trainer
echo.
echo Stop System:
echo   docker-compose down
echo.
echo For more details, see docs\docker-deployment.md
echo.
pause
