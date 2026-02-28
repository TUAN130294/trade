@echo off
REM VN-QUANT Live Monitoring Script (Windows)
REM Shows agent signals, trades, and system status in real-time

setlocal enabledelayedexpansion

:main_loop
cls

echo.
echo =====================================================================
echo      VN-QUANT LIVE MONITORING DASHBOARD
echo      Watching: Agent Signals ^| Trades ^| Status
echo      Updated: %date% %time%
echo =====================================================================
echo.

REM Show system status
echo --- SYSTEM STATUS ---
docker-compose ps --format "table {{.Names}}\t{{.Status}}"
echo.

REM Show trading status
echo --- TRADING STATUS ---
for /f "delims=" %%a in ('curl -s http://localhost:5176/api/status') do (
    echo %%a
)
echo.

REM Show agent signals (last 20)
echo --- AGENT SIGNALS (Last 20) ---
for /f "delims=" %%a in ('docker-compose logs --tail^=200 autonomous ^2^>nul ^| findstr /I "Scout Alex Bull Bear Chief RiskDoctor" ^| findstr /V "^!pip"') do (
    echo %%a
)
echo.

REM Show trades (last 15)
echo --- TRADES EXECUTED (Last 15) ---
for /f "delims=" %%a in ('docker-compose logs --tail^=100 autonomous ^2^>nul ^| findstr /I "ORDER EXECUTED POSITION EXIT"') do (
    echo %%a
)
echo.

REM Show errors (last 10)
echo --- WARNINGS ^& ERRORS (Last 10) ---
for /f "delims=" %%a in ('docker-compose logs --tail^=500 autonomous ^2^>nul ^| findstr /I "error warning failed exception"') do (
    echo [WARNING] %%a
)
echo.

echo ===================================================================
echo Press Ctrl+C to stop - Refreshing in 5 seconds...
echo ===================================================================

timeout /t 5 /nobreak > nul

goto main_loop

:end
endlocal
