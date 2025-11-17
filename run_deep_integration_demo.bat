@echo off
REM Deep Integration Demo Launcher
REM This script demonstrates the TRUE integration between Java and Python/LLM

echo ======================================================================
echo           DEEP INTEGRATION DEMO - Java + Python + LLM
echo ======================================================================
echo.
echo This demo showcases:
echo   1. Java receives transaction
echo   2. Automatically triggers Python/LLM analysis (NO manual steps)
echo   3. Python reads shared PostgreSQL database
echo   4. LLM analyzes with full context
echo   5. Results flow back to Java in real-time
echo   6. Java makes intelligent decision
echo.
echo Prerequisites:
echo   - PostgreSQL running on localhost:5432
echo   - Python FastAPI service running on port 8000
echo.

REM Check if Python service is running
echo Checking Python service...
curl -s http://localhost:8000/health > nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Python service not detected on port 8000
    echo.
    echo To start Python service:
    echo   cd LLM
    echo   python -m uvicorn app.integration_api:app --reload --port 8000
    echo.
    pause
)

echo.
echo Starting Java Deep Integration Demo...
echo.

cd BankFraudTest
mvn compile exec:java -Dexec.mainClass="com.bankfraud.integration.DeepIntegrationDemo" -q

echo.
echo ======================================================================
echo                     DEMO COMPLETE
echo ======================================================================
pause
