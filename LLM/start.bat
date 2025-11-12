@echo off
REM Quick Start Script for Windows - LLM Document Intelligence System

echo 🚀 Starting Multi-LLM Document Intelligence System...
echo.

REM Check if .env exists
if not exist .env (
    echo ⚠️  .env file not found!
    echo 📝 Creating .env from .env.example...
    copy .env.example .env
    echo ✅ .env file created. Please edit it with your API keys.
    echo.
    pause
)

REM Check if Docker is available
where docker-compose >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo 🐳 Docker Compose found!
    echo.
    echo Choose startup option:
    echo 1^) Docker ^(Recommended^)
    echo 2^) Local Python
    set /p choice="Enter choice (1 or 2): "
    
    if "!choice!"=="1" (
        echo.
        echo 🐳 Building Docker images...
        docker-compose build
        
        echo.
        echo 🚀 Starting services...
        docker-compose up -d
        
        echo.
        echo ✅ Services started!
        echo.
        echo 📊 Access points:
        echo   - Streamlit Dashboard: http://localhost:8501
        echo   - FastAPI API: http://localhost:8000
        echo   - API Docs: http://localhost:8000/docs
        echo.
        echo 📋 View logs: docker-compose logs -f
        echo 🛑 Stop services: docker-compose down
        
        pause
        exit /b 0
    )
)

REM Local Python startup
echo 🐍 Starting with local Python...
echo.

REM Check Python
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

python --version
echo.

REM Check if virtual environment exists
if not exist venv (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
pip install -r requirements.txt --quiet

echo.
echo ✅ Setup complete!
echo.
echo Choose service to start:
echo 1^) Streamlit Dashboard
echo 2^) FastAPI API
echo 3^) Both
set /p service_choice="Enter choice (1, 2, or 3): "

if "%service_choice%"=="1" (
    echo 🚀 Starting Streamlit Dashboard...
    streamlit run app/dashboard.py
) else if "%service_choice%"=="2" (
    echo 🚀 Starting FastAPI API...
    uvicorn app.api:app --reload --port 8000
) else if "%service_choice%"=="3" (
    echo 🚀 Starting both services...
    start "FastAPI" cmd /c "uvicorn app.api:app --port 8000"
    start "Streamlit" cmd /c "streamlit run app/dashboard.py --server.port 8501"
    
    echo.
    echo ✅ Services started in new windows!
    echo 📊 Access points:
    echo   - Dashboard: http://localhost:8501
    echo   - API: http://localhost:8000
) else (
    echo ❌ Invalid choice
    exit /b 1
)

pause
