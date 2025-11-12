#!/bin/bash

# Quick Start Script for LLM Document Intelligence System

echo "🚀 Starting Multi-LLM Document Intelligence System..."
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found!"
    echo "📝 Creating .env from .env.example..."
    cp .env.example .env
    echo "✅ .env file created. Please edit it with your API keys."
    echo ""
    echo "Edit .env file now? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        ${EDITOR:-nano} .env
    fi
fi

# Check if Docker is available
if command -v docker-compose &> /dev/null; then
    echo "🐳 Docker Compose found!"
    echo ""
    echo "Choose startup option:"
    echo "1) Docker (Recommended)"
    echo "2) Local Python"
    read -p "Enter choice (1 or 2): " choice
    
    if [ "$choice" = "1" ]; then
        echo ""
        echo "🐳 Building Docker images..."
        docker-compose build
        
        echo ""
        echo "🚀 Starting services..."
        docker-compose up -d
        
        echo ""
        echo "✅ Services started!"
        echo ""
        echo "📊 Access points:"
        echo "  - Streamlit Dashboard: http://localhost:8501"
        echo "  - FastAPI API: http://localhost:8000"
        echo "  - API Docs: http://localhost:8000/docs"
        echo ""
        echo "📋 View logs: docker-compose logs -f"
        echo "🛑 Stop services: docker-compose down"
        
        # Offer to show logs
        echo ""
        echo "Show logs now? (y/n)"
        read -r show_logs
        if [[ "$show_logs" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            docker-compose logs -f
        fi
        
        exit 0
    fi
fi

# Local Python startup
echo "🐍 Starting with local Python..."
echo ""

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "❌ Python not found. Please install Python 3.8+"
    exit 1
fi

echo "Python: $($PYTHON --version)"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    $PYTHON -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt --quiet

echo ""
echo "✅ Setup complete!"
echo ""
echo "Choose service to start:"
echo "1) Streamlit Dashboard"
echo "2) FastAPI API"
echo "3) Both (in background)"
read -p "Enter choice (1, 2, or 3): " service_choice

case $service_choice in
    1)
        echo "🚀 Starting Streamlit Dashboard..."
        streamlit run app/dashboard.py
        ;;
    2)
        echo "🚀 Starting FastAPI API..."
        uvicorn app.api:app --reload --port 8000
        ;;
    3)
        echo "🚀 Starting both services..."
        uvicorn app.api:app --port 8000 > logs/api.log 2>&1 &
        API_PID=$!
        streamlit run app/dashboard.py --server.port 8501 > logs/dashboard.log 2>&1 &
        DASH_PID=$!
        
        echo ""
        echo "✅ Services started!"
        echo "  - API PID: $API_PID"
        echo "  - Dashboard PID: $DASH_PID"
        echo ""
        echo "📊 Access points:"
        echo "  - Dashboard: http://localhost:8501"
        echo "  - API: http://localhost:8000"
        echo ""
        echo "🛑 To stop: kill $API_PID $DASH_PID"
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac
