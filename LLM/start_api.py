"""Simple starter script for integration API"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

if __name__ == "__main__":
    import uvicorn
    from app.integration_api import app
    
    print("Starting Financial Intelligence Integration API...")
    print("Endpoints available at http://localhost:8000")
    print("- POST /api/analyze-transaction")
    print("- POST /api/search-documents")  
    print("- POST /api/generate-report")
    print("- GET /health")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
