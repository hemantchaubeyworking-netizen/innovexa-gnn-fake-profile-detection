"""
INNOVEXA FastAPI Main Application
"""

import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Ensure proper path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.core.config import settings
from backend.app.core.database import init_db
from backend.app.services.detection_service import service
from backend.app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize SQLite database and preload models
    print("[INNOVEXA] Starting backend services...")
    init_db()
    service.initialize()
    print("[INNOVEXA] Ready to serve inference requests.")
    yield
    print("[INNOVEXA] Shutting down backend.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Research-Oriented Graph Neural Network Fake Instagram Profile Detection API",
    lifespan=lifespan
)

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(router, prefix=settings.API_PREFIX)


@app.get("/")
def root():
    return {
        "project": "INNOVEXA",
        "description": "Fake Profile Detection Using Graph Neural Networks (GNNs)",
        "api_docs": "/docs",
        "health_check": f"{settings.API_PREFIX}/health",
        "dashboard": f"{settings.API_PREFIX}/dashboard"
    }


if __name__ == "__main__":
    uvicorn.run("backend.app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
