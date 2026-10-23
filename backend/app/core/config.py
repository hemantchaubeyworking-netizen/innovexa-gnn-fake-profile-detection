"""
INNOVEXA Configuration Settings
"""

import os
from pydantic import BaseModel
from typing import List


class Settings(BaseModel):
    PROJECT_NAME: str = "INNOVEXA: Fake Profile Detection Using GNN"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    HOST: str = os.getenv("BACKEND_HOST", "127.0.0.1")
    PORT: int = int(os.getenv("BACKEND_PORT", "8000"))
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "*"
    ]
    DATABASE_PATH: str = os.path.join(os.path.dirname(__file__), "..", "innovexa.db")
    MODELS_DIR: str = os.path.join(os.path.dirname(__file__), "..", "..", "models")
    RESULTS_DIR: str = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results")
    DATA_PATH: str = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "dataset.csv")


settings = Settings()
