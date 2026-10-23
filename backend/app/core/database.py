"""
INNOVEXA SQLite Database Module
Stores profile prediction history with timestamps and model metadata.
"""

import sqlite3
import json
import datetime
from typing import List, Dict, Any, Optional
from backend.app.core.config import settings


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    # Ensure schema exists
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            model_name TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            fake_probability REAL NOT NULL,
            risk_category TEXT NOT NULL,
            input_summary TEXT NOT NULL,
            input_json TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            model_name TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            fake_probability REAL NOT NULL,
            risk_category TEXT NOT NULL,
            input_summary TEXT NOT NULL,
            input_json TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def insert_prediction(
    model_name: str,
    predicted_class: str,
    fake_probability: float,
    risk_category: str,
    input_data: Dict[str, Any]
) -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    # Quick summary of input
    summary = f"Posts: {input_data.get('#posts', 0)}, Followers: {input_data.get('#followers', 0)}, Follows: {input_data.get('#follows', 0)}"
    
    cursor.execute("""
        INSERT INTO prediction_history (
            timestamp, model_name, predicted_class, fake_probability, risk_category, input_summary, input_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        now_str,
        model_name,
        predicted_class,
        float(fake_probability),
        risk_category,
        summary,
        json.dumps(input_data)
    ))
    
    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    return inserted_id


def get_recent_history(limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, timestamp, model_name, predicted_class, fake_probability, risk_category, input_summary, input_json
        FROM prediction_history
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for r in rows:
        history.append({
            "id": r["id"],
            "timestamp": r["timestamp"],
            "model_name": r["model_name"],
            "predicted_class": r["predicted_class"],
            "fake_probability": round(float(r["fake_probability"]), 4),
            "risk_category": r["risk_category"],
            "input_summary": r["input_summary"],
            "input_data": json.loads(r["input_json"]) if r["input_json"] else {}
        })
    return history


def clear_history():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM prediction_history")
    conn.commit()
    conn.close()
