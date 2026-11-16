import sqlite3
from pathlib import Path
import numpy as np


class FaceDatabase:
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.create_table()
        self.embeddings = None
        self.persons = []
        self.load_embeddings()

    def create_table(self):
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                hometown TEXT,
                phone TEXT,
                role TEXT,
                notes TEXT,
                image_path TEXT,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.commit()

    def insert_person(self, key, info, embedding):
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        emb_bytes = emb.tobytes()

        self.conn.execute(
            """
            INSERT INTO persons (key, name, age, gender, hometown, phone, role, notes, image_path, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                name = excluded.name,
                age = excluded.age,
                gender = excluded.gender,
                hometown = excluded.hometown,
                phone = excluded.phone,
                role = excluded.role,
                notes = excluded.notes,
                image_path = excluded.image_path,
                embedding = excluded.embedding
            """,
            (
                key,
                info.get("name", key),
                info.get("age", 0),
                info.get("gender", ""),
                info.get("hometown", ""),
                info.get("phone", ""),
                info.get("role", ""),
                info.get("notes", ""),
                info.get("image", ""),
                emb_bytes,
            ),
        )
        self.conn.commit()

    def load_embeddings(self):
        cursor = self.conn.execute(
            "SELECT key, name, age, gender, hometown, phone, role, notes, image_path, embedding FROM persons"
        )
        rows = cursor.fetchall()
        self.persons = []
        emb_list = []

        for row in rows:
            p = {
                "key": row[0],
                "name": row[1],
                "age": row[2],
                "gender": row[3],
                "hometown": row[4],
                "phone": row[5],
                "role": row[6],
                "notes": row[7],
                "image": row[8],
            }
            emb = np.frombuffer(row[9], dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            self.persons.append(p)
            emb_list.append(emb)

        if emb_list:
            self.embeddings = np.vstack(emb_list)
        else:
            self.embeddings = None

    def search(self, query_embedding, threshold=0.48):
        if self.embeddings is None or len(self.persons) == 0:
            return None, 0.0

        q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return None, 0.0
        q = q / q_norm

        sims = np.dot(self.embeddings, q)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= threshold:
            return self.persons[best_idx], best_sim
        return None, best_sim

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
