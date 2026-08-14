import json
import sqlite3
from pathlib import Path

from app.models.investigation import (
    Investigation,
    InvestigationCreate,
    InvestigationUpdate,
    utc_now,
)


class InvestigationRepository:
    def __init__(self, database_path: str = "data/inquiry.db") -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS investigations (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    original_question TEXT NOT NULL,
                    objective TEXT,
                    brief_json TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    confidence REAL,
                    parent_investigation_id TEXT,
                    version INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def create(self, payload: InvestigationCreate) -> Investigation:
        investigation = Investigation(**payload.model_dump())

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO investigations (
                    id,
                    workspace_id,
                    original_question,
                    objective,
                    brief_json,
                    report_json,
                    confidence,
                    parent_investigation_id,
                    version,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    investigation.id,
                    investigation.workspace_id,
                    investigation.original_question,
                    investigation.objective,
                    json.dumps(investigation.brief),
                    json.dumps(investigation.report),
                    investigation.confidence,
                    investigation.parent_investigation_id,
                    investigation.version,
                    investigation.created_at.isoformat(),
                    investigation.updated_at.isoformat(),
                ),
            )

        return investigation

    def list_all(self) -> list[Investigation]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM investigations
                ORDER BY created_at DESC
                """
            ).fetchall()

        return [self._row_to_model(row) for row in rows]

    def get(self, investigation_id: str) -> Investigation | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM investigations
                WHERE id = ?
                """,
                (investigation_id,),
            ).fetchone()

        return self._row_to_model(row) if row else None

    def update(
        self,
        investigation_id: str,
        payload: InvestigationUpdate,
    ) -> Investigation | None:
        current = self.get(investigation_id)

        if current is None:
            return None

        changes = payload.model_dump(exclude_unset=True)

        if payload.version is not None and payload.version != current.version:
            raise ValueError("Version conflict")

        changes.pop("version", None)

        updated = current.model_copy(
            update={
                **changes,
                "version": current.version + 1,
                "updated_at": utc_now(),
            }
        )

        with self._connect() as connection:
            connection.execute(
                """
                UPDATE investigations
                SET workspace_id = ?,
                    original_question = ?,
                    objective = ?,
                    brief_json = ?,
                    report_json = ?,
                    confidence = ?,
                    parent_investigation_id = ?,
                    version = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    updated.workspace_id,
                    updated.original_question,
                    updated.objective,
                    json.dumps(updated.brief),
                    json.dumps(updated.report),
                    updated.confidence,
                    updated.parent_investigation_id,
                    updated.version,
                    updated.updated_at.isoformat(),
                    updated.id,
                ),
            )

        return updated

    def delete(self, investigation_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM investigations
                WHERE id = ?
                """,
                (investigation_id,),
            )

        return cursor.rowcount > 0

    @staticmethod
    def _row_to_model(row: sqlite3.Row) -> Investigation:
        return Investigation(
            id=row["id"],
            workspace_id=row["workspace_id"],
            original_question=row["original_question"],
            objective=row["objective"],
            brief=json.loads(row["brief_json"]),
            report=json.loads(row["report_json"]),
            confidence=row["confidence"],
            parent_investigation_id=row["parent_investigation_id"],
            version=row["version"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )