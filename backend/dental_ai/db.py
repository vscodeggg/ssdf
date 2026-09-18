import sqlite3
import threading
from pathlib import Path
from typing import Iterable, Sequence


class DentalDB:
    """Local SQLite store for patient sessions, transcripts, and chart findings."""

    def __init__(self, db_path: str | None = None):
        base_dir = Path(__file__).resolve().parent.parent
        self.db_path = Path(db_path) if db_path else base_dir / "dental_data.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # The web server (ThreadingHTTPServer) touches the DB from worker
        # threads, so the connection must be shared across threads and every
        # access serialized through self._lock.
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_name TEXT NOT NULL,
                transcript TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'voice',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS findings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                patient_name TEXT NOT NULL,
                tooth_number TEXT NOT NULL,
                surface TEXT NOT NULL,
                finding_type TEXT NOT NULL,
                value TEXT,
                unit TEXT,
                notes TEXT,
                needs_review INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_findings_patient ON findings(patient_name, created_at)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_findings_session ON findings(session_id)"
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_name TEXT NOT NULL,
                recipient TEXT NOT NULL,
                subject TEXT NOT NULL,
                report_body TEXT NOT NULL,
                delivery_status TEXT NOT NULL DEFAULT 'draft',
                error_message TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reports_patient ON reports(patient_name, created_at)"
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dictations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_name TEXT NOT NULL,
                audio_file TEXT NOT NULL,
                transcript TEXT,
                source TEXT NOT NULL DEFAULT 'python-mic',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.commit()

    def save_transcript_session(self, patient_name: str, transcript: str, source: str = "voice") -> int:
        with self._lock:
            cursor = self.conn.execute(
                "INSERT INTO sessions (patient_name, transcript, source) VALUES (?, ?, ?)",
                (patient_name.strip() or "Active Patient", transcript.strip(), source),
            )
            self.conn.commit()
            return int(cursor.lastrowid)

    def save_finding(
        self,
        session_id: int,
        patient_name: str,
        tooth_number: str,
        surface: str,
        finding_type: str,
        value: str | None = None,
        unit: str | None = None,
        notes: str | None = None,
        needs_review: bool = False,
        status: str = "active",
    ) -> int:
        with self._lock:
            cursor = self.conn.execute(
                """
                INSERT INTO findings (
                    session_id, patient_name, tooth_number, surface, finding_type, value, unit,
                    notes, needs_review, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    patient_name.strip() or "Active Patient",
                    str(tooth_number),
                    str(surface or "unspecified"),
                    str(finding_type or "clinical finding"),
                    value,
                    unit,
                    notes,
                    1 if needs_review else 0,
                    status,
                ),
            )
            self.conn.commit()
            return int(cursor.lastrowid)

    def save_parsed_findings(
        self,
        patient_name: str,
        transcript: str,
        findings: Sequence[object],
        source: str = "voice",
    ) -> int:
        session_id = self.save_transcript_session(patient_name, transcript, source)
        if not findings:
            return session_id

        for finding in findings:
            if hasattr(finding, "as_dict"):
                payload = finding.as_dict()
            else:
                payload = dict(finding)

            self.save_finding(
                session_id=session_id,
                patient_name=patient_name,
                tooth_number=payload.get("tooth_number") or payload.get("toothId") or "",
                surface=payload.get("surface") or "unspecified",
                finding_type=payload.get("finding_type") or payload.get("condition") or "clinical finding",
                value=payload.get("value"),
                unit=payload.get("unit"),
                notes=payload.get("notes"),
                needs_review=bool(payload.get("needs_review", False)),
                status=payload.get("status", "active"),
            )
        return session_id

    def get_recent_sessions(self, limit: int = 20) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT id, patient_name, transcript, source, created_at
                FROM sessions
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_patient_findings(self, patient_name: str, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT id, patient_name, tooth_number, surface, finding_type, value, unit, notes,
                       needs_review, status, created_at
                FROM findings
                WHERE patient_name = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (patient_name, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_report(
        self,
        patient_name: str,
        recipient: str,
        subject: str,
        report_body: str,
        delivery_status: str = "draft",
        error_message: str | None = None,
    ) -> int:
        """Persist a generated patient report (system database record)."""
        with self._lock:
            cursor = self.conn.execute(
                """
                INSERT INTO reports (patient_name, recipient, subject, report_body,
                                     delivery_status, error_message)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    patient_name.strip() or "Active Patient",
                    recipient.strip(),
                    subject,
                    report_body,
                    delivery_status,
                    error_message,
                ),
            )
            self.conn.commit()
            return int(cursor.lastrowid)

    def get_reports(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT id, patient_name, recipient, subject, report_body,
                       delivery_status, error_message, created_at
                FROM reports
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_dictation(self, patient_name: str, audio_file: str, transcript: str, source: str = "python-mic") -> int:
        """Archive a recorded dictation chunk (audio kept on disk for replay/audit)."""
        with self._lock:
            cursor = self.conn.execute(
                "INSERT INTO dictations (patient_name, audio_file, transcript, source) VALUES (?, ?, ?, ?)",
                (patient_name.strip() or "Active Patient", audio_file, transcript, source),
            )
            self.conn.commit()
            return int(cursor.lastrowid)

    def get_dictations(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT id, patient_name, audio_file, transcript, source, created_at
                FROM dictations
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        self.conn.close()
