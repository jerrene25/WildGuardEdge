import sqlite3
import json
import logging
import io
import csv
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger("WildGuardDatabase")


class WildGuardDatabase:
    """
    Hardened, authoritative SQLite event database for WildGuard Edge.
    Features:
    - WAL mode for concurrent multi-threaded read/write safety.
    - Path traversal elimination on snapshot deletion.
    - Parameterized queries with input bounds validation and sanitization.
    - Optimized indexing on timestamps, threat scores, and modes.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else Path(config.DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Opens connection with WAL journal mode, busy timeout, and row factory."""
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
        except sqlite3.OperationalError as e:
            logger.warning(f"[Database] Could not set PRAGMA settings: {e}")
        return conn

    def init_db(self):
        """Initializes the database schema with performance indexes."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    threat_score REAL NOT NULL,
                    audio_class TEXT,
                    audio_confidence REAL,
                    audio_rms REAL,
                    person_detected INTEGER NOT NULL,
                    person_confidence REAL,
                    clip_score REAL,
                    caption TEXT,
                    snapshot_path TEXT,
                    reason TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_threat_score ON events(threat_score);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_mode ON events(mode);")
            conn.commit()

    def insert_event(self, event_data: Dict[str, Any]) -> int:
        """
        Inserts a confirmed threat event into the database with strict input validation.
        Returns the inserted event id.
        """
        try:
            raw_ts = str(event_data.get("timestamp", ""))
            try:
                datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                timestamp = raw_ts
            except Exception:
                timestamp = datetime.now().isoformat()

            raw_mode = str(event_data.get("mode", "DAY")).upper().strip()
            mode = raw_mode if raw_mode in ("DAY", "NIGHT") else "DAY"

            # Clamp confidence and threat scores into [0.0, 1.0]
            def _clamp_float(val: Any, default: float = 0.0) -> float:
                try:
                    f = float(val) if val is not None else default
                    if f != f:  # NaN check
                        return default
                    return min(1.0, max(0.0, f))
                except (ValueError, TypeError):
                    return default

            threat_score = _clamp_float(event_data.get("threat_score"), 0.0)
            audio_class = str(event_data.get("audio_class", "unknown"))[:100]
            audio_confidence = _clamp_float(event_data.get("audio_confidence"), 0.0)
            audio_rms = max(0.0, float(event_data.get("audio_rms", 0.0) or 0.0))
            person_detected = 1 if event_data.get("person_detected") else 0
            person_confidence = _clamp_float(event_data.get("person_confidence"), 0.0)
            clip_score = _clamp_float(event_data.get("clip_score"), 0.0)
            caption = str(event_data.get("caption", ""))[:500] if event_data.get("caption") else None
            snapshot_path = str(event_data.get("snapshot_path", ""))[:500] if event_data.get("snapshot_path") else ""
            reason = str(event_data.get("reason", ""))[:500]

            query = """
                INSERT INTO events (
                    timestamp, mode, threat_score, audio_class, audio_confidence,
                    audio_rms, person_detected, person_confidence, clip_score,
                    caption, snapshot_path, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (
                    timestamp, mode, threat_score, audio_class, audio_confidence,
                    audio_rms, person_detected, person_confidence, clip_score,
                    caption, snapshot_path, reason
                ))
                conn.commit()
                return cursor.lastrowid or 0

        except Exception as e:
            logger.error(f"[Database] Failed to insert event: {e}", exc_info=True)
            return 0

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves recent alerts ordered by ID descending with bounded limit."""
        safe_limit = max(1, min(int(limit), 1000))
        query = "SELECT * FROM events ORDER BY id DESC LIMIT ?"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (safe_limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_total_count(self) -> int:
        """Returns total number of logged events."""
        query = "SELECT COUNT(*) FROM events"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            row = cursor.fetchone()
            return row[0] if row else 0

    def get_event_count(self) -> int:
        return self.get_total_count()

    def get_event_by_id(self, event_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves a single event by ID."""
        try:
            safe_id = int(event_id)
        except (ValueError, TypeError):
            return None
        query = "SELECT * FROM events WHERE id = ?"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (safe_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def delete_event(self, event_id: int, delete_snapshot: bool = True) -> bool:
        """
        Deletes a single event by ID.
        Fortified with path confinement: prevents arbitrary file deletion by ensuring
        snapshot_path strictly resides within config.SNAPSHOT_DIR.
        """
        try:
            safe_id = int(event_id)
        except (ValueError, TypeError):
            return False

        event = self.get_event_by_id(safe_id)
        if not event:
            return False

        if delete_snapshot and event.get("snapshot_path"):
            try:
                snap_path = Path(event["snapshot_path"]).resolve()
                snapshot_root = Path(config.SNAPSHOT_DIR).resolve()

                # Verify snapshot is safely inside the configured snapshots directory
                if snap_path.is_relative_to(snapshot_root):
                    if snap_path.exists() and snap_path.is_file():
                        snap_path.unlink()
                else:
                    logger.warning(
                        f"[Security Alert] Prevented arbitrary file deletion. "
                        f"Path '{snap_path}' is outside root '{snapshot_root}'"
                    )
            except (ValueError, AttributeError) as e:
                # Python <3.9 fallback or invalid relative path
                try:
                    if str(snap_path).startswith(str(snapshot_root)):
                        if snap_path.exists() and snap_path.is_file():
                            snap_path.unlink()
                except Exception as inner_e:
                    logger.warning(f"[Database] Could not delete snapshot file: {inner_e}")
            except Exception as e:
                logger.warning(f"[Database] Error removing snapshot file {event.get('snapshot_path')}: {e}")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM events WHERE id = ?", (safe_id,))
            conn.commit()
        return True

    def delete_latest_events(self, count: int = 5, delete_snapshots: bool = True) -> List[int]:
        """Deletes latest N events and their snapshots with bounded count."""
        safe_count = max(1, min(int(count), 500))
        recent = self.get_recent_events(limit=safe_count)
        deleted_ids = []
        for ev in recent:
            if self.delete_event(ev["id"], delete_snapshot=delete_snapshots):
                deleted_ids.append(ev["id"])
        return deleted_ids

    def clear_events(self):
        """Clears all events safely."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM events")
            conn.commit()

    def migrate_from_json(self, json_path: Path):
        """Migrates legacy alert_log.json events if the database is currently empty."""
        if not json_path.exists() or self.get_total_count() > 0:
            return

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                alerts = json.load(f)
            
            for a in alerts:
                event_data = {
                    "timestamp": a.get("timestamp", datetime.now().isoformat()),
                    "mode": "NIGHT" if "NIGHT" in a.get("type", "") else "DAY",
                    "threat_score": a.get("clip_threat_score", a.get("audio_confidence", 0.8)),
                    "audio_class": "threat",
                    "audio_confidence": a.get("audio_confidence", 0.0),
                    "audio_rms": 0.0,
                    "person_detected": 1 if a.get("human_detections") else 0,
                    "person_confidence": 0.8 if a.get("human_detections") else 0.0,
                    "clip_score": a.get("clip_threat_score", 0.0),
                    "caption": a.get("blip2_caption"),
                    "snapshot_path": a.get("snapshot", ""),
                    "reason": f"Migrated {a.get('type', 'ALERT')}",
                }
                self.insert_event(event_data)
        except Exception as e:
            logger.warning(f"[Database] Could not migrate legacy alert log: {e}")

    def export_to_csv(self, limit: int = 100) -> str:
        """Exports recent event history as CSV string with bounded limit."""
        safe_limit = max(1, min(int(limit), 5000))
        events = self.get_recent_events(limit=safe_limit)
        if not events:
            return ""
        output = io.StringIO()
        fieldnames = ["id", "timestamp", "mode", "threat_score", "audio_class", "audio_confidence", "person_detected", "clip_score", "reason"]
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for e in events:
            writer.writerow(e)
        return output.getvalue()

    def export_to_latex(self, limit: int = 10) -> str:
        """Exports recent event history as a publication-ready LaTeX table."""
        safe_limit = max(1, min(int(limit), 100))
        events = self.get_recent_events(limit=safe_limit)
        if not events:
            return "% No events logged yet."
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{WildGuard Edge Real-Time Multimodal Intrusion Log}",
            "\\label{tab:wildguard_events}",
            "\\small",
            "\\begin{tabular}{rlllrrrl}",
            "\\hline",
            "\\textbf{ID} & \\textbf{Timestamp} & \\textbf{Mode} & \\textbf{Acoustic Class} & \\textbf{Audio Conf} & \\textbf{CLIP} & \\textbf{Score} & \\textbf{Rationale} \\\\",
            "\\hline",
        ]
        for e in events:
            ts = str(e.get("timestamp", ""))[:19].replace("T", " ")
            mode = str(e.get("mode", ""))
            ac = str(e.get("audio_class", "ambient"))
            aconf = f"{float(e.get('audio_confidence', 0.0)):.2f}"
            clip = f"{float(e.get('clip_score', 0.0)):.2f}"
            tscore = f"{float(e.get('threat_score', 0.0)):.2f}"
            reason = str(e.get("reason", "")).replace("_", "\\_").replace("%", "\\%")
            if len(reason) > 50:
                reason = reason[:47] + "..."
            lines.append(f"{e.get('id')} & {ts} & {mode} & {ac} & {aconf} & {clip} & {tscore} & {reason} \\\\")
        lines.extend([
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
        ])
        return "\n".join(lines)
