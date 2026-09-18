"""
AuraDent AI - Unified Python Backend & Web Server

Integrates the CLI dental voice engine (MicrophoneRecorder, DentalAI, Finding)
with the interactive 32-tooth web interface on port 3000.
"""

import base64
import json
import mimetypes
import os
import shutil
import sys
import tempfile
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Windows consoles with legacy codepages (e.g. cp1252) crash on emoji prints
# when stdout is redirected to a file; make the streams encoding-tolerant.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

# Import dental_ai package components
from dental_ai.audio import MicrophoneRecorder, wav_peak_amplitude, SILENCE_PEAK_THRESHOLD
from dental_ai.db import DentalDB
from dental_ai.models import Finding
from dental_ai.services import DentalAI, DemoAI, send_email

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
FRONTEND_WEB_DIR = BASE_DIR.parent / "frontend" / "web"
MAMMOTTY_DIR = BASE_DIR.parent / "mammotty"

# Determine web static root
if WEB_DIR.exists() and (WEB_DIR / "index.html").exists():
    STATIC_DIR = WEB_DIR
elif FRONTEND_WEB_DIR.exists() and (FRONTEND_WEB_DIR / "index.html").exists():
    STATIC_DIR = FRONTEND_WEB_DIR
elif MAMMOTTY_DIR.exists() and (MAMMOTTY_DIR / "index.html").exists():
    STATIC_DIR = MAMMOTTY_DIR
else:
    STATIC_DIR = BASE_DIR

# Global state and logs
SERVER_LOGS = []
MAX_LOGS = 100


def log_event(message: str, category: str = "info"):
    timestamp = time.strftime("%H:%M:%S")
    entry = {"time": timestamp, "message": message, "category": category}
    SERVER_LOGS.append(entry)
    if len(SERVER_LOGS) > MAX_LOGS:
        SERVER_LOGS.pop(0)
    print(f"[{timestamp}] [{category.upper()}] {message}", flush=True)


# Initialize AI service
AI_SERVICE = None
try:
    _candidate = DentalAI()
    # The Groq client validates nothing at construction and model detection
    # swallows auth errors, so probe with a real (cheap) API call to ensure an
    # invalid key can never boot the server into a broken "live" mode.
    _candidate.client.models.list()
    AI_SERVICE = _candidate
    log_event(f"Initialized DentalAI with Whisper and {AI_SERVICE.chat_model}", "system")
except Exception as e:
    log_event(f"DentalAI init error ({e}); falling back to DemoAI", "warning")
    AI_SERVICE = DemoAI()

# Lock guarding AI_SERVICE upgrades when the user pastes a GROQ_API_KEY into
# .env while the server is running (checked on every /api/status poll).
AI_SERVICE_LOCK = threading.Lock()
# Remember the last key value that failed validation so a stored invalid key
# is not re-probed against the Groq API on every status poll.
LAST_FAILED_GROQ_KEY = None


def try_upgrade_to_live_ai() -> None:
    """Hot-reload the Groq key from .env without requiring a server restart.

    Called from /api/status. If the key appears (or is fixed) in .env, a live
    DentalAI is built and swapped in; if the key is invalid the server stays
    in demo mode and tells the frontend what to display.
    """
    global AI_SERVICE, LAST_FAILED_GROQ_KEY
    with AI_SERVICE_LOCK:
        if isinstance(AI_SERVICE, DentalAI):
            return  # already live
        load_dotenv(override=True)  # re-read .env in case the key was just pasted
        key = os.getenv("GROQ_API_KEY", "").strip()
        if not key:
            return  # still no key; stay in demo mode silently
        if key == LAST_FAILED_GROQ_KEY:
            return  # already rejected this exact key; don't re-probe every poll
        try:
            candidate = DentalAI(api_key=key)
            # _detect_best_chat_model() swallows auth errors, so probe the key
            # with a real (cheap) API call before upgrading to live mode.
            candidate.client.models.list()
            AI_SERVICE = candidate
            LAST_FAILED_GROQ_KEY = None
            log_event(f"GROQ_API_KEY validated - upgraded to live AI ({candidate.chat_model}). No restart needed.", "system")
        except Exception as exc:
            # Only cache auth failures so a transient network outage doesn't
            # permanently block the upgrade retry loop.
            text = str(exc)
            if "401" in text or "403" in text or "api key" in text.lower():
                LAST_FAILED_GROQ_KEY = key
            log_event(f"GROQ_API_KEY found but invalid/unusable ({exc}); staying in demo mode. Check the key at console.groq.com/keys.", "warning")

RECORDER = MicrophoneRecorder()
DB = DentalDB()
DICTATION_DIR = BASE_DIR / "dictation_audio"
DICTATION_DIR.mkdir(exist_ok=True)


class DentalAppRequestHandler(SimpleHTTPRequestHandler):
    """Handles API requests and serves web interface assets."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.OK)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/status":
            self.handle_get_status()
        elif path == "/api/logs":
            self.handle_get_logs()
        elif path == "/api/sessions":
            self.handle_get_sessions()
        elif path == "/api/reports":
            self.handle_get_reports()
        elif path == "/api/dictations":
            self.handle_get_dictations()
        else:
            # Fallback to serving static files from STATIC_DIR
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_len = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_len) if content_len > 0 else b"{}"

        try:
            payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except Exception:
            payload = {}

        if path == "/api/record-mic":
            self.handle_record_mic(payload)
        elif path == "/api/transcribe":
            self.handle_transcribe(raw_body)
        elif path == "/api/parse":
            self.handle_parse(payload)
        elif path == "/api/report":
            self.handle_report(payload)
        elif path == "/api/send-email":
            self.handle_send_email(payload)
        elif path == "/api/config/credentials":
            self.handle_save_credentials(payload)
        else:
            self._send_json({"error": f"Endpoint not found: {path}"}, status=404)

    # ------------------ API Handlers ------------------

    def handle_get_status(self):
        try_upgrade_to_live_ai()  # pick up a freshly pasted GROQ_API_KEY from .env
        is_live = isinstance(AI_SERVICE, DentalAI)
        model_name = getattr(AI_SERVICE, "chat_model", "Local Regex Engine")
        whisper_model = getattr(AI_SERVICE, "whisper_model", "Local Demo")
        self._send_json({
            "status": "online",
            "backend": "Python DentalAI (Groq Whisper + LLM)",
            "live_mode": is_live,
            "whisper_model": whisper_model,
            "chat_model": model_name,
            "groq_key_configured": is_live or bool(os.getenv("GROQ_API_KEY", "").strip()),
            "smtp_configured": bool(os.getenv("SMTP_USERNAME", "").strip() and os.getenv("SMTP_PASSWORD", "").strip()),
            "static_root": str(STATIC_DIR),
            "log_count": len(SERVER_LOGS)
        })

    def handle_get_logs(self):
        self._send_json({"logs": SERVER_LOGS})

    def handle_get_sessions(self):
        sessions = DB.get_recent_sessions(limit=20)
        self._send_json({"sessions": sessions})

    def handle_get_reports(self):
        reports = DB.get_reports(limit=50)
        self._send_json({"reports": reports})

    def handle_get_dictations(self):
        dictations = DB.get_dictations(limit=50)
        self._send_json({"dictations": dictations})

    def handle_save_credentials(self, payload):
        """Saves GROQ_API_KEY / SMTP credentials into .env (in-memory only),
        then hot-validates the Groq key so live mode activates immediately."""
        try:
            env_path = BASE_DIR / ".env"
            lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []

            values = {
                "GROQ_API_KEY": str(payload.get("groq_api_key") or "").strip(),
                "SMTP_USERNAME": str(payload.get("smtp_username") or "").strip(),
                "SMTP_PASSWORD": str(payload.get("smtp_password") or "").strip(),
            }
            updated = []
            seen = set()
            for line in lines:
                key = line.split("=", 1)[0].strip() if "=" in line else None
                if key in values:
                    updated.append(f"{key}={values[key]}")
                    seen.add(key)
                else:
                    updated.append(line)
            for key, value in values.items():
                if key not in seen and value:  # append only keys actually provided
                    updated.append(f"{key}={value}")

            env_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
            load_dotenv(override=True)
            log_event("Credentials saved to .env from setup checklist", "system")

            result = {"success": True, "message": "Credentials saved."}
            if values["GROQ_API_KEY"]:
                try_upgrade_to_live_ai()
                is_live = isinstance(AI_SERVICE, DentalAI)
                result["groq_live"] = is_live
                result["message"] = (
                    f"Groq key valid - live Whisper mode active ({getattr(AI_SERVICE, 'chat_model', '')})."
                    if is_live
                    else "Key saved but validation failed - check the value at console.groq.com/keys."
                )
            if values["SMTP_USERNAME"] and values["SMTP_PASSWORD"]:
                result["smtp_saved"] = True
            self._send_json(result)
        except Exception as exc:
            log_event(f"Credential save failed: {exc}", "error")
            self._send_json({"success": False, "error": str(exc)}, status=500)

    def handle_record_mic(self, payload):
        """Records 5 seconds of audio from the physical microphone and transcribes it."""
        seconds = int(payload.get("seconds", 5))
        patient_name = str(payload.get("patient_name") or "Active Patient")
        log_event(f"Microphone recording requested ({seconds}s)...", "mic")

        try:
            audio_path = RECORDER.record_chunk(seconds=seconds)
            log_event(f"Recorded chunk to {audio_path.name}. Transcribing with Whisper...", "mic")

            # Archive the dictation audio for replay/audit (never deleted).
            archive_name = f"{time.strftime('%Y%m%d-%H%M%S')}_{patient_name.strip().replace(' ', '_') or 'patient'}.wav"
            archive_path = DICTATION_DIR / archive_name
            try:
                shutil.copyfile(audio_path, archive_path)
                archived = True
            except Exception as arch_exc:
                log_event(f"Dictation archive failed: {arch_exc}", "warning")
                archived = False

            transcript = ""
            if isinstance(AI_SERVICE, DentalAI):
                # Silence gate: Whisper only hallucinates on quiet/noise-only
                # audio, so check the RAW (pre-normalization) mic level first
                # and skip the API entirely when the mic heard nothing usable.
                peak = RECORDER.last_raw_peak
                if peak < SILENCE_PEAK_THRESHOLD:
                    audio_path.unlink(missing_ok=True)
                    if archived:
                        DB.save_dictation(patient_name, str(archive_path), transcript=None, source="python-mic-silent")
                    log_event(f"Mic captured silence (peak {peak}) - transcription skipped", "warning")
                    self._send_json({
                        "success": False,
                        "error": "no_speech",
                        "message": "No speech detected - the mic heard only silence. Speak a little louder and closer to the mic, then try again.",
                        "archived": archived,
                    })
                    return
                try:
                    transcript = AI_SERVICE.transcribe(str(audio_path)).strip()
                except Exception as ex:
                    log_event(f"Transcription failed: {ex}", "error")
                    transcript = ""
            else:
                # DEMO MODE: no GROQ_API_KEY configured, so the captured audio
                # cannot be transcribed. Tell the client clearly instead of
                # silently injecting a canned finding.
                if archived:
                    DB.save_dictation(patient_name, str(archive_path), transcript=None, source="python-mic-untranscribed")
                self._send_json({
                    "success": False,
                    "error": "demo_mode_no_api_key",
                    "message": "Demo mode: paste your GROQ_API_KEY in the setup card (or .env) to enable Whisper transcription - it activates automatically, no restart. Meanwhile the browser microphone dictation works without any key.",
                    "archived": archived,
                })
                return

            audio_path.unlink(missing_ok=True)

            findings = []
            findings_objs = []
            if transcript:
                log_event(f"Transcript: '{transcript}'. Extracting findings...", "ai")
                findings_objs = AI_SERVICE.parse_findings(transcript)
                findings = [f.as_dict() for f in findings_objs]
                DB.save_parsed_findings(patient_name, transcript, findings_objs, source="voice")
                if archived:
                    DB.save_dictation(patient_name, str(archive_path), transcript=transcript, source="python-mic")
                log_event(f"Extracted {len(findings)} finding(s) and saved to SQLite (audio archived: {archived})", "ai")
            else:
                if archived:
                    DB.save_dictation(patient_name, str(archive_path), transcript=None, source="python-mic-silent")
                log_event("Silence or no audio transcribed", "warning")

            self._send_json({
                "success": True,
                "transcript": transcript,
                "findings": findings,
                "archived": archived,
                "timestamp": time.strftime("%H:%M:%S")
            })
        except Exception as exc:
            log_event(f"Mic record error: {exc}", "error")
            self._send_json({"success": False, "error": str(exc)}, status=500)

    def handle_transcribe(self, raw_body):
        """Transcribes incoming audio file data."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(raw_body)
                tmp_path = tmp.name

            try:
                # Same silence gate as /api/record-mic: never send quiet
                # noise-only audio to Whisper.
                try:
                    peak = wav_peak_amplitude(Path(tmp_path))
                except Exception:
                    peak = SILENCE_PEAK_THRESHOLD
                if peak < SILENCE_PEAK_THRESHOLD:
                    log_event("Uploaded audio is silence - transcription skipped", "warning")
                    self._send_json({"success": True, "transcript": "", "message": "No speech detected in audio."})
                    return
                transcript = AI_SERVICE.transcribe(tmp_path).strip()
                log_event(f"Transcribed audio chunk: '{transcript}'", "ai")
                self._send_json({"success": True, "transcript": transcript})
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        except Exception as exc:
            log_event(f"Audio transcribe error: {exc}", "error")
            self._send_json({"success": False, "error": str(exc)}, status=500)

    def handle_parse(self, payload):
        """Parses transcript into structured Finding objects."""
        transcript = payload.get("transcript", "").strip()
        patient_name = str(payload.get("patient_name") or "Active Patient")
        recent_data = payload.get("recent", [])

        recent_findings = []
        for r in recent_data:
            try:
                recent_findings.append(Finding(
                    tooth_number=str(r.get("tooth_number", r.get("toothId", ""))),
                    surface=str(r.get("surface", "unspecified")),
                    finding_type=str(r.get("finding_type", r.get("condition", "clinical finding"))),
                    value=r.get("value"),
                    unit=r.get("unit"),
                    notes=r.get("notes"),
                    needs_review=bool(r.get("needs_review", False)),
                    status=str(r.get("status", "active"))
                ))
            except Exception:
                pass

        log_event(f"Parsing speech input: '{transcript}'", "ai")
        findings_objs = AI_SERVICE.parse_findings(transcript, recent=recent_findings)
        findings = [f.as_dict() for f in findings_objs]
        if transcript:
            DB.save_parsed_findings(patient_name, transcript, findings_objs, source="voice")
        log_event(f"Identified {len(findings)} finding(s) from speech and saved to SQLite", "ai")

        self._send_json({
            "success": True,
            "transcript": transcript,
            "findings": findings
        })

    def handle_report(self, payload):
        """Generates patient-friendly report from approved findings."""
        findings_data = payload.get("findings", [])
        findings_objs = []
        for f in findings_data:
            try:
                findings_objs.append(Finding(
                    tooth_number=str(f.get("tooth_number", f.get("toothId", ""))),
                    surface=str(f.get("surface", "unspecified")),
                    finding_type=str(f.get("finding_type", f.get("condition", "clinical finding"))),
                    value=f.get("value"),
                    unit=f.get("unit"),
                    notes=f.get("notes") or f.get("clinicalNote"),
                    needs_review=bool(f.get("needs_review", False)),
                    status=str(f.get("status", "active"))
                ))
            except Exception:
                pass

        log_event(f"Generating patient report for {len(findings_objs)} findings...", "ai")
        report = AI_SERVICE.generate_report(findings_objs)
        log_event("Patient report generated successfully", "ai")

        self._send_json({
            "success": True,
            "report": report
        })

    def handle_send_email(self, payload):
        """Generates (if needed), stores in the database, and emails the patient report."""
        report = (payload.get("report") or "").strip()
        recipient = payload.get("recipient") or os.getenv("PATIENT_EMAIL", "patient@example.com")
        subject = payload.get("subject", "Your Dental Visit Summary - AuraDent AI")
        patient_name = str(payload.get("patient_name") or "Active Patient")
        findings_data = payload.get("findings", [])

        # Generate the report server-side when the client sends no body
        # (e.g. the UI passes approved findings instead of a composed email).
        if not report and findings_data:
            findings_objs = []
            for f in findings_data:
                try:
                    findings_objs.append(Finding(
                        tooth_number=str(f.get("tooth_number", f.get("toothId", ""))),
                        surface=str(f.get("surface", "unspecified")),
                        finding_type=str(f.get("finding_type", f.get("condition", "clinical finding"))),
                        value=f.get("value"),
                        unit=f.get("unit"),
                        notes=f.get("notes") or f.get("clinicalNote"),
                        needs_review=bool(f.get("needs_review", False)),
                        status=str(f.get("status", "active")),
                    ))
                except Exception:
                    pass
            try:
                report = AI_SERVICE.generate_report(findings_objs)
            except Exception as exc:
                log_event(f"Report generation failed: {exc}", "error")
                self._send_json({"success": False, "error": f"Report generation failed: {exc}"}, status=500)
                return

        if not report:
            self._send_json({"success": False, "error": "No report body or findings provided"}, status=400)
            return

        log_event(f"Dispatching patient report email to {recipient}...", "email")
        delivery_status = "sent"
        error_message = None
        response = {
            "success": True,
            "message": f"Report emailed to {recipient}",
            "recipient": recipient,
            "stored": True,
        }

        try:
            send_email(report, recipient, subject)
            log_event(f"Email sent to {recipient}", "email")
        except Exception as exc:
            # No SMTP credentials or relay problem: keep the report, mark it
            # 'failed' in the DB and save a local preview file as fallback.
            delivery_status = "failed"
            error_message = str(exc)
            out_file = BASE_DIR / "sent_report_preview.txt"
            out_file.write_text(f"To: {recipient}\nSubject: {subject}\n\n{report}", encoding="utf-8")
            response["warning"] = f"SMTP unavailable ({exc}); report saved to sent_report_preview.txt and stored in database."
            log_event(f"Email send warning: {exc} (report stored with status 'failed'; preview saved)", "warning")

        # Persist to the clinic database (dental_data.db, reports table)
        try:
            report_id = DB.save_report(
                patient_name=patient_name,
                recipient=recipient,
                subject=subject,
                report_body=report,
                delivery_status=delivery_status,
                error_message=error_message,
            )
            response["report_id"] = report_id
        except Exception as exc:
            log_event(f"Failed to store report in database: {exc}", "error")
            response["warning"] = (response.get("warning") or "") + f" DB storage failed: {exc}"

        self._send_json(response)


def run_server(port: int = 3000, open_browser: bool = True):
    server_address = ("", port)
    httpd = ThreadingHTTPServer(server_address, DentalAppRequestHandler)
    url = f"http://localhost:{port}"

    print("=" * 65)
    print("🦷 AuraDent AI - Voice-First Dental Assistant")
    print("=" * 65)
    print(f"Web Interface: {url}")
    print(f"Serving From:  {STATIC_DIR}")
    print(f"Backend Engine: {'DentalAI (Groq Whisper + LLM)' if isinstance(AI_SERVICE, DentalAI) else 'DemoAI'}")
    print("API Endpoints: /api/status, /api/record-mic, /api/parse, /api/report, /api/logs")
    print("=" * 65)

    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping AuraDent AI server.")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 3000
    run_server(port=port, open_browser=False)
