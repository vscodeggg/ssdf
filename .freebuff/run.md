# AuraDent AI — Preview Run Doc

Voice-first dental charting app. A Python backend serves both the API and the
static web UI on **port 3000**. No JS build step and no npm install are needed:
React is vendored (`frontend/web/vendor/`) and transpiled in the browser by
Babel standalone. Python dependencies (`groq`, `sounddevice`, `numpy`,
`python-dotenv`) are installed system-wide.

> NOTE: The workspace was restructured into `backend/` + `frontend/`. The run
> doc below reflects the new layout.

## 1. Layout

- `backend/server.py` — API + static file server (default port **3000**;
  `python server.py 3001` overrides)
- `backend/dental_ai/` — Python package (audio, services, db, models)
- `backend/.env` — secrets: `GROQ_API_KEY` (from console.groq.com/keys) and
  `SMTP_USERNAME` / `SMTP_PASSWORD` (16-char Gmail App Password). Never commit.
- `backend/dental_data.db` — SQLite store (sessions, findings, reports, dictations)
- `backend/dictation_audio/` — archived dictation WAV files
- `frontend/web/` — static UI served by the server (`index.html`,
  `standalone-app.js`, `vendor/`). `frontend/package.json` exists for an
  optional Vite build but is NOT needed to run.
- `backend/server.py` resolves the static root in order: `backend/web/` →
  `frontend/web/` → `../mammotty` → backend dir. In this workspace it serves
  `frontend/web/`.

## 2. Run the server (from backend/)

```powershell
cd backend
python server.py            # serves http://localhost:3000
```

On Windows consoles `server.py` reconfigures stdout/stderr to UTF-8 so the
emoji banner does not crash when output is redirected.

Detached launch used for the Freebuff preview (PowerShell; stdout and stderr
must go to DIFFERENT files; use `-WorkingDirectory backend`):

```powershell
powershell -NoProfile -Command "(Start-Process -FilePath 'python.exe' -ArgumentList 'server.py' -WorkingDirectory 'C:\Users\krish\Documents\Default Project\backend' -RedirectStandardOutput 'C:\Users\krish\Documents\Default Project\.freebuff\preview.log' -RedirectStandardError 'C:\Users\krish\Documents\Default Project\.freebuff\preview.log.err' -WindowStyle Hidden -PassThru).Id"
```

(The Start-Process call can hang the invoking shell past its timeout while the
server still starts — check `netstat -ano | findstr :3000` before retrying.)

## 3. Health check

- `GET http://localhost:3000/api/status` →
  `{"status":"online","live_mode":true,"groq_key_configured":true,"smtp_configured":false,"static_root":"...\\frontend\\web"}`
- `live_mode` is true once `GROQ_API_KEY` is valid (Whisper large-v3-turbo +
  gpt-oss-20b via Groq). Without a key it runs in DEMO mode: mic audio is
  captured and archived, and the UI falls back to the browser's speech engine.
  Pasting a key into the in-app setup checklist (or `.env`) hot-activates
  Whisper without a restart.
