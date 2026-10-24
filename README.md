# Long-Lead Wildfire Outlook

Predicts wildfire danger ~30 days ahead for forecast areas in all 50 US states.

## Backend (Python / FastAPI)

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env           # then put your real FIRMS_MAP_KEY in .env
python -m pytest tests/        # offline tests, no network needed
```

Download real data (NASA POWER + FIRMS + NOAA CPC) for one or more states:

```bash
python prepare_data.py --states CO
python prepare_data.py              # all 50 states (slow - run in background)
```

States without a downloaded cache automatically fall back to synthetic
practice data (`synthetic_data.py`), so the API always has something to serve.

Run the API:

```bash
uvicorn main:app --reload --port 8000
```

## Frontend (React + Vite + Tailwind + React-Leaflet)

```bash
cd frontend
npm install
npm run dev
```

Opens at http://localhost:5173, proxying `/api/*` to the backend on port 8000.

## Opening in VS Code

Open the repo root folder. `.vscode/settings.json` points the Python
extension at `backend/venv`, so the right interpreter and test runner are
picked up automatically. Run `backend/tests/` from the Testing panel, or
"Backend: FastAPI (uvicorn)" from the Run and Debug panel.

## Export a static snapshot

For a demo page that can't call the live API:

```bash
cd backend
venv\Scripts\python.exe ..\tools\export_snapshot.py
```

Writes `all_states.json` at the repo root.
