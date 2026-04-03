# Day 1 — Tensor Ops Lab (tooling)

The code generator lives **one directory up**: `../setup.py`.  
This folder (**`day1_tensor_lab/`**) holds the generated lesson **and** the helper scripts.

## Layout

| Path | Purpose |
|------|---------|
| `../setup.py` | Generates/refreshes files in this directory (model, app, train, tests, `README.md`). |
| `setup.sh` | Runs `../setup.py`. |
| `build.py` | Calls `../setup.py`, verifies outputs, optional `.venv`, runs `test_model.py`. |
| `start.py` / `startup.sh` | Start Streamlit on `./app.py`. |
| `cleanup.sh` | Local junk cleanup under this tree + destructive Docker prune (see below). |
| `requirements.txt` | Python deps for tests and the Streamlit app. |
| `README.md` | Lesson quick start (from generator). |

## Quick start

```bash
cd day1/day1_tensor_lab
python3 -m pip install -r requirements.txt   # use a venv if PEP 668–managed
./build.py
./start.py
```

## Docker cleanup warning

`./cleanup.sh` stops **all** running containers and runs **`docker system prune -af --volumes`**.

## Secrets

Do not commit API keys. Use env vars or `.env` (gitignored).
