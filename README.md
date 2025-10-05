# 1) Create venv
python -m venv .venv && source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run server (GPU if available)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 4) Try file transcription
curl -F "file=@sample.wav" http://localhost:8000/v1/transcribe

# 5) Open the client
# serve client/demo.html (e.g., with Python http.server) and speak
python -m http.server 9000
# then open http://localhost:9000/client/demo.html
