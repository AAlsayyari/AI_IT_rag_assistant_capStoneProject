import os
import sys
import subprocess
import threading
from pathlib import Path
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

# Directory where this app.py and ingest.py live
APP_DIR = Path(__file__).resolve().parent

# Global state
process = None
process_lock = threading.Lock()
logs = []

def run_process():
    global process
    # Prefer conda Python (has all dependencies), fall back to sys.executable
    python_exe = r"C:\Users\Abdullah\miniconda3\python.exe"
    if not os.path.exists(python_exe):
        python_exe = sys.executable
    
    # Run the ingestion script using the conda Python interpreter
    proc = subprocess.Popen(
        [python_exe, "ingest.py"],
        cwd=str(APP_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1, # Line buffered
        env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    )

    with process_lock:
        process = proc

    # Read logs line by line
    for line in iter(proc.stdout.readline, ''):
        if line:
            logs.append(line.strip())
            # Keep only the last 500 lines to prevent memory overflow
            if len(logs) > 500:
                logs.pop(0)

    proc.wait()

    with process_lock:
        # Only clear if it's still the same process (not replaced by a new start)
        if process is proc:
            process = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status')
def status():
    with process_lock:
        running = process is not None and process.poll() is None
    return jsonify({"running": running})

@app.route('/toggle', methods=['POST'])
def toggle():
    global process
    action = request.json.get('action')

    if action == 'start':
        with process_lock:
            if process is None or process.poll() is not None:
                logs.clear()
                logs.append("=== Starting Ingestion Process ===")
                thread = threading.Thread(target=run_process)
                thread.daemon = True
                thread.start()
                return jsonify({"status": "started"})
            return jsonify({"status": "already_running"})

    elif action == 'stop':
        with process_lock:
            if process is not None and process.poll() is None:
                process.terminate()
                logs.append("=== Process Terminated by User ===")
                return jsonify({"status": "stopped"})
            return jsonify({"status": "already_stopped"})

@app.route('/logs')
def get_logs():
    return jsonify({"logs": logs})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
