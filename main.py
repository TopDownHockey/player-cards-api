from flask import Flask
from flask_cors import CORS
import os
import platform
import subprocess
import psutil

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Import route handlers
from routes.live_games import live_games_route
from routes.live_games_pbp import live_games_pbp_route
from routes.test import test_route

@app.route("/")
def home():
    return {"status": "healthy", "message": "Hockey Stats API", "version": "1.0.0"}

@app.route("/health")
def health():
    return {"status": "ok"}

@app.route("/system-info")
def system_info():
    """Detailed system information to diagnose Railway performance"""
    
    # Get CPU info
    cpu_info = {
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_percent_current": psutil.cpu_percent(interval=1),
        "cpu_freq": None,
        "cpu_model": None,
    }
    
    # Try to get CPU frequency
    try:
        freq = psutil.cpu_freq()
        if freq:
            cpu_info["cpu_freq"] = {
                "current_mhz": freq.current,
                "min_mhz": freq.min,
                "max_mhz": freq.max
            }
    except:
        pass
    
    # Try to get CPU model (Linux)
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if 'model name' in line:
                    cpu_info["cpu_model"] = line.split(':')[1].strip()
                    break
    except:
        pass
    
    # Memory info
    mem = psutil.virtual_memory()
    memory_info = {
        "total_gb": round(mem.total / (1024**3), 2),
        "available_gb": round(mem.available / (1024**3), 2),
        "used_gb": round(mem.used / (1024**3), 2),
        "percent_used": mem.percent
    }
    
    # Disk I/O
    disk = psutil.disk_usage('/')
    disk_info = {
        "total_gb": round(disk.total / (1024**3), 2),
        "used_gb": round(disk.used / (1024**3), 2),
        "free_gb": round(disk.free / (1024**3), 2),
        "percent_used": disk.percent
    }
    
    # Network test - time a simple request
    import time
    import requests
    net_start = time.time()
    try:
        requests.get('https://api-web.nhle.com/v1/schedule/now', timeout=5)
        nhl_api_latency_ms = round((time.time() - net_start) * 1000, 2)
    except:
        nhl_api_latency_ms = "error"
    
    # Platform info
    platform_info = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
    }
    
    # Environment
    env_info = {
        "railway": bool(os.environ.get('RAILWAY_ENVIRONMENT')),
        "vercel": bool(os.environ.get('VERCEL_ENV')),
        "port": os.environ.get('PORT', 'not set'),
    }
    
    return {
        "cpu": cpu_info,
        "memory": memory_info,
        "disk": disk_info,
        "platform": platform_info,
        "environment": env_info,
        "nhl_api_latency_ms": nhl_api_latency_ms
    }

# Register routes
app.add_url_rule("/api/live-games", "live_games", live_games_route, methods=["GET"])
app.add_url_rule("/api/live-games-pbp", "live_games_pbp", live_games_pbp_route, methods=["GET"])
app.add_url_rule("/api/test", "test", test_route, methods=["GET"])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

