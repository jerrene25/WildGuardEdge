"""
WildGuard Edge — single entrypoint.
Launches the Streamlit dashboard, which owns the full fusion pipeline
(audio CNN -> YOLOv8n -> CLIP -> BLIP-2 -> alert).

Usage:
    python main.py
"""

import subprocess
import sys
import os

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    dashboard_path = os.path.join(project_root, "dashboard", "app.py")

    if not os.path.exists(dashboard_path):
        print(f"ERROR: Could not find dashboard at {dashboard_path}")
        sys.exit(1)

    print("=" * 60)
    print("  WildGuard Edge — Real-Time Anti-Poaching Detection")
    print("=" * 60)
    # Automatically ensure we use the virtual environment's Python (with CUDA PyTorch & PyAudio)
    venv_python = os.path.join(project_root, "venv", "Scripts", "python.exe")
    python_bin = venv_python if os.path.exists(venv_python) else sys.executable

    print(f"  Python Environment: {python_bin}")
    print(f"  Launching dashboard from: {dashboard_path}")
    print("  Press Ctrl+C in this terminal to stop the system.")
    print("=" * 60)

    try:
        subprocess.run(
            [python_bin, "-m", "streamlit", "run", dashboard_path],
            cwd=project_root,
            check=True,
        )
    except KeyboardInterrupt:
        print("\n[main] WildGuard Edge stopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"\n[main] Streamlit exited with an error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()