#!/bin/bash
# Launcher Mac: guna virtual environment sendiri dalam folder ni (.venv)
cd "$(dirname "$0")" || exit 1

if ! command -v python3 >/dev/null 2>&1; then
  echo "❌ Python 3 tak jumpa. Install dari https://www.python.org/downloads/ dan cuba semula."
  read -r -p "Tekan Enter untuk tutup..."
  exit 1
fi

echo "Python: $(python3 --version)"

if [ ! -d ".venv" ]; then
  echo "Sediakan environment (sekali je)..."
  python3 -m venv .venv || { echo "❌ Gagal buat venv"; read -r -p "Tekan Enter..."; exit 1; }
fi

source .venv/bin/activate
echo "Memasang library (kali pertama ambil 1-3 minit)..."
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt -q || { echo "❌ Gagal install library. Screenshot error atas ni."; read -r -p "Tekan Enter..."; exit 1; }

echo "✅ Siap. Membuka app dalam browser..."
python -m streamlit run app.py --server.headless false --browser.gatherUsageStats false
