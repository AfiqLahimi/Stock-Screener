@echo off
cd /d "%~dp0"
echo Memasang library (sekali je, mungkin ambil masa 1-2 minit)...
python -m pip install -r requirements.txt
echo Membuka app dalam browser...
python -m streamlit run app.py
pause
