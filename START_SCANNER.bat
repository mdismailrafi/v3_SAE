@echo off
cd /d %~dp0
python -m streamlit run scanner\app.py
pause
