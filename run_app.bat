@echo off
title CivicCura Offline AI Launcher
echo ===================================================
echo Launching CivicCura Local Primary Diagnostic Engine
echo ===================================================
cd /d "%~dp0"
".\venv\Scripts\python.exe" -m streamlit run app.py
pause