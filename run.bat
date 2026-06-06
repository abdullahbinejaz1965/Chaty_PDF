@echo off
cd /d "E:\Generative material\LLM\work\Main llm"

echo Starting backend...
start cmd /k "python -m uvicorn app:app --host 0.0.0.0 --port 8000 "

timeout /t 3 > nul

echo Opening frontend...
start "" "static\index.html"
