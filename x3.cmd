@echo off
rem Runs the exploratory X3 model (MVTec LOCO) with this folder's .venv:
rem   x3.cmd prepare     fits branches and normalizers (requires data and weights)
rem   x3.cmd evaluate    LOCO test evaluation, outputs labelled EXPLORATORY in runs_rerun\exploratory
rem   x3.cmd smoke       synthetic fixture (code correctness only)
setlocal
set "ROOT=%~dp0"
set "PYTHONPATH=%ROOT%src;%PYTHONPATH%"
"%ROOT%.venv\Scripts\python.exe" "%ROOT%scripts\x3_run.py" %*
exit /b %ERRORLEVEL%
