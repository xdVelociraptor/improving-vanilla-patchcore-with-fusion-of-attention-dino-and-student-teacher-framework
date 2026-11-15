@echo off
rem Full research archive only: this check requires original datasets, weights, models and saved runs.
setlocal
set "ROOT=%~dp0"
set "PYTHONPATH=%ROOT%src;%PYTHONPATH%"
"%ROOT%.venv\Scripts\python.exe" "%ROOT%check_package.py" %*
exit /b %ERRORLEVEL%
