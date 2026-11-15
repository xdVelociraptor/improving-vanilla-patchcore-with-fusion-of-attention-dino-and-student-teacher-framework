@echo off
rem Runs the hybridad command line with this folder's .venv, e.g.
rem   hybridad.cmd --help
rem   hybridad.cmd evaluate --experiment experiments/G_full.yaml --dataset mvtec_loco --exploratory
setlocal
set "ROOT=%~dp0"
set "PYTHONPATH=%ROOT%src;%PYTHONPATH%"
"%ROOT%.venv\Scripts\python.exe" -m hybridad.cli %*
exit /b %ERRORLEVEL%
