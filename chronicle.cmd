@echo off
rem The run registry (Agents\Training\runlog.py) from the repository root, e.g.
rem   chronicle run --kind duel --name NAME -- COMMAND ...      chronicle list --kind train      chronicle show NAME
rem Uses the Agents\env interpreter if there is one, otherwise python from PATH.
setlocal
set "CHRONICLE_PYTHON=%~dp0Agents\env\python.exe"
if not exist "%CHRONICLE_PYTHON%" set "CHRONICLE_PYTHON=python"
"%CHRONICLE_PYTHON%" "%~dp0Agents\Training\runlog.py" %*
exit /b %ERRORLEVEL%
