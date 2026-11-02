@echo off
rem The run registry (Agents\Training\runlog.py) from the repository root, e.g.
rem   hexrun run --kind duel --name NAME -- COMMAND ...      hexrun list --kind train      hexrun show NAME
rem Uses the Agents\env interpreter if there is one, otherwise python from PATH.
setlocal
set "HEXRUN_PYTHON=%~dp0Agents\env\python.exe"
if not exist "%HEXRUN_PYTHON%" set "HEXRUN_PYTHON=python"
"%HEXRUN_PYTHON%" "%~dp0Agents\Training\runlog.py" %*
exit /b %ERRORLEVEL%
