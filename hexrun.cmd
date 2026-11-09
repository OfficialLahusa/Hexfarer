@echo off
rem The run registry's former name: forwards everything to chronicle.cmd
call "%~dp0chronicle.cmd" %*
exit /b %ERRORLEVEL%
