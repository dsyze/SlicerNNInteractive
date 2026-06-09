@echo off
setlocal EnableExtensions

rem Change only this line when moving this project to another computer.
set "SERVER_DIR=F:\DicomProjectStage\Stage1\SlicerNNInteractive\server"

rem Optional settings. Usually these do not need to be changed.
set "HOST=0.0.0.0"
set "PORT=1527"
rem set "NNI_DEVICE=cuda:0"
rem set "NNI_DEVICE=cpu"

rem Keep the window open when the script is double-clicked.
if /i not "%~1"=="__run" (
    start "nnInteractive Backend" cmd /d /k ""%~f0" __run"
    exit /b
)
shift /1

set "LOG_FILE=%~dp0nni_backend_start.log"
> "%LOG_FILE%" echo nnInteractive backend startup log
>> "%LOG_FILE%" echo Started at %DATE% %TIME%
>> "%LOG_FILE%" echo.

if not exist "%SERVER_DIR%\pyproject.toml" (
    call :log [ERROR] SERVER_DIR is not a valid nnInteractive server source directory:
    call :log         %SERVER_DIR%
    echo.
    call :log Edit SERVER_DIR at the top of this .bat file.
    call :finish 1
    exit /b 1
)

cd /d "%SERVER_DIR%"
if errorlevel 1 (
    call :log [ERROR] Failed to enter server directory:
    call :log         %SERVER_DIR%
    call :finish 1
    exit /b 1
)

call :log [INFO] Server directory: %CD%
call :log [INFO] Listening on: http://localhost:%PORT%
if defined NNI_DEVICE call :log [INFO] NNI_DEVICE=%NNI_DEVICE%
echo.

if exist ".venv\Scripts\nninteractive-slicer-server.exe" (
    call :log [INFO] Using existing .venv console script.
    ".venv\Scripts\nninteractive-slicer-server.exe" --host "%HOST%" --port "%PORT%"
    goto done
)

if exist ".venv\Scripts\python.exe" (
    call :log [INFO] Found .venv Python but no server launcher. Installing package first.
    call :install_server_package
    if errorlevel 1 (
        call :finish 1
        exit /b 1
    )
    if exist ".venv\Scripts\nninteractive-slicer-server.exe" (
        call :log [INFO] Starting server with newly installed launcher.
        ".venv\Scripts\nninteractive-slicer-server.exe" --host "%HOST%" --port "%PORT%"
    ) else (
        call :log [INFO] Launcher was not generated. Starting server as Python module.
        ".venv\Scripts\python.exe" -m nninteractive_slicer_server.main --host "%HOST%" --port "%PORT%"
    )
    goto done
)

where uv >nul 2>nul
if not errorlevel 1 (
    call :log [INFO] Using uv. Dependencies may be installed on first run.
    uv run nninteractive-slicer-server --host "%HOST%" --port "%PORT%"
    goto done
)

set "PY_CMD="
where python >nul 2>nul
if not errorlevel 1 set "PY_CMD=python"

if not defined PY_CMD (
    where py >nul 2>nul
    if not errorlevel 1 set "PY_CMD=py -3"
)

if not defined PY_CMD (
    call :log [ERROR] Neither uv nor Python was found in PATH.
    call :log Install uv or Python 3, then run this script again.
    call :finish 1
    exit /b 1
)

call :log [INFO] Creating .venv with: %PY_CMD%
%PY_CMD% -m venv .venv >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    call :log [ERROR] Failed to create .venv.
    call :finish 1
    exit /b 1
)

call :install_server_package
if errorlevel 1 (
    call :finish 1
    exit /b 1
)

call :log [INFO] Starting server.
".venv\Scripts\nninteractive-slicer-server.exe" --host "%HOST%" --port "%PORT%"

:done
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" call :log [ERROR] Server stopped with exit code %EXIT_CODE%.
call :finish %EXIT_CODE%
exit /b %EXIT_CODE%

:log
echo %*
>> "%LOG_FILE%" echo %*
exit /b 0

:install_server_package
call :log [INFO] Installing server package from source. This can take several minutes.
".venv\Scripts\python.exe" -m pip install --upgrade pip >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    call :log [ERROR] Failed to upgrade pip.
    exit /b 1
)

".venv\Scripts\python.exe" -m pip install -e . >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    call :log [ERROR] Failed to install server package.
    exit /b 1
)
exit /b 0

:finish
set "FINAL_EXIT_CODE=%~1"
echo.
echo Log file:
echo %LOG_FILE%
echo.
if "%FINAL_EXIT_CODE%"=="0" (
    echo Server process has stopped.
) else (
    echo Startup failed or server stopped with exit code %FINAL_EXIT_CODE%.
)
echo Press any key to close this window.
pause >nul
exit /b 0
