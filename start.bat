@echo off
setlocal EnableExtensions
chcp 65001 >nul

cd /d "%~dp0"
set "APP_DIR=%~dp0optichat"

if not exist "%APP_DIR%\package.json" (
  echo [ERROR] optichat\package.json not found.
  exit /b 1
)

if /I "%~1"=="--help" goto :help
if /I "%~1"=="-h" goto :help
if /I "%~1"=="--dev" goto :dev
if /I "%~1"=="--check" goto :check

echo [mode] electron desktop start
pushd "%APP_DIR%"
call :ensure_deps
if errorlevel 1 goto :fail

call npm.cmd run desktop:start
if errorlevel 1 goto :fail
popd
exit /b 0

:dev
echo [mode] electron desktop dev
pushd "%APP_DIR%"
call :ensure_deps
if errorlevel 1 goto :fail

call npm.cmd run desktop:dev
if errorlevel 1 goto :fail
popd
exit /b 0

:check
echo [mode] dependency check
pushd "%APP_DIR%"
call :ensure_deps
set "CHECK_CODE=%ERRORLEVEL%"
popd
exit /b %CHECK_CODE%

:ensure_deps
where node >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Node.js not found. Please install Node.js first.
  exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
  echo [ERROR] npm not found. Please install npm first.
  exit /b 1
)

if not exist "node_modules" goto :install_deps

call node -e "require.resolve('vite/package.json'); require.resolve('electron/package.json'); require.resolve('react/package.json')"
if errorlevel 1 goto :install_deps

echo [OK] dependencies are ready.
exit /b 0

:install_deps
echo [INFO] installing or repairing npm dependencies...
call npm.cmd install
if errorlevel 1 (
  echo [ERROR] npm install failed.
  exit /b 1
)

call node -e "require.resolve('vite/package.json'); require.resolve('electron/package.json'); require.resolve('react/package.json')"
if errorlevel 1 (
  echo [ERROR] dependencies are still incomplete after npm install.
  exit /b 1
)

echo [OK] dependencies are ready.
exit /b 0

:help
echo OptiChat desktop launcher
echo.
echo Usage:
echo   start.bat         ^(install check + build + electron start^)
echo   start.bat --dev   ^(install check + vite + electron dev mode^)
echo   start.bat --check ^(dependency check only^)
echo   start.bat --help
exit /b 0

:fail
set "EXIT_CODE=%ERRORLEVEL%"
popd
echo.
echo [ERROR] start failed.
exit /b %EXIT_CODE%
