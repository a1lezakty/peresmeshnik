@echo off
rem Build peresmeshnik-audio.exe with MSVC, no project file (the counterpart of native/macos/audio/build.sh).
rem   build.cmd [out_exe]        (the source main.cpp is always next to this script; out defaults to .\build\)
rem Finds the Visual Studio C++ toolset with vswhere, runs vcvars64, then cl. Used by CI and packaging/windows.
setlocal
set SRC=%~dp0
set OUT=%~1
if "%OUT%"=="" set OUT=%SRC%build\peresmeshnik-audio.exe

set VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe
if not exist "%VSWHERE%" set VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set VSPATH=%%i
if "%VSPATH%"=="" ( echo no Visual Studio C++ toolset found >&2 & exit /b 1 )
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1

for %%d in ("%OUT%") do if not exist "%%~dpd" mkdir "%%~dpd"
if not exist "%SRC%build" mkdir "%SRC%build"
cl /nologo /std:c++17 /EHsc /O2 /W3 /utf-8 /DUNICODE /D_UNICODE "%SRC%main.cpp" ^
   /Fe:"%OUT%" /Fo:"%SRC%build\\" ^
   ole32.lib || exit /b 1
echo built %OUT%
