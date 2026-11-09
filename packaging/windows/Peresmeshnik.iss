; Inno Setup 6 script: the installer of the one-folder build in dist\Peresmeshnik (packaging/windows/build.sh runs
; ISCC with /DVersion=<version>). Per-user by default (no administrator rights, %LOCALAPPDATA%\Programs), a Start menu
; shortcut, start at sign-in only when ticked, an uninstaller. Settings and models stay in the user's folders.
#ifndef Version
  #error pass /DVersion=<version>
#endif
#define Root SourcePath + "..\.."

[Setup]
AppId={{F900FAE6-BF0A-4AA3-B66D-0CE8539E4E55}
AppName=Peresmeshnik
AppVersion={#Version}
AppVerName=Peresmeshnik {#Version}
AppPublisher=Akty
AppPublisherURL=https://github.com/a1lezakty/peresmeshnik
AppSupportURL=https://github.com/a1lezakty/peresmeshnik/issues
DefaultDirName={autopf}\Peresmeshnik
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Windows 10 2004: the call app's sound is captured by process loopback
MinVersion=10.0.19041
LicenseFile={#Root}\LICENSE
OutputDir={#Root}\dist
OutputBaseFilename=Peresmeshnik-{#Version}-windows-x64-setup
SetupIconFile={#Root}\assets\icons\app.ico
UninstallDisplayIcon={app}\Peresmeshnik.exe
UninstallDisplayName=Peresmeshnik
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
CloseApplicationsFilter=*.exe,*.dll,*.pyd
RestartApplications=no

[Tasks]
Name: autostart; Description: "Start Peresmeshnik when I sign in"; Flags: unchecked

[InstallDelete]
; an update replaces the libraries as a whole: no file of the old version stays next to the new ones
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "{#Root}\dist\Peresmeshnik\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Peresmeshnik"; Filename: "{app}\Peresmeshnik.exe"
Name: "{autostartup}\Peresmeshnik"; Filename: "{app}\Peresmeshnik.exe"; Tasks: autostart

[Run]
Filename: "{app}\Peresmeshnik.exe"; Description: "Start Peresmeshnik"; Flags: nowait postinstall skipifsilent

[Code]
{ A running Peresmeshnik is asked to quit (WM_CLOSE from outside, as taskkill does, to its hidden window: the title is
  HOST_TITLE in peresmeshnik/ui/windows.py), so it gives the call app its sound back first; then its files can be
  replaced or removed. }
const
  HostTitle = 'Peresmeshnik F900FAE6';

procedure CloseApp();
var
  I: Integer;
begin
  if FindWindowByWindowName(HostTitle) = 0 then
    Exit;
  PostMessage(FindWindowByWindowName(HostTitle), 16, 0, 0);  { WM_CLOSE }
  for I := 1 to 75 do begin  { up to 15 s }
    Sleep(200);
    if FindWindowByWindowName(HostTitle) = 0 then
      Exit;
  end;
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  CloseApp();
  Result := '';
end;

{ The voice pack (the CUDA clone, ~8 GB, in the app's data folder: config.data_dir) goes only when the user says so,
  and with it what an install leaves beside it (local/voicepack.py): voice-pack.download (up to 3.7 GB), voice-pack.tmp,
  voice-pack.pip-tmp, voice-pack.old-*; asked about also when only those are there. A silent uninstall (/SILENT,
  /VERYSILENT: an update, a script) leaves it all, as it leaves the settings, models and voices. }
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Pack: String;
  F: TFindRec;
begin
  if CurUninstallStep = usUninstall then  { after the user confirmed the removal }
    CloseApp();
  if (CurUninstallStep = usPostUninstall) and not UninstallSilent() then begin
    Pack := ExpandConstant('{localappdata}\Peresmeshnik\voice-pack');
    if FindFirst(Pack + '*', F) then begin  { voice-pack itself or voice-pack.<leftover> }
      FindClose(F);
      if MsgBox('Remove the voice pack and its downloads too (up to 12 GB, ' + Pack + ')? Your voice, settings and '
                + 'models stay either way.', mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then begin
        DelTree(Pack, True, True, True);
        DelTree(Pack + '.*', False, True, True);  { the leftovers above: a wildcard, files and folders }
      end;
    end;
  end;
end;
