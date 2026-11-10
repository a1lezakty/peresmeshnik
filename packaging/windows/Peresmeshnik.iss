; SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR GPL-3.0-or-later
; (GPL too: it installs the GPL program piper-runner, and GPL-3.0 counts its install scripts in its source)
; Inno Setup 6 script: the installer of the one-folder build in dist\Peresmeshnik (packaging/windows/build.sh runs
; ISCC with /DVersion=<version>). Per-user by default (no administrator rights, %LOCALAPPDATA%\Programs), a Start menu
; shortcut, start at sign-in only when ticked, an uninstaller. Settings (with the Gemini key) go only when the user says
; so at uninstall; voices and models stay in the user's folders.
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
; the sign-in shortcut of an earlier install, when the box is unticked now (Inno Setup keeps it otherwise)
Type: files; Name: "{autostartup}\Peresmeshnik.lnk"; Tasks: not autostart

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
  replaced or removed. A `keep` helper it left (peresmeshnik-loopback.exe waiting to give a call app its volume back,
  audio/capture_windows.py) outlives it and holds that file: it is ended too (the next start watches again). }
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
      Break;
  end;
end;

procedure EndHelpers();
var
  Code: Integer;
begin
  Exec(ExpandConstant('{sys}\taskkill.exe'), '/F /IM peresmeshnik-loopback.exe', '', SW_HIDE, ewWaitUntilTerminated, Code);
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  CloseApp();
  EndHelpers();
  Result := '';
end;

{ The names of the call apps whose sound Peresmeshnik turned almost all the way down and did not give back yet: the
  keys of muted-session.json (audio/capture_windows.py: a JSON object of the call app's exe name and its state, as
  "Zoom.exe": {"level": 0.8, "devices": ["{0.0.0.00000000}.{...}"], "keeper": 1234}), that is every quoted string
  ending in .exe (the device ids do not), joined with ", "; '' when there is none. }
function MutedApps(const FileName: String): String;
var
  Raw: AnsiString;
  S, Key: String;
  I, J: Integer;
begin
  Result := '';
  if not LoadStringFromFile(FileName, Raw) then
    Exit;
  S := String(Raw);
  I := Pos('"', S);
  while I > 0 do begin
    Delete(S, 1, I);
    J := Pos('"', S);
    if J > 0 then begin
      Key := Copy(S, 1, J - 1);
      if (Length(Key) > 4) and (Lowercase(Copy(Key, Length(Key) - 3, 4)) = '.exe') and (Pos(Key, Result) = 0) then begin
        if Result <> '' then
          Result := Result + ', ';
        Result := Result + Key;
      end;
      Delete(S, 1, J);
      I := Pos('"', S);
    end else
      I := 0;
  end;
end;

{ After the files are gone, unless the uninstall is silent (/SILENT, /VERYSILENT: an update, a script), which leaves
  all of it:
  - the call apps still at volume 0 (muted-session.json in the settings folder) are named, to raise them by hand;
  - the voice pack (the CUDA clone, ~8 GB, in the app's data folder: config.data_dir) goes only when the user says so,
    and with it what an install leaves beside it (local/voicepack.py): voice-pack.download (up to 3.7 GB),
    voice-pack.tmp, voice-pack.pip-tmp, voice-pack.old-*; asked about also when only those are there;
  - the settings go only when the user says so, and with them the Gemini API key, which is a plain-text file there.
  Settings, data and cache are one folder on Windows (config.config_dir, data_dir, cache_dir: %LOCALAPPDATA%\Peresmeshnik):
  the voices (voices), logs, crash reports and the models (Cache\models) stay, and the dialog says where. }
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Dir, Pack, Apps: String;
  F: TFindRec;
begin
  if CurUninstallStep = usUninstall then begin  { after the user confirmed the removal }
    CloseApp();
    EndHelpers();
  end;
  if (CurUninstallStep = usPostUninstall) and not UninstallSilent() then begin
    Dir := ExpandConstant('{localappdata}\Peresmeshnik');
    Apps := MutedApps(Dir + '\muted-session.json');
    if Apps <> '' then
      MsgBox('Peresmeshnik turned the sound of ' + Apps + ' almost all the way down while translating and could not '
             + 'turn it back up yet. Raise it in the Windows Volume mixer (right-click the speaker icon on the taskbar, '
             + 'then Volume mixer).',
             mbInformation, MB_OK);
    Pack := Dir + '\voice-pack';
    if FindFirst(Pack + '*', F) then begin  { voice-pack itself or voice-pack.<leftover> }
      FindClose(F);
      if MsgBox('Remove the voice pack and its downloads too (up to 12 GB, ' + Pack + ')? Your voices and models stay '
                + 'either way.', mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then begin
        DelTree(Pack, True, True, True);
        DelTree(Pack + '.*', False, True, True);  { the leftovers above: a wildcard, files and folders }
      end;
    end;
    if FileExists(Dir + '\settings.toml') or FileExists(Dir + '\gemini-key') then
      if MsgBox('Remove your settings and your Gemini API key too? The key is stored as plain text in ' + Dir
                + '\gemini-key: anyone signed in to your Windows account can read it.' + #13#10#13#10
                + 'Your voices, logs and the downloaded models stay either way, in ' + Dir + ' (the models in its Cache '
                + 'folder): delete that folder to remove them as well.', mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES
      then begin
        DeleteFile(Dir + '\gemini-key');
        DeleteFile(Dir + '\muted-session.json');
        DeleteFile(Dir + '\peresmeshnik.lock');
        DelTree(Dir + '\settings.toml*', False, True, False);  { settings.toml and the temporary file of a save }
      end;
  end;
end;
