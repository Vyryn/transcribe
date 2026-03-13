#ifndef SourceDir
  #error "SourceDir macro must be defined."
#endif

#ifndef OutputDir
  #error "OutputDir macro must be defined."
#endif

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

#ifndef OutputBaseFilename
  #define OutputBaseFilename "transcribe-setup"
#endif

#define AppName "Transcribe"
#define AppPublisher "Transcribe"

[Setup]
AppId={{2D9D88E0-12D3-4B3A-A1E5-15C84CAAB4A3}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
UninstallDisplayIcon={app}\transcribe.exe
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
OutputDir={#OutputDir}
OutputBaseFilename={#OutputBaseFilename}
PrivilegesRequired=admin

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\Transcribe"; Filename: "{app}\transcribe.exe"
Name: "{group}\Uninstall Transcribe"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\transcribe.exe"; Description: "Launch Transcribe"; Flags: nowait postinstall skipifsilent unchecked

[Code]
var
  ModelBootstrapPage: TOutputProgressWizardPage;

procedure InitializeWizard();
begin
  ModelBootstrapPage := CreateOutputProgressPage(
    'Preparing Transcribe',
    'Downloading the required offline models.'
  );
end;

procedure FailBootstrap(const MessageText: string);
begin
  MsgBox(MessageText, mbCriticalError, MB_OK);
  Abort;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
begin
  if CurStep <> ssPostInstall then
    exit;

  ModelBootstrapPage.SetText(
    'Downloading required models',
    'This can take several minutes on the first install. Please keep this window open.'
  );
  ModelBootstrapPage.Show;
  try
    if not Exec(
      ExpandConstant('{app}\transcribe.exe'),
      'models install --default --quiet',
      ExpandConstant('{app}'),
      SW_HIDE,
      ewWaitUntilTerminated,
      ResultCode
    ) then
    begin
      FailBootstrap('Transcribe was installed, but the required model download could not be started.');
    end;

    if ResultCode <> 0 then
    begin
      FailBootstrap(Format('Transcribe was installed, but the required model download failed with exit code %d.', [ResultCode]));
    end;
  finally
    ModelBootstrapPage.Hide;
  end;
end;
