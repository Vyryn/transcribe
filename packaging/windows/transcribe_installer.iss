#ifndef AppVersion
  #error AppVersion define is required.
#endif
#ifndef SourceDir
  #error SourceDir define is required.
#endif
#ifndef OutputDir
  #error OutputDir define is required.
#endif
#ifndef OutputBaseFilename
  #error OutputBaseFilename define is required.
#endif

#define AppName "Transcribe"
#define AppExeName "Transcribe.exe"
#define AppId "{{7C2A4F4B-5D3D-4C52-BD28-AE43FC7044AC}}"
#define AppPublisher "Transcribe"

[Setup]
AppId={#AppId}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
Compression=lzma2/ultra64
SolidCompression=yes
PrivilegesRequired=admin
OutputDir={#OutputDir}
OutputBaseFilename={#OutputBaseFilename}
SetupIconFile={#SourceDir}\transcribe.ico
UninstallDisplayIcon={app}\transcribe.ico

[Tasks]
Name: "startmenuicon"; Description: "Create a Start menu shortcut"; Flags: checkedonce
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Files]
Source: "{#SourceDir}\Transcribe.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\packaged-assets.json"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\transcribe.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\prompts\*"; DestDir: "{app}\prompts"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#SourceDir}\runtime\*"; DestDir: "{app}\runtime"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Transcribe"; Filename: "{app}\Transcribe.exe"; WorkingDir: "{app}"; IconFilename: "{app}\Transcribe.exe"; IconIndex: 0; Tasks: startmenuicon
Name: "{commondesktop}\Transcribe"; Filename: "{app}\Transcribe.exe"; WorkingDir: "{app}"; IconFilename: "{app}\Transcribe.exe"; IconIndex: 0; Tasks: desktopicon

[Code]
var
  ModelsPage: TWizardPage;
  VoiceParakeetCheckBox: TNewCheckBox;
  VoiceCanaryCheckBox: TNewCheckBox;
  Notes4BCheckBox: TNewCheckBox;
  Notes2BCheckBox: TNewCheckBox;

function BuildSelectedModelArguments(): String;
begin
  Result := '';
  if VoiceParakeetCheckBox.Checked then
    Result := Result + ' --model nvidia/parakeet-tdt-0.6b-v3';
  if VoiceCanaryCheckBox.Checked then
    Result := Result + ' --model nvidia/canary-qwen-2.5b';
  if Notes4BCheckBox.Checked then
    Result := Result + ' --model qwen3.5:4b-q4_K_M';
  if Notes2BCheckBox.Checked then
    Result := Result + ' --model qwen3.5:2b-q4_K_M';
end;

function InstallSelectedModels(): Boolean;
var
  ModelArguments: String;
  CommandLine: String;
  ResultCode: Integer;
begin
  Result := True;
  ModelArguments := BuildSelectedModelArguments();
  if ModelArguments = '' then
    Exit;

  WizardForm.StatusLabel.Caption :=
    'Downloading the selected transcription and notes models. This can take several minutes.';
  CommandLine :=
    '/C set "TRANSCRIBE_ALLOW_NETWORK=1" && "' + ExpandConstant('{app}\Transcribe.exe') +
    '" models install' + ModelArguments;

  if not Exec(
    ExpandConstant('{cmd}'),
    CommandLine,
    '',
    SW_SHOW,
    ewWaitUntilTerminated,
    ResultCode
  ) then
  begin
    MsgBox(
      'The installer could not launch the bundled model downloader. ' +
      'You can retry later from the installed application.',
      mbError,
      MB_OK
    );
    Result := False;
    Exit;
  end;

  if ResultCode <> 0 then
  begin
    MsgBox(
      'The installer could not download the selected models. ' +
      'You can retry later from the installed application.',
      mbError,
      MB_OK
    );
    Result := False;
  end;
end;

procedure InitializeWizard();
begin
  ModelsPage := CreateCustomPage(
    wpSelectTasks,
    'Model Downloads',
    'Choose which transcription and notes models to download during setup.'
  );

  VoiceParakeetCheckBox := TNewCheckBox.Create(ModelsPage);
  VoiceParakeetCheckBox.Parent := ModelsPage.Surface;
  VoiceParakeetCheckBox.Left := 0;
  VoiceParakeetCheckBox.Top := 8;
  VoiceParakeetCheckBox.Width := ModelsPage.SurfaceWidth;
  VoiceParakeetCheckBox.Caption := 'Voice: NVIDIA Parakeet (default)';
  VoiceParakeetCheckBox.Checked := True;

  VoiceCanaryCheckBox := TNewCheckBox.Create(ModelsPage);
  VoiceCanaryCheckBox.Parent := ModelsPage.Surface;
  VoiceCanaryCheckBox.Left := 0;
  VoiceCanaryCheckBox.Top := VoiceParakeetCheckBox.Top + ScaleY(24);
  VoiceCanaryCheckBox.Width := ModelsPage.SurfaceWidth;
  VoiceCanaryCheckBox.Caption := 'Voice: NVIDIA Canary-Qwen 2.5B';
  VoiceCanaryCheckBox.Checked := False;

  Notes4BCheckBox := TNewCheckBox.Create(ModelsPage);
  Notes4BCheckBox.Parent := ModelsPage.Surface;
  Notes4BCheckBox.Left := 0;
  Notes4BCheckBox.Top := VoiceCanaryCheckBox.Top + ScaleY(32);
  Notes4BCheckBox.Width := ModelsPage.SurfaceWidth;
  Notes4BCheckBox.Caption := 'Notes: Qwen 3.5 4B GGUF (default)';
  Notes4BCheckBox.Checked := True;

  Notes2BCheckBox := TNewCheckBox.Create(ModelsPage);
  Notes2BCheckBox.Parent := ModelsPage.Surface;
  Notes2BCheckBox.Left := 0;
  Notes2BCheckBox.Top := Notes4BCheckBox.Top + ScaleY(24);
  Notes2BCheckBox.Width := ModelsPage.SurfaceWidth;
  Notes2BCheckBox.Caption := 'Notes: Qwen 3.5 2B GGUF';
  Notes2BCheckBox.Checked := False;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    if not InstallSelectedModels() then
      Abort();
  end;
end;
