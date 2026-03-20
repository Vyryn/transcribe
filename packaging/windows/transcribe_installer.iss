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
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[UninstallDelete]
Type: filesandordirs; Name: "{app}"

[Icons]
Name: "{group}\Transcribe"; Filename: "{app}\Transcribe.exe"; WorkingDir: "{app}"; IconFilename: "{app}\Transcribe.exe"; IconIndex: 0; Tasks: startmenuicon
Name: "{commondesktop}\Transcribe"; Filename: "{app}\Transcribe.exe"; WorkingDir: "{app}"; IconFilename: "{app}\Transcribe.exe"; IconIndex: 0; Tasks: desktopicon

[Code]
var
  ModelsPage: TWizardPage;
  VoiceParakeetCheckBox: TNewCheckBox;
  VoiceCanaryCheckBox: TNewCheckBox;
  VoiceGraniteCheckBox: TNewCheckBox;
  Notes4BCheckBox: TNewCheckBox;
  Notes2BCheckBox: TNewCheckBox;
  ModelInstallProgressPage: TOutputProgressWizardPage;

function SetEnvironmentVariableW(lpName, lpValue: string): Boolean;
  external 'SetEnvironmentVariableW@kernel32.dll stdcall';

procedure ConfigureModelCheckBox(
  CheckBox: TNewCheckBox;
  const Caption: String;
  const Top: Integer;
  const Checked: Boolean
);
begin
  CheckBox.Parent := ModelsPage.Surface;
  CheckBox.Left := 0;
  CheckBox.Top := Top;
  CheckBox.Width := ModelsPage.SurfaceWidth;
  CheckBox.Height := ScaleY(28);
  CheckBox.Caption := Caption;
  CheckBox.Checked := Checked;
end;

function NextModelCheckBoxTop(PreviousCheckBox: TNewCheckBox): Integer;
begin
  Result := PreviousCheckBox.Top + PreviousCheckBox.Height + ScaleY(8);
end;

function CountSelectedModels(): Integer;
begin
  Result := 0;
  if VoiceParakeetCheckBox.Checked then
    Result := Result + 1;
  if VoiceCanaryCheckBox.Checked then
    Result := Result + 1;
  if VoiceGraniteCheckBox.Checked then
    Result := Result + 1;
  if Notes4BCheckBox.Checked then
    Result := Result + 1;
  if Notes2BCheckBox.Checked then
    Result := Result + 1;
end;

procedure UpdateModelInstallProgress(const CurrentIndex, TotalCount: Integer; const ModelLabel: String);
begin
  ModelInstallProgressPage.SetText(
    'Downloading the selected transcription and notes models.',
    'Installing model ' + IntToStr(CurrentIndex) + ' of ' + IntToStr(TotalCount) + ': ' + ModelLabel
  );
  ModelInstallProgressPage.SetProgress(CurrentIndex - 1, TotalCount);
  WizardForm.StatusLabel.Caption :=
    'Installing model ' + IntToStr(CurrentIndex) + ' of ' + IntToStr(TotalCount) + ': ' + ModelLabel;
end;

function InstallSelectedModel(
  const ModelId, ModelLabel: String;
  const CurrentIndex, TotalCount: Integer
): Boolean;
var
  ResultCode: Integer;
begin
  Result := True;
  UpdateModelInstallProgress(CurrentIndex, TotalCount, ModelLabel);

  if not Exec(
    ExpandConstant('{app}\Transcribe.exe'),
    'models install --quiet --model ' + ModelId,
    ExpandConstant('{app}'),
    SW_HIDE,
    ewWaitUntilTerminated,
    ResultCode
  ) then
  begin
    MsgBox(
      'The installer could not launch the bundled model downloader for ' + ModelLabel + '. ' +
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
      'The installer could not download ' + ModelLabel + '. ' +
      'You can retry later from the installed application.',
      mbError,
      MB_OK
    );
    Result := False;
    Exit;
  end;

  ModelInstallProgressPage.SetProgress(CurrentIndex, TotalCount);
end;

function InstallSelectedModels(): Boolean;
var
  OriginalNetworkValue: String;
  CurrentIndex: Integer;
  TotalCount: Integer;
begin
  Result := True;
  TotalCount := CountSelectedModels();
  if TotalCount = 0 then
    Exit;

  OriginalNetworkValue := GetEnv('TRANSCRIBE_ALLOW_NETWORK');
  WizardForm.StatusLabel.Caption :=
    'Downloading the selected transcription and notes models. This can take several minutes.';
  ModelInstallProgressPage.SetText(
    'Downloading the selected transcription and notes models.',
    'Preparing model downloads...'
  );
  ModelInstallProgressPage.SetProgress(0, TotalCount);
  ModelInstallProgressPage.Show;
  SetEnvironmentVariableW('TRANSCRIBE_ALLOW_NETWORK', '1');

  try
    CurrentIndex := 0;

    if VoiceParakeetCheckBox.Checked then
    begin
      CurrentIndex := CurrentIndex + 1;
      if not InstallSelectedModel(
        'nvidia/parakeet-tdt-0.6b-v3',
        'Voice: NVIDIA Parakeet',
        CurrentIndex,
        TotalCount
      ) then
      begin
        Result := False;
        Exit;
      end;
    end;

    if VoiceCanaryCheckBox.Checked then
    begin
      CurrentIndex := CurrentIndex + 1;
      if not InstallSelectedModel(
        'nvidia/canary-qwen-2.5b',
        'Voice: NVIDIA Canary-Qwen 2.5B',
        CurrentIndex,
        TotalCount
      ) then
      begin
        Result := False;
        Exit;
      end;
    end;

    if VoiceGraniteCheckBox.Checked then
    begin
      CurrentIndex := CurrentIndex + 1;
      if not InstallSelectedModel(
        'ibm-granite/granite-4.0-1b-speech',
        'Voice: IBM Granite 4.0 1B Speech',
        CurrentIndex,
        TotalCount
      ) then
      begin
        Result := False;
        Exit;
      end;
    end;

    if Notes4BCheckBox.Checked then
    begin
      CurrentIndex := CurrentIndex + 1;
      if not InstallSelectedModel(
        'qwen3.5:4b-q4_K_M',
        'Notes: Qwen 3.5 4B GGUF',
        CurrentIndex,
        TotalCount
      ) then
      begin
        Result := False;
        Exit;
      end;
    end;

    if Notes2BCheckBox.Checked then
    begin
      CurrentIndex := CurrentIndex + 1;
      if not InstallSelectedModel(
        'qwen3.5:2b-q4_K_M',
        'Notes: Qwen 3.5 2B GGUF',
        CurrentIndex,
        TotalCount
      ) then
      begin
        Result := False;
        Exit;
      end;
    end;
  finally
    SetEnvironmentVariableW('TRANSCRIBE_ALLOW_NETWORK', OriginalNetworkValue);
    ModelInstallProgressPage.Hide;
  end;
end;

procedure InitializeWizard();
begin
  ModelInstallProgressPage := CreateOutputProgressPage(
    'Installing models',
    'Downloading the selected transcription and notes models.'
  );

  ModelsPage := CreateCustomPage(
    wpSelectTasks,
    'Model Downloads',
    'Choose which transcription and notes models to download during setup.'
  );

  VoiceParakeetCheckBox := TNewCheckBox.Create(ModelsPage);
  ConfigureModelCheckBox(
    VoiceParakeetCheckBox,
    'Voice: NVIDIA Parakeet (default)',
    ScaleY(8),
    True
  );

  VoiceCanaryCheckBox := TNewCheckBox.Create(ModelsPage);
  ConfigureModelCheckBox(
    VoiceCanaryCheckBox,
    'Voice: NVIDIA Canary-Qwen 2.5B',
    NextModelCheckBoxTop(VoiceParakeetCheckBox),
    False
  );

  VoiceGraniteCheckBox := TNewCheckBox.Create(ModelsPage);
  ConfigureModelCheckBox(
    VoiceGraniteCheckBox,
    'Voice: IBM Granite 4.0 1B Speech',
    NextModelCheckBoxTop(VoiceCanaryCheckBox),
    False
  );

  Notes4BCheckBox := TNewCheckBox.Create(ModelsPage);
  ConfigureModelCheckBox(
    Notes4BCheckBox,
    'Notes: Qwen 3.5 4B GGUF (default)',
    NextModelCheckBoxTop(VoiceGraniteCheckBox),
    True
  );

  Notes2BCheckBox := TNewCheckBox.Create(ModelsPage);
  ConfigureModelCheckBox(
    Notes2BCheckBox,
    'Notes: Qwen 3.5 2B GGUF',
    NextModelCheckBoxTop(Notes4BCheckBox),
    False
  );
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    if not InstallSelectedModels() then
      Abort();
  end;
end;
