#ifndef SourceDir
  #define SourceDir "dist\\windows\\transcribe"
#endif

#ifndef OutputDir
  #define OutputDir "dist\\windows\\installer"
#endif

#define AppName "Transcribe"
#define AppVersion "0.1.0"
#define AppPublisher "Transcribe"

[Setup]
AppId={{2D9D88E0-12D3-4B3A-A1E5-15C84CAAB4A3}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
UninstallDisplayIcon={app}\transcribe.exe
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
OutputDir={#OutputDir}
OutputBaseFilename=transcribe-windows-standalone
DiskSpanning=yes
DiskSliceSize=max

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\Transcribe"; Filename: "{app}\transcribe.exe"
Name: "{group}\Uninstall Transcribe"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\transcribe.exe"; Description: "Launch Transcribe"; Flags: nowait postinstall skipifsilent
