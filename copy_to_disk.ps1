param(
    [Parameter(Mandatory=$true)][string]$DiskRoot,
    [string]$PythonExe = 'python',
    [switch]$AllowIncomplete,
    [switch]$HardwareOnly
)
$ErrorActionPreference = 'Stop'
$taskSource = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$taskDisk = (Resolve-Path -LiteralPath $DiskRoot).Path
if (-not (Test-Path -LiteralPath $taskDisk -PathType Container)) { throw 'Disk root is not a directory.' }
$taskVolumeRoot = [IO.Path]::GetPathRoot($taskDisk)
if ($taskDisk.TrimEnd('\','/') -ne $taskVolumeRoot.TrimEnd('\','/')) { throw 'Use an explicit drive root, such as E:\.' }
if ($taskVolumeRoot -eq [IO.Path]::GetPathRoot($taskSource)) { throw 'Destination must be a different drive from the source.' }
if (-not $AllowIncomplete) {
    $taskVerifyOptions = @('verify')
    if ($HardwareOnly) { $taskVerifyOptions += '--hardware-only' }
    & $PythonExe (Join-Path $taskSource 'bundle.py') @taskVerifyOptions
    if ($LASTEXITCODE -ne 0) { throw 'Complete and seal the offline bundle first; see README.' }
} else {
    Write-Warning 'Copying an incomplete preparation folder. It is not ready for an offline server.'
}
$taskFiles = @(Get-ChildItem -LiteralPath $taskSource -Recurse -File -Force | Where-Object {
    $_.Name -notmatch '\.(part|tmp|pyc)$' -and $_.FullName -notmatch '[\\/](\.git|\.github|__pycache__|tests|b300-results|demo-report|public-reference-preview)[\\/]' -and
    (-not $HardwareOnly -or ($_.FullName -notmatch '[\\/]models[\\/]' -and $_.Name -ne 'bundle.sha256.json'))
})
$taskBytes = ($taskFiles | Measure-Object -Property Length -Sum).Sum
$taskDrive = Get-PSDrive -Name $taskVolumeRoot.Substring(0,1) -PSProvider FileSystem
if ($taskDrive.Free -lt ($taskBytes + 2GB)) { throw 'Insufficient free space (package + 2GB reserve).' }
if (Get-Command Get-Volume -ErrorAction SilentlyContinue) {
    $taskVolume = Get-Volume -DriveLetter $taskVolumeRoot.Substring(0,1) -ErrorAction SilentlyContinue
    if ($taskVolume.FileSystem -eq 'FAT32') { throw 'FAT32 cannot carry the large weight/archive files. Use an existing exFAT, NTFS, or Linux-compatible disk.' }
}
$taskDestination = Join-Path $taskDisk ('b300-usb-kit-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
if (Test-Path -LiteralPath $taskDestination) { throw 'Destination already exists.' }
New-Item -ItemType Directory -Path $taskDestination | Out-Null
foreach ($taskFile in $taskFiles) {
    $taskRelative = $taskFile.FullName.Substring($taskSource.Length + 1)
    $taskTarget = [IO.Path]::GetFullPath((Join-Path $taskDestination $taskRelative))
    if (-not $taskTarget.StartsWith($taskDestination + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Destination path escaped package directory.' }
    New-Item -ItemType Directory -Path ([IO.Path]::GetDirectoryName($taskTarget)) -Force | Out-Null
    Copy-Item -LiteralPath $taskFile.FullName -Destination $taskTarget
    if ((Get-Item -LiteralPath $taskTarget).Length -ne $taskFile.Length) { throw ('Copy size mismatch: ' + $taskRelative) }
}
if (-not $AllowIncomplete) {
    & $PythonExe (Join-Path $taskDestination 'bundle.py') @taskVerifyOptions
    if ($LASTEXITCODE -ne 0) { throw 'Destination verification failed; do not use this copy.' }
} else {
    & $PythonExe (Join-Path $taskDestination 'download_models.py') --verify
    if ($LASTEXITCODE -ne 0) { Write-Warning 'Model files are still missing/corrupt in this preparation copy.' }
}
Write-Host ('Copied to: ' + $taskDestination)
if ($AllowIncomplete) { Write-Warning 'Runtime image still needs to be built/exported on an online Linux/Docker host.' }
