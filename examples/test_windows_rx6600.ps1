param(
    [string]$HipLibrary,
    [string]$Clang,
    [string]$Python
)

$ErrorActionPreference = "Stop"
if ($env:OS -ne "Windows_NT") { throw "Run this test in Windows." }

$directory = $PSScriptRoot
$env:OSC_HIP_DIAGNOSTICS = "1"
$env:OSC_GPU_ARCH = "gfx1032"
$env:OSC_PTX_TOOL = Join-Path $directory "osc_ptx.py"
$env:OSC_BACKEND = $null

if ($HipLibrary) {
    $dll = Get-Item -LiteralPath $HipLibrary
    $env:OSC_HIP_LIBRARY = $dll.FullName
    Write-Output "HIP DLL: $($dll.FullName) [$($dll.VersionInfo.FileVersion)]"
}
if ($Clang) { $env:OSC_CLANG = (Get-Item -LiteralPath $Clang).FullName }
if ($Python) { $env:OSC_PYTHON = (Get-Item -LiteralPath $Python).FullName }

Write-Output "Windows: $([Environment]::OSVersion.VersionString)"
if (Get-Command Get-CimInstance -ErrorAction SilentlyContinue) {
    Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
        Select-Object Name, DriverVersion | Format-Table | Out-String | Write-Output
}

$script:failed = $false
function Invoke-Phase([string]$Title, [string]$Name, [string[]]$Arguments = @()) {
    $ErrorActionPreference = "Continue"
    Write-Output ""
    Write-Output "=== $Title ==="
    & (Join-Path $directory $Name) @Arguments 2>&1 | ForEach-Object { "$_" }
    $code = $LASTEXITCODE
    Write-Output "Exit code: $code"
    if ($code -ne 0) { $script:failed = $true }
}

Invoke-Phase "CUDA registration (host only)" "cuda_registration_test.exe"
Invoke-Phase "HIP device and memory" "amd_runtime_test.exe"
$modulePath = Join-Path $directory "vector_add_gfx1032.hsaco"
Invoke-Phase "Precompiled gfx1032 module (no Python or Clang)" "amd_vector_add.exe" @($modulePath)
Invoke-Phase "CUDA executable with embedded fatbin" "cuda_vector_add.exe"
if ($script:failed) { exit 1 }
