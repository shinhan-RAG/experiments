param(
    [ValidateSet("preflight", "smoke", "full")][string]$Stage = "preflight",
    [string]$PythonBin = "python",
    [string]$CodexBin = "$env:APPDATA\npm\codex.cmd",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $here
try {
    if ($Stage -eq "preflight") {
        & $PythonBin preflight.py --codex-bin $CodexBin
        exit $LASTEXITCODE
    }
    & $PythonBin preflight.py --codex-bin $CodexBin --skip-tool-smoke
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $runnerArgs = @("agent_runner.py", "--workers", "4", "--codex-bin", $CodexBin)
    if ($Stage -eq "smoke") {
        $runnerArgs += @("--n", "4", "--run", "c29_dual_tool_smoke4_luna_medium_w4")
    } else {
        $runnerArgs += @("--n", "-1", "--run", "c29_dual_tool_train281_luna_medium_w4")
    }
    if ($Resume) { $runnerArgs += "--resume" }
    & $PythonBin @runnerArgs
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
