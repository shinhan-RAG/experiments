param(
    [ValidateSet("smoke", "full")][string]$Stage = "smoke",
    [string]$PythonBin = "python",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $here
try {
    $runnerArgs = @("agent_runner.py", "--workers", "2")
    if ($Stage -eq "smoke") {
        $runnerArgs += @("--n", "4", "--run", "c29_dual_tool_smoke4_sonnet5_w2")
    } else {
        $runnerArgs += @("--n", "-1", "--run", "c29_dual_tool_clean257_sonnet5_w2")
    }
    if ($Resume) { $runnerArgs += "--resume" }
    & $PythonBin @runnerArgs
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
