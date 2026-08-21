param(
    [ValidateSet("smoke", "train60", "trainfull", "test")][string]$Stage = "smoke",
    [string]$CodexBin = "$env:APPDATA\npm\codex.cmd", [string]$PythonBin = "python", [string]$Model = "gpt-5.6-luna",
    [int]$Workers = 2, [bool]$RunBaseline = $true
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = (Resolve-Path (Join-Path $here "..\..\..")).Path
$selectedFile = Join-Path $here "out\det\selected_arm.txt"
if (-not (Test-Path -LiteralPath $selectedFile)) { throw "먼저 .\prepare_experiment.ps1 을 실행하세요: $selectedFile" }
$det = Get-Content (Join-Path $here "out\det\summary.json") -Raw -Encoding UTF8 | ConvertFrom-Json
if (-not $det.full_train_screen) { throw "train337 arm screening이 완료되지 않았습니다." }
$candidate = (Get-Content $selectedFile -Encoding UTF8).Trim(); $gold = Join-Path $root "out\noah\gold_v4_train_full.jsonl"; $n = -1
if ($Stage -eq "smoke") { $n = 5 }; if ($Stage -eq "train60") { $n = 60 }
if ($Stage -eq "test") { $m = Get-Content (Join-Path $here "out\gold_manifest.json") -Raw -Encoding UTF8 | ConvertFrom-Json; $gold = if ($m.optional_test138) { $m.optional_test138.path } else { $m.default_test90.path } }
$common = @("agent_runner.py", "--gold", $gold, "--n", "$n", "--workers", "$Workers", "--model", $Model, "--codex-bin", $CodexBin, "--resume")
Push-Location $here
try {
    if ($RunBaseline) { & $PythonBin @common --run "0821v2_${Stage}_baseline" --arm baseline_fs_slot }
    & $PythonBin @common --run "0821v2_${Stage}_candidate" --arm $candidate
    if ($RunBaseline) { & $PythonBin compare_runs.py --baseline "out\agent\0821v2_${Stage}_baseline\results.jsonl" --candidate "out\agent\0821v2_${Stage}_candidate\results.jsonl" --out "out\${Stage}_comparison.json" }
} finally { Pop-Location }
