param(
    [ValidateSet("smoke", "train60", "trainfull", "test")][string]$Stage = "smoke",
    [string]$ClaudeBin = "claude",
    [string]$PythonBin = "python",
    [string]$Model = "sonnet",
    [int]$Workers = 3,
    [bool]$RunBaseline = $true
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = (Resolve-Path (Join-Path $here "..\..\..")).Path
$selectedFile = Join-Path $here "out\det\selected_arm.txt"
if (-not (Test-Path -LiteralPath $selectedFile)) { throw "eval_det.py를 먼저 실행하십시오: $selectedFile" }
$detSummary = Get-Content -LiteralPath (Join-Path $here "out\det\summary.json") -Raw -Encoding UTF8 | ConvertFrom-Json
if (-not $detSummary.full_train_screen) { throw "5개 arm train337 전수 선발이 완료되지 않았습니다. prepare_experiment.ps1을 실행하십시오." }
$candidate = (Get-Content -LiteralPath $selectedFile -Encoding UTF8).Trim()
$trainGold = Join-Path $root "out\noah\gold_v4_train_full.jsonl"
$gold = $trainGold
$n = -1
if ($Stage -eq "smoke") { $n = 5 }
if ($Stage -eq "train60") { $n = 60 }
if ($Stage -eq "test") {
    $manifest = Get-Content -LiteralPath (Join-Path $here "out\gold_manifest.json") -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($manifest.optional_test138) { $gold = $manifest.optional_test138.path } else { $gold = $manifest.default_test90.path }
}
$common = @("agent_runner.py", "--gold", $gold, "--n", "$n", "--reps", "1", "--workers", "$Workers", "--model", $Model, "--claude-bin", $ClaudeBin, "--pybin", $PythonBin, "--resume")
Push-Location $here
try {
    if ($RunBaseline) {
        & $PythonBin @common --run "0821_${Stage}_baseline" --arm baseline_fs_slot
    }
    & $PythonBin @common --run "0821_${Stage}_candidate" --arm $candidate
    if ($RunBaseline) {
        & $PythonBin compare_runs.py --baseline "out\agent\0821_${Stage}_baseline\results.jsonl" --candidate "out\agent\0821_${Stage}_candidate\results.jsonl" --out "out\${Stage}_comparison.json"
    }
} finally {
    Pop-Location
}
