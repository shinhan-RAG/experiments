param()
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $here
try {
    python prepare_gold.py
    python verify_frozen_meta.py --manifest-only
    python -m unittest -v test_tag_hybrid.py
    python build_tag_index.py
    python eval_det.py
} finally {
    Pop-Location
}
