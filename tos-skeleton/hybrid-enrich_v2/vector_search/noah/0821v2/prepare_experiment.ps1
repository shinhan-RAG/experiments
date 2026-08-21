param()
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $here
try {
    python build_gold_v5.py
    python prepare_gold.py
    python verify_frozen_meta.py --manifest-only
    python -m unittest -v test_tag_hybrid.py test_v5_pipeline.py
} finally {
    Pop-Location
}
