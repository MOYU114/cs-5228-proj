# Runs the three steps in order and stops at the first failure:
# training stage 1 (pretraining), training stage 2 (fine-tuning), test-set prediction.
$python = "$PSScriptRoot/.venv/Scripts/python.exe"
& $python -u "$PSScriptRoot/scripts/train_stage1_pretrain.py"; if ($LASTEXITCODE) { exit $LASTEXITCODE }
& $python -u "$PSScriptRoot/scripts/train_stage2_finetune.py"; if ($LASTEXITCODE) { exit $LASTEXITCODE }
& $python "$PSScriptRoot/tests/test.py"; exit $LASTEXITCODE
