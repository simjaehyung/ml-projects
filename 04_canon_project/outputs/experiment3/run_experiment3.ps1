$ErrorActionPreference = 'Stop'
$python = 'C:/Users/jhsim/miniconda3/python.exe'
$script = Join-Path $PSScriptRoot 'src/run_experiment3.py'

& $python $script --epochs 3 --batch-size 32 --gate-per-class 1000 --classify-per-class 400 --train-ratio 0.6 --gate-backbone efficientnet_b0
