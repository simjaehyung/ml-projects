$ErrorActionPreference = 'Stop'
$python = 'C:/Users/jhsim/miniconda3/python.exe'
$script = Join-Path $PSScriptRoot 'src/run_experiment2.py'

& $python $script --epochs 3 --batch-size 32 --gate-per-class 1000 --classify-per-class 400
