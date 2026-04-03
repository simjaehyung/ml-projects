$ErrorActionPreference = 'Stop'
$python = 'C:/Users/jhsim/miniconda3/python.exe'
$root = Split-Path -Parent $PSScriptRoot
$entry = Join-Path $root 'src/canon_program/main.py'
$config = Join-Path $root 'configs/baseline_program.json'

& $python $entry --config $config
