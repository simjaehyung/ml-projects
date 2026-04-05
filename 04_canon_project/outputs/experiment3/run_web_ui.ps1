$ErrorActionPreference = 'Stop'
$python = 'C:/Users/jhsim/miniconda3/python.exe'
$uiScript = Join-Path $PSScriptRoot 'src/web_ui_experiment3.py'

& $python -m streamlit run $uiScript
