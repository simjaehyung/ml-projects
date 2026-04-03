$ErrorActionPreference = 'Stop'

$root = "c:\Users\jhsim\Erica261\M.L\projects\04_canon_project\outputs\experiment2\artifacts"
$outDir = Join-Path $root "labeling"
New-Item -ItemType Directory -Path $outDir -Force | Out-Null
$csv = Import-Csv (Join-Path $root "eval_3fps_predictions.csv")

$neg = $csv | Where-Object { [int]$_.gate_pred -eq 0 }
$coreAmbiguous = $neg |
    Where-Object { [double]$_.gate_prob_target -ge 0.005 -and [double]$_.gate_prob_target -lt 0.5 } |
    Sort-Object {[double]$_.gate_prob_target} -Descending
$hardNegTop = $coreAmbiguous | Select-Object -First 120

$indexMap = @{}
foreach ($r in $csv) {
    if ($r.path -match '2_frame_(\d+)\.jpg') {
        $fid = [int]$Matches[1]
        $indexMap[$fid] = $r
    }
}

$selected = @{}
foreach ($r in $hardNegTop) {
    if ($r.path -match '2_frame_(\d+)\.jpg') {
        $fid = [int]$Matches[1]
        foreach ($k in ($fid - 2)..($fid + 2)) {
            if ($indexMap.ContainsKey($k)) {
                $selected[$k] = $indexMap[$k]
            }
        }
    }
}

# Ensure known miss neighborhood is included.
foreach ($k in 1284..1294) {
    if ($indexMap.ContainsKey($k)) {
        $selected[$k] = $indexMap[$k]
    }
}

function Build-Rows($items, $namePrefix) {
    $rows = @()
    $i = 1
    $items.GetEnumerator() | Sort-Object Name | ForEach-Object {
        $r = $_.Value
        $fid = if ($r.path -match '2_frame_(\d+)\.jpg') { [int]$Matches[1] } else { -1 }
        $p = [double]$r.gate_prob_target

        $priority = if ($p -ge 0.10) { "P0" } elseif ($p -ge 0.03) { "P1" } elseif ($p -ge 0.01) { "P2" } else { "P3" }
        $reason = if (($fid -ge 1284) -and ($fid -le 1294)) {
            "miss-neighborhood"
        } elseif ($p -ge 0.03) {
            "hard-negative-high-prob"
        } elseif ($p -ge 0.01) {
            "hard-negative-mid-prob"
        } else {
            "temporal-neighbor"
        }

        $rows += [pscustomobject]@{
            sample_id         = ("{0}{1:D4}" -f $namePrefix, $i)
            frame_id          = $fid
            path              = $r.path
            gate_prob_target  = [math]::Round($p, 6)
            current_gate_pred = [int]$r.gate_pred
            current_cls_pred  = $r.cls_pred
            current_cls_conf  = $r.cls_conf
            priority          = $priority
            reason            = $reason
            ox_target         = ""
            class_if_target   = ""
            note              = ""
        }
        $i++
    }
    return $rows
}

$core = @{}
foreach ($r in $coreAmbiguous) {
    if ($r.path -match '2_frame_(\d+)\.jpg') {
        $fid = [int]$Matches[1]
        $core[$fid] = $r
    }
}
foreach ($k in 1284..1294) {
    if ($indexMap.ContainsKey($k)) {
        $core[$k] = $indexMap[$k]
    }
}

$rows = Build-Rows -items $selected -namePrefix "A"
$rowsCore = Build-Rows -items $core -namePrefix "C"

$labelCsv = Join-Path $outDir "ambiguous_candidates_for_ox_labeling.csv"
$rows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $labelCsv
$labelCsvCore = Join-Path $outDir "ambiguous_candidates_core_for_ox_labeling.csv"
$rowsCore | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $labelCsvCore

$guide = @"
ox_target input:
- O = target
- X = non_target

class_if_target input (only when ox_target=O):
- t1 / t2 / t3 / t4

files:
- ambiguous_candidates_core_for_ox_labeling.csv : strict ambiguous only
- ambiguous_candidates_for_ox_labeling.csv : ambiguous + temporal neighbors
"@
Set-Content -Path (Join-Path $outDir "labeling_guide.txt") -Value $guide -Encoding UTF8

Write-Output "created=$labelCsv"
Write-Output "created=$labelCsvCore"
Write-Output "core_count=$($rowsCore.Count)"
Write-Output "count=$($rows.Count)"
