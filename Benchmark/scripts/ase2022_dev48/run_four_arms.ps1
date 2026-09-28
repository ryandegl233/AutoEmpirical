param(
    [ValidateSet('All', 'S0', 'A', 'B', 'AB')][string]$Arm = 'All',
    [switch]$Full,
    [switch]$DryRun,
    [switch]$Run,
    [string]$Python = 'python',
    [string]$Model = 'gemini-3.7-flash',
    [string]$BatchId = '',
    [string]$RunId = ''
)
$ErrorActionPreference = 'Stop'
if ($Run -and $DryRun) { throw 'Choose -Run or -DryRun.' }
$DryRun = -not $Run
$repoPath = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$inputPath = Join-Path $repoPath 'Benchmark/inputs/ase2022_dev48'
$aPath = Join-Path $inputPath 'examples'
$outputPath = Join-Path $repoPath 'Benchmark/runs/ase2022_dev48'
$cohortPath = Join-Path $inputPath 'runtime'
$pythonPath = $Python
if (-not $RunId) { $RunId = if ($Full) { 'full48-v2' } else { 'pilot3-v2' } }
if (-not $BatchId) { $BatchId = Get-Date -Format 'yyyyMMdd-HHmmss-fff' }
if ($BatchId -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') { throw 'BatchId must use letters, digits, dot, underscore or hyphen.' }
$scope = if ($Full) { 'full48' } else { 'pilot3' }
$mode = if ($DryRun) { 'check' } else { 'live' }
$experimentId = "formal-ab-v2-$BatchId-$scope-$mode"
Write-Host "BatchId: $BatchId (reuse with -BatchId to resume unchanged inputs/code)"
Write-Host "Results: $(Join-Path $outputPath "$experimentId/evaluation48")"
$arms = if ($Arm -eq 'All') { @('S0', 'A', 'B', 'AB') } else { @($Arm) }
Push-Location $repoPath
try {
    foreach ($currentArm in $arms) {
        Write-Host "Running arm: $currentArm ($scope)"
        $runArgs = @('-B', 'Benchmark/scripts/run_adaptive_empirical_workflow.py',
            '--domain', 'ase2022', '--stage', 'stage3', '--provider', 'gemini', '--model', $Model,
            '--cohort-path', (Join-Path $cohortPath 'evaluation48.csv'),
            '--split-manifest', (Join-Path $cohortPath 'split_manifest.json'),
            '--output-dir', $outputPath, '--experiment-id', $experimentId,
            '--arm-id', $currentArm, '--run-id', $RunId, '--concurrency', '2', '--max-network-retries', '2',
            '--thinking-profile', 'label-thinking')
        if (-not $Full) { $runArgs += @('--limit', '3') }
        if ($DryRun) { $runArgs += '--dry-run' }
        if ($currentArm -in @('A', 'AB')) {
            $runArgs += @('--module-a', '--a-protocol', 'formal-v2',
                '--a-examples', (Join-Path $aPath 'examples_formal_v2.json'),
                '--a-example-source', (Join-Path $aPath 'example_source.csv'))
        }
        if ($currentArm -in @('B', 'AB')) { $runArgs += '--module-b' }
        & $pythonPath @runArgs
        if ($LASTEXITCODE -ne 0) { throw "Arm $currentArm failed with exit code $LASTEXITCODE" }
    }
} finally { Pop-Location }
