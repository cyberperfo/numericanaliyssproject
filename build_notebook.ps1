# build_notebook.ps1
# .py dosyalarını MAT353_Proje.ipynb içine %%writefile hücreleri olarak gömer.
# Colab'a yalnızca .ipynb yüklenir; hücreler çalışınca .py dosyaları oluşur.
# Herhangi bir .py dosyası değiştiğinde yeniden çalıştırın (eski gömülü kopyaları değiştirir):
#     powershell -ExecutionPolicy Bypass -File build_notebook.ps1

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$nbPath = Join-Path $root 'MAT353_Proje.ipynb'
$utf8 = New-Object System.Text.UTF8Encoding($false)

$modules = @('models.py', 'optimizers.py', 'gradient_check.py', 'hessian.py',
             'experiments.py', 'plot_results.py', 'tests.py')
$header = '## 0. Proje dosyaları'
$anchor = '=== Çalışma klasörü ==='

function Get-Src($cell) {
    if ($cell.source -is [array]) { return ($cell.source -join '') }
    return [string]$cell.source
}

$nb = [System.IO.File]::ReadAllText($nbPath, $utf8) | ConvertFrom-Json

# Önceki gömülü kopyaları ve başlığı kaldır
$kept = @($nb.cells | Where-Object {
    $s = Get-Src $_
    -not ($s.StartsWith('%%writefile') -or $s.StartsWith($header))
})

$new = New-Object System.Collections.ArrayList
$inserted = $false
foreach ($cell in $kept) {
    [void]$new.Add($cell)
    if (-not $inserted -and (Get-Src $cell).Contains($anchor)) {
        [void]$new.Add([pscustomobject]@{
            cell_type = 'markdown'; metadata = @{}
            source = "$header`nAşağıdaki hücreler projenin Python modüllerini çalışma klasörüne yazar. " +
                     "Kodu incelemek için hücreleri açabilirsiniz."
        })
        foreach ($m in $modules) {
            $code = [System.IO.File]::ReadAllText((Join-Path $root $m), $utf8).Replace("`r`n", "`n")
            [void]$new.Add([pscustomobject]@{
                cell_type = 'code'; metadata = @{}; execution_count = $null; outputs = @()
                source = "%%writefile $m`n$code"
            })
        }
        $inserted = $true
    }
}
if (-not $inserted) { throw "Çapa hücresi bulunamadı: '$anchor'" }

$nb.cells = $new.ToArray()
$json = $nb | ConvertTo-Json -Depth 20
[System.IO.File]::WriteAllText($nbPath, $json, $utf8)
Write-Output "Gömülen modüller: $($modules -join ', ')  ->  $nbPath ($($new.Count) hücre)"
