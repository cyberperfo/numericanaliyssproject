param([string]$Docx, [string]$OutPrefix, [int]$Chunk = 9)
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$doc = $null
try {
  $doc = $word.Documents.Open($Docx, $false, $true)
  $n = $doc.ComputeStatistics(2)
  for ($a = 1; $a -le $n; $a += $Chunk) {
    $b = [Math]::Min($n, $a + $Chunk - 1)
    $f = "{0}_{1:00}-{2:00}.pdf" -f $OutPrefix, $a, $b
    # ExportAsFixedFormat(yol, PDF=17, açma=hayır, optimize=0, aralık=wdExportFromTo(3), başlangıç, bitiş)
    $doc.ExportAsFixedFormat($f, 17, $false, 0, 3, $a, $b)
    $f
  }
}
finally {
  if ($doc) { $doc.Close(0); [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($doc) }
  $word.Quit()
  [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($word)
  [GC]::Collect(); [GC]::WaitForPendingFinalizers()
}
