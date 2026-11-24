param(
  [Parameter(Mandatory)] [string]$Content,
  [Parameter(Mandatory)] [string]$OutDocx,
  [Parameter(Mandatory)] [string]$ImgDir
)
# Satır tabanlı içerik dosyasından Word (.docx) raporu üretir (Word COM otomasyonu).
# Biçim komutları: H1, H2, P, B, EQ, FIG, TAB..ENDTAB, CODE..ENDCODE, CTR, SP, PB, REFS, REFDEF
$ErrorActionPreference = 'Stop'
$utf8 = New-Object System.Text.UTF8Encoding($false)
$lines = [IO.File]::ReadAllLines($Content, $utf8)

# ---------------- Kaynak numaralandırma (ilk geçiş sırasına göre, IEEE) ----------------
$refText = [ordered]@{}
foreach ($ln in $lines) { if ($ln.StartsWith('REFDEF ')) { $k, $v = $ln.Substring(7).Split('|', 2); $refText[$k.Trim()] = $v.Trim() } }
$refNum = [ordered]@{}
$citeRx = [regex]'\[@([^\]]+)\]'
$body = New-Object System.Collections.Generic.List[string]
foreach ($ln in $lines) {
  if ($ln.StartsWith('REFDEF ')) { continue }
  $ln2 = $citeRx.Replace($ln, {
      param($m)
      $nums = foreach ($key in $m.Groups[1].Value.Split(';')) {
        $key = $key.Trim().TrimStart('@')
        if (-not $refText.Contains($key)) { throw "Tanımsız kaynak: $key" }
        if (-not $refNum.Contains($key)) { $refNum[$key] = $refNum.Count + 1 }
        $refNum[$key]
      }
      ($nums | ForEach-Object { "[$_]" }) -join ', '
    })
  $body.Add($ln2)
}

$ZCH = [char]0x2063            # INVISIBLE SEPARATOR
$PAD = ([string]$ZCH) * 24
$EM = [string][char]0x2003     # em boşluk (denklem içi ayraç)
$script:spaceNext = $null      # tablo/kod sonrası paragrafa üst boşluk

# ---------------- Satır içi biçim ayrıştırıcı: **kalın**, __italik__, $matematik$ ----------------
function Parse-Inline([string]$s) {
  $sb = New-Object System.Text.StringBuilder
  $spans = New-Object System.Collections.ArrayList
  $i = 0
  while ($i -lt $s.Length) {
    $two = if ($i + 1 -lt $s.Length) { $s.Substring($i, 2) } else { '' }
    if ($two -eq '**' -or $two -eq '__') {
      $j = $s.IndexOf($two, $i + 2); if ($j -lt 0) { throw "Kapanmayan $two : $s" }
      $t = $s.Substring($i + 2, $j - $i - 2)
      [void]$spans.Add(@{ k = $(if ($two -eq '**') { 'b' } else { 'i' }); s = $sb.Length; l = $t.Length })
      [void]$sb.Append($t); $i = $j + 2; continue
    }
    if ($s[$i] -eq '$') {
      $j = $s.IndexOf('$', $i + 1); if ($j -lt 0) { throw "Kapanmayan `$ : $s" }
      $t = $s.Substring($i + 1, $j - $i - 1)
      [void]$spans.Add(@{ k = 'm'; s = $sb.Length; l = $t.Length })
      # Word'ün BuildUp işlemi denklemden sonraki birkaç karakteri silebiliyor;
      # görünmez dolgu bu silmeyi soğurur, artanı belge sonunda temizlenir.
      [void]$sb.Append($t); [void]$sb.Append($PAD); $i = $j + 1; continue
    }
    [void]$sb.Append($s[$i]); $i++
  }
  return , @($sb.ToString(), $spans)
}

# Biçimleri ve matematiği verilen başlangıç konumundan itibaren uygular (matematik sondan başa)
function Apply-Spans($base, $spans) {
  foreach ($sp in $spans) {
    $r = $doc.Range($base + $sp.s, $base + $sp.s + $sp.l)
    if ($sp.k -eq 'b') { $r.Font.Bold = 1 } elseif ($sp.k -eq 'i') { $r.Font.Italic = 1 }
  }
  $maths = @($spans | Where-Object { $_.k -eq 'm' } | Sort-Object { $_.s } -Descending)
  foreach ($sp in $maths) {
    $r = $doc.Range($base + $sp.s, $base + $sp.s + $sp.l)
    [void]$r.OMaths.Add($r)
    $om = $r.OMaths.Item(1)
    $om.BuildUp()
    $nxt = $doc.Range($om.Range.End, $om.Range.End + 1).Text
    if ($nxt -ne [string]$ZCH) { Write-Warning "Dolgu tükendi, metin kaybı olabilir: $($om.Range.Text)" }
  }
}

$cm = 28.3465
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$doc = $null
try {
  $doc = $word.Documents.Add()
  $sel = $word.Selection

  # ---------------- Sayfa düzeni ve stiller ----------------
  $ps = $doc.PageSetup
  $ps.PaperSize = 7                       # A4
  $ps.TopMargin = 2.5 * $cm; $ps.BottomMargin = 2.5 * $cm
  $ps.LeftMargin = 2.5 * $cm; $ps.RightMargin = 2.5 * $cm
  $FONT = 'Times New Roman'

  function Set-Style([int]$id, [double]$size, [int]$bold, [double]$before, [double]$after, [int]$align, [int]$keepNext) {
    $st = $doc.Styles.Item([int]$id)
    $st.Font.Name = $FONT; $st.Font.Size = $size; $st.Font.Bold = $bold
    $st.Font.Italic = 0; $st.Font.Color = 0
    $pf = $st.ParagraphFormat
    $pf.SpaceBefore = $before; $pf.SpaceAfter = $after; $pf.Alignment = $align
    $pf.KeepWithNext = $keepNext
    $pf.LineSpacingRule = 5; $pf.LineSpacing = 12 * 1.15
  }
  Set-Style -1  11 0 0  6 3 0      # Normal (iki yana yaslı)
  Set-Style -2  14 1 14 6 0 -1     # Başlık 1
  Set-Style -3  12 1 8  4 0 -1     # Başlık 2
  Set-Style -35 9.5 0 2 8 1 0      # Resim yazısı
  Set-Style -49 11 0 0  3 3 0      # Madde işaretli liste
  $doc.Styles.Item(-2).ParagraphFormat.OutlineLevel = 1
  $doc.Styles.Item(-3).ParagraphFormat.OutlineLevel = 2

  function Goto-End { [void]$sel.EndKey(6) }
  function New-Para { Goto-End; $sel.InsertParagraphAfter(); Goto-End }

  # Bir paragraf yazar; stil, hizalama ve satır içi biçimler uygulanır
  function Write-Par([string]$text, [int]$style, $align = $null, $size = $null, $bold = $null) {
    Goto-End
    $sel.Style = $style
    $res = Parse-Inline $text
    $plain = $res[0]; $spans = $res[1]
    if ($plain.Length -gt 0) {
      $sel.InsertAfter($plain)
      $base = $sel.Start
      $rng = $doc.Range($base, $base + $plain.Length)
      if ($null -ne $size) { $rng.Font.Size = [double]$size }
      if ($null -ne $bold) { $rng.Font.Bold = [int]$bold }
      Apply-Spans $base $spans
    }
    $last = $doc.Paragraphs.Item($doc.Paragraphs.Count)
    if ($null -ne $align) { $last.Alignment = [int]$align }
    if ($null -ne $script:spaceNext) { $last.SpaceBefore = $script:spaceNext; $script:spaceNext = $null }
    New-Para
  }

  function Write-Table([string]$caption, [string[]]$rows, [string]$opts) {
    $o = $opts.Split('|')
    $widths = if ($o.Count -gt 0 -and $o[0]) { $o[0].Split(',') | ForEach-Object { [double]$_ } } else { $null }
    $align = if ($o.Count -gt 1 -and $o[1] -eq 'L') { 0 } else { 1 }
    Write-Par $caption -35 1
    $doc.Paragraphs.Item($doc.Paragraphs.Count - 1).KeepWithNext = -1
    $cells = foreach ($r in $rows) { , @($r.Trim().Trim('|').Split('|') | ForEach-Object { $_.Trim() }) }
    $nR = $cells.Count; $nC = $cells[0].Count
    Goto-End
    $t = $doc.Tables.Add($sel.Range, $nR, $nC)
    $t.Borders.Enable = 1
    $t.Range.Font.Name = $FONT; $t.Range.Font.Size = 9
    $t.Range.ParagraphFormat.SpaceAfter = 1; $t.Range.ParagraphFormat.SpaceBefore = 1
    $t.Range.ParagraphFormat.LineSpacingRule = 0
    $t.Range.ParagraphFormat.Alignment = $align
    # Kısa tablolar bölünmez; uzun tablolarda yalnızca başlık satırı sonrakine bağlı kalır
    $t.Range.ParagraphFormat.KeepWithNext = $(if ($nR -le 9) { -1 } else { 0 })
    $t.TopPadding = 1; $t.BottomPadding = 1
    for ($i = 0; $i -lt $nR; $i++) {
      for ($j = 0; $j -lt $nC; $j++) {
        $res = Parse-Inline $cells[$i][$j]
        $cr = $t.Cell($i + 1, $j + 1).Range
        $cr.Text = $res[0]
        Apply-Spans $t.Cell($i + 1, $j + 1).Range.Start $res[1]
      }
    }
    $t.Rows.Item(1).Range.Font.Bold = 1
    $t.Rows.Item(1).Shading.BackgroundPatternColor = 14277081   # gri %15
    $t.Rows.Item(1).HeadingFormat = -1
    $t.Rows.Item(1).Range.ParagraphFormat.Alignment = 1
    $t.Rows.Item(1).Range.ParagraphFormat.KeepWithNext = -1
    $t.Rows.Item($nR).Range.ParagraphFormat.KeepWithNext = 0
    if ($widths) {
      for ($j = 0; $j -lt $nC; $j++) { $t.Columns.Item($j + 1).Width = $widths[$j] * $cm }
    }
    $t.Rows.Alignment = 1
    Goto-End
    $script:spaceNext = 8
  }

  function Write-Figure([string]$spec) {
    $f = $spec.Split('|', 3)
    $path = Join-Path $ImgDir $f[0].Trim()
    Goto-End
    $sel.Style = -1
    $doc.Paragraphs.Last.Alignment = 1
    $doc.Paragraphs.Last.KeepWithNext = -1
    $doc.Paragraphs.Last.SpaceBefore = 4
    $doc.Paragraphs.Last.LineSpacingRule = 0
    $shape = $sel.InlineShapes.AddPicture($path, $false, $true)
    $shape.LockAspectRatio = -1
    $shape.Width = [double]$f[1] * $cm
    New-Para
    Write-Par $f[2].Trim() -35 1
  }

  function Write-Code([string]$title, [string[]]$codeLines) {
    if ($title) { Write-Par $title -35 0 9.5 }
    $doc.Paragraphs.Item($doc.Paragraphs.Count - 1).KeepWithNext = -1
    Goto-End
    $startPos = $sel.Start
    for ($i = 0; $i -lt $codeLines.Count; $i++) {
      Goto-End
      $sel.Style = -1
      if ($codeLines[$i].Length -gt 0) { $sel.InsertAfter($codeLines[$i]) }
      $p = $doc.Paragraphs.Last
      $p.Range.Font.Name = 'Consolas'; $p.Range.Font.Size = 8.5
      $p.Alignment = 0; $p.SpaceAfter = 0; $p.SpaceBefore = 0; $p.LineSpacingRule = 0
      $p.LeftIndent = 0.2 * $cm
      $p.KeepWithNext = $(if ($i -lt $codeLines.Count - 1) { -1 } else { 0 })
      New-Para
    }
    $blk = $doc.Range($startPos, $doc.Paragraphs.Item($doc.Paragraphs.Count - 1).Range.End)
    $blk.Shading.BackgroundPatternColor = 15987699     # açık gri
    # Yalnızca dış kenarlar: ardışık paragraflar tek kutu olarak çizilir
    foreach ($side in -1, -2, -3, -4) { $blk.ParagraphFormat.Borders.Item([int]$side).LineStyle = 1 }
    $doc.Paragraphs.Last.Style = -1
    $script:spaceNext = 8
  }

  # ---------------- Ana döngü ----------------
  $i = 0
  while ($i -lt $body.Count) {
    $ln = $body[$i]
    if ($ln.Trim().Length -eq 0 -or $ln.StartsWith('//')) { $i++; continue }
    $cmd = $ln.Split(' ', 2)[0]
    $arg = if ($ln.Length -gt $cmd.Length) { $ln.Substring($cmd.Length + 1) } else { '' }
    switch ($cmd) {
      'H1' { Write-Par $arg -2 }
      'H2' { Write-Par $arg -3 }
      'P'  { Write-Par $arg -1 }
      'B'  { Write-Par $arg -49 }
      'EQ' {
        # Ayraçlar: ", " ve çift boşluk -> em boşluk; numara öncesi boşluklar silinir
        $eq = [regex]::Replace($arg, '\s+#\(', '#(')
        $eq = $eq.Replace(', ', ',' + $EM + $EM)
        $eq = [regex]::Replace($eq, ' {2,}', $EM + $EM)
        Write-Par ('$' + $eq + '$') -1 1
      }
      'CTR' { $f = $arg.Split('|', 3); Write-Par $f[2] -1 1 ([double]$f[0]) ([int]$f[1]) }
      'SP' { for ($k = 0; $k -lt [int]$arg; $k++) { Write-Par '' -1 } }
      'PB' { Goto-End; $sel.InsertBreak(7); Goto-End }
      'FIG' { Write-Figure $arg }
      'TAB' {
        $cap, $opts = $arg.Split('|', 2); if (-not $opts) { $opts = '' }
        $rows = @(); $i++
        while ($body[$i] -ne 'ENDTAB') { $rows += $body[$i]; $i++ }
        Write-Table $cap $rows $opts
      }
      'CODE' {
        $code = @(); $i++
        while ($body[$i] -ne 'ENDCODE') { $code += $body[$i]; $i++ }
        Write-Code $arg $code
      }
      'REFS' {
        foreach ($k in $refNum.Keys) {
          Write-Par ("[{0}]`t{1}" -f $refNum[$k], $refText[$k]) -1 0 10
          $p = $doc.Paragraphs.Item($doc.Paragraphs.Count - 1)
          $p.LeftIndent = 0.9 * $cm; $p.FirstLineIndent = -0.9 * $cm; $p.SpaceAfter = 3
          $p.TabStops.Add(0.9 * $cm) | Out-Null
        }
      }
      default { throw "Bilinmeyen komut: $cmd" }
    }
    $i++
  }

  # Dolgu karakterlerini temizle (Bul/Değiştir, tümünü değiştir)
  $f = $doc.Content.Find
  [void]$f.Execute([string]$ZCH, $false, $false, $false, $false, $false, $true, 1, $false, '', 2)
  $leftZ = ($doc.Content.Text.ToCharArray() | Where-Object { $_ -eq $ZCH }).Count
  if ($leftZ -gt 0) { Write-Warning "Temizlenemeyen dolgu karakteri: $leftZ" }

  # Kullanılmayan kaynak uyarısı
  foreach ($k in $refText.Keys) { if (-not $refNum.Contains($k)) { Write-Warning "Metinde atıf yapılmayan kaynak: $k" } }

  # Altbilgi: sayfa numarası (kapak hariç)
  $sec = $doc.Sections.Item(1)
  $sec.PageSetup.DifferentFirstPageHeaderFooter = -1
  [void]$sec.Footers.Item(1).PageNumbers.Add(1, $false)

  $doc.SaveAs2($OutDocx, 16)
  $pdf = [IO.Path]::ChangeExtension($OutDocx, '.pdf')
  $doc.ExportAsFixedFormat($pdf, 17)
  "Sayfa: $($doc.ComputeStatistics(2))  Kelime: $($doc.ComputeStatistics(0))  Denklem: $($doc.OMaths.Count)  Tablo: $($doc.Tables.Count)  Kaynak: $($refNum.Count)"
  $doc.Close(0)
}
catch { if ($doc) { $doc.Close(0) }; throw }
finally { $word.Quit(); [void][Runtime.InteropServices.Marshal]::ReleaseComObject($word) }
