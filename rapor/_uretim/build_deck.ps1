param([string]$Out, [string]$ImgDir, [string]$PngDir = '')
$ErrorActionPreference = 'Stop'

function C([string]$hex) { [Convert]::ToInt32($hex.Substring(1,2),16) + 256*[Convert]::ToInt32($hex.Substring(3,2),16) + 65536*[Convert]::ToInt32($hex.Substring(5,2),16) }
$INK = C '#1B2430'; $BODY = C '#3E4956'; $MUTED = C '#5B6470'; $PAPER = C '#F7F6F2'; $PAPER2 = C '#ECE9E1'
$CARD = C '#FFFFFF'; $LINE = C '#D6D1C4'; $ACC = C '#A8432A'; $BLUE = C '#2F5D8A'; $NIGHT = C '#C9D2DC'; $CREAM = C '#F3EFE6'
$ACC_D = C '#E59A7E'; $NMUT = C '#93A0AE'
$HEAD = 'Georgia'; $SANS = 'Segoe UI'; $MATH = 'Cambria Math'; $MONO = 'Consolas'
$TOTAL = 10

function Set-Bg($sl, $col) { $sl.FollowMasterBackground = 0; $sl.Background.Fill.Solid(); $sl.Background.Fill.ForeColor.RGB = $col }

# **kalın** işaretli metin; `n paragraf ayırır
function Add-Text($sl, [double]$l, [double]$t, [double]$w, [double]$h, [string]$text, [double]$size, [int]$col,
                  [string]$font = $SANS, [int]$bold = 0, [int]$align = 1, [double]$after = 0, [int]$boldCol = -1) {
  $sh = $sl.Shapes.AddTextbox(1, $l, $t, $w, $h)
  $tf = $sh.TextFrame; $tf.WordWrap = -1; $tf.AutoSize = 0
  $tf.MarginLeft = 0; $tf.MarginRight = 0; $tf.MarginTop = 0; $tf.MarginBottom = 0
  $plain = ''; $spans = @(); $parts = $text -split '\*\*'
  for ($i = 0; $i -lt $parts.Count; $i++) {
    $p = $parts[$i] -replace "`n", "`r"
    if ($i % 2 -eq 1) { $spans += , @(($plain.Length + 1), $p.Length) }
    $plain += $p
  }
  $tr = $tf.TextRange; $tr.Text = $plain
  $tr.Font.Name = $font; $tr.Font.Size = $size; $tr.Font.Color.RGB = $col; $tr.Font.Bold = $bold
  $tr.ParagraphFormat.Alignment = $align; $tr.ParagraphFormat.SpaceAfter = $after
  foreach ($s in $spans) { $r = $tr.Characters($s[0], $s[1]); $r.Font.Bold = -1; if ($boldCol -ge 0) { $r.Font.Color.RGB = $boldCol } }
  return $sh
}

function Set-Bullets($sh, [int]$numbered = 0, [int]$col = -1) {
  $tr = $sh.TextFrame.TextRange
  $tr.ParagraphFormat.Bullet.Visible = -1
  if ($numbered) { $tr.ParagraphFormat.Bullet.Type = 2; $tr.ParagraphFormat.Bullet.Style = 3 }
  else { $tr.ParagraphFormat.Bullet.Type = 1; $tr.ParagraphFormat.Bullet.Character = 8226 }
  if ($col -ge 0) { $tr.ParagraphFormat.Bullet.Font.Color.RGB = $col }
  $sh.TextFrame.Ruler.Levels(1).FirstMargin = 0; $sh.TextFrame.Ruler.Levels(1).LeftMargin = 16
}

function Add-Rect($sl, [double]$l, [double]$t, [double]$w, [double]$h, [int]$fill, [int]$line = -1) {
  $sh = $sl.Shapes.AddShape(1, $l, $t, $w, $h)
  $sh.Fill.Solid(); $sh.Fill.ForeColor.RGB = $fill
  if ($line -ge 0) { $sh.Line.ForeColor.RGB = $line; $sh.Line.Weight = 0.75 } else { $sh.Line.Visible = 0 }
  $sh.Shadow.Visible = 0
  return $sh
}

function Add-Pic($sl, [string]$file, [double]$l, [double]$t, [double]$w, [double]$ratio, [string]$alt) {
  $sh = $sl.Shapes.AddPicture((Join-Path $ImgDir $file), 0, -1, $l, $t, $w, ($w / $ratio))
  $sh.AlternativeText = $alt
  return $sh
}

function Add-Table($sl, [double]$l, [double]$t, [double[]]$widths, $rows, [double]$size, [int[]]$mathCols = @(), [int]$headBg = $INK) {
  $nr = $rows.Count; $nc = $widths.Count; $w = ($widths | Measure-Object -Sum).Sum
  $sh = $sl.Shapes.AddTable($nr, $nc, $l, $t, $w, $nr * 18)
  $tb = $sh.Table
  try { $tb.ApplyStyle('{2D5ABB26-0587-4C30-8999-92F81FD0307C}', $false) } catch {}
  for ($c = 1; $c -le $nc; $c++) { $tb.Columns.Item($c).Width = $widths[$c-1] }
  for ($r = 1; $r -le $nr; $r++) {
    for ($c = 1; $c -le $nc; $c++) {
      $cell = $tb.Cell($r, $c); $cs = $cell.Shape
      $cs.Fill.Visible = -1; $cs.Fill.Solid()
      if ($r -eq 1) { $cs.Fill.ForeColor.RGB = $headBg } elseif ($r % 2 -eq 0) { $cs.Fill.ForeColor.RGB = $CARD } else { $cs.Fill.ForeColor.RGB = C '#FAF8F3' }
      $tf = $cs.TextFrame; $tf.MarginLeft = 7; $tf.MarginRight = 7; $tf.MarginTop = 4; $tf.MarginBottom = 4; $tf.VerticalAnchor = 3
      $tr = $tf.TextRange; $tr.Text = [string]$rows[$r-1][$c-1]
      $tr.Font.Size = $size; $tr.Font.Name = $SANS
      if ($r -eq 1) { $tr.Font.Bold = -1; $tr.Font.Color.RGB = $CREAM }
      else { $tr.Font.Bold = 0; $tr.Font.Color.RGB = $BODY; if ($mathCols -contains $c) { $tr.Font.Name = $MATH } }
      $b = $cell.Borders.Item(3); $b.Visible = -1; $b.ForeColor.RGB = $LINE; $b.Weight = 0.75
    }
    $tb.Rows.Item($r).Height = 16
  }
  return $sh
}

function Add-Head($sl, [string]$eyebrow, [string]$title, [int]$dark = 0) {
  [void](Add-Text $sl 48 40 864 16 $eyebrow 11 $(if ($dark) { $ACC_D } else { $ACC }) $SANS 1)
  $sh = Add-Text $sl 48 60 864 70 $title 28 $(if ($dark) { $CREAM } else { $INK }) $HEAD 1
  $sh.TextFrame.TextRange.Font.Bold = 0
  return $sh
}

function Add-Foot($sl, [int]$n, [int]$dark = 0) {
  $col = $(if ($dark) { $NMUT } else { $MUTED })
  [void](Add-Text $sl 48 510 500 14 'MAT353 · Nümerik Analiz · Birinci mertebe optimizasyon yöntemleri' 10 $col)
  [void](Add-Text $sl 812 510 100 14 "$n / $TOTAL" 10 $col $SANS 0 3)
}

function Add-Stat($sl, [double]$l, [double]$t, [double]$w, [string]$num, [string]$lbl, [int]$numCol = $ACC, [int]$lblCol = $BODY, [int]$lineCol = $LINE, [double]$lblH = 46) {
  $ln = $sl.Shapes.AddLine($l, $t, $l + $w, $t); $ln.Line.ForeColor.RGB = $lineCol; $ln.Line.Weight = 1
  [void](Add-Text $sl $l ($t + 8) $w 38 $num 30 $numCol $HEAD 1)
  [void](Add-Text $sl $l ($t + 48) $w $lblH $lbl 12.5 $lblCol)
}

function Set-Notes($sl, [string]$txt) { $sl.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text = $txt }

$app = New-Object -ComObject PowerPoint.Application
$pres = $null
try {
  $pres = $app.Presentations.Add(0)
  $pres.PageSetup.SlideWidth = 960; $pres.PageSetup.SlideHeight = 540

  # ---------------- 1. KAPAK ----------------
  $s = $pres.Slides.Add(1, 12); Set-Bg $s $INK
  [void](Add-Rect $s 48 52 40 4 $ACC_D)
  [void](Add-Text $s 48 70 864 16 'MAT353 · NÜMERİK ANALİZ · DÖNEM PROJESİ' 12 $ACC_D $SANS 1)
  [void](Add-Text $s 48 100 780 160 'Evrişimli sinir ağlarında birinci mertebe optimizasyon yöntemleri' 44 $CREAM $HEAD 0)
  [void](Add-Text $s 48 262 760 60 'Yakınsama, gradyan sönmesi, Hessian spektrumu ve sonlu fark hata analizi' 20 $NIGHT)
  $ln = $s.Shapes.AddLine(48, 416, 912, 416); $ln.Line.ForeColor.RGB = (C '#3A4757'); $ln.Line.Weight = 1
  [void](Add-Text $s 48 430 864 20 '[Ad Soyad] · [Ad Soyad] · [Ad Soyad]' 15 $CREAM)
  [void](Add-Text $s 48 456 864 18 'Öğretim üyesi: Doç. Dr. Murat Gezer · İstanbul Medeniyet Üniversitesi · Aralık 2026' 12 $NMUT)
  Set-Notes $s 'Açılış (≈30 sn). Ekibi tanıtın. Tek cümlelik özet: SGD, Momentum, RMSprop ve Adam’’ı formüllerinden sıfırdan yazdık ve Fashion-MNIST üzerinde yalnızca son doğrulukla değil, nümerik analiz araçlarıyla karşılaştırdık: gradyan sönmesi, Hessian’’ın en büyük özdeğeri ve sonlu fark hata analizi.'

  # ---------------- 2. PROBLEM ----------------
  $s = $pres.Slides.Add(2, 12); Set-Bg $s $PAPER
  [void](Add-Head $s 'GİRİŞ · PROBLEM' 'Bir CNN’’i eğitmek, yaklaşık 493 bin boyutlu bir minimizasyon problemidir')
  [void](Add-Text $s 48 140 864 60 'Kayıp L(θ) konveks değildir ve yalnızca gürültülü mini-batch gradyanlarıyla hesaplanabilir. Açık Hessian **P² ≈ 2,4·10¹¹ eleman**, float64’’te **≈ 1,9 TB** tutar; bu yüzden Newton yöntemi yerine yalnızca gradyan kullanan birinci mertebe yöntemler uygulanır.' 15 $BODY $SANS 0 1 0 $INK)
  $cw = 272; $titles = @('Adım boyu η', 'Gradyan sönmesi', 'Kayan nokta hassasiyeti')
  $texts = @('Küçük η yakınsamayı yavaşlatır; büyük η iterasyonu ıraksatır ve NaN/Inf üretir.',
             'Geri yayılımda gradyanlar katman katman çarpılır; derin ağda ilk katmanlara çok zayıf sinyal ulaşır.',
             'float32 ile float64 arasındaki fark, gradyanı sonlu farkla ne kadar doğru sınayabileceğimizi belirler.')
  for ($i = 0; $i -lt 3; $i++) {
    $x = 48 + $i * ($cw + 24)
    [void](Add-Rect $s $x 228 $cw 160 $CARD)
    [void](Add-Rect $s $x 228 $cw 4 $(if ($i -eq 2) { $BLUE } else { $ACC }))
    [void](Add-Text $s ($x + 20) 248 ($cw - 40) 30 ([string]($i + 1)) 26 $(if ($i -eq 2) { $BLUE } else { $ACC }) $HEAD 1)
    [void](Add-Text $s ($x + 20) 284 ($cw - 40) 24 $titles[$i] 17 $INK $HEAD 1)
    [void](Add-Text $s ($x + 20) 314 ($cw - 40) 100 $texts[$i] 13.5 $BODY)
  }
  [void](Add-Text $s 48 410 864 16 'Veri: Fashion-MNIST · 60 000 eğitim ve 10 000 test görüntüsü · 28×28 gri seviye · 10 sınıf' 12 $MUTED)
  Add-Foot $s 2
  Set-Notes $s 'Problem (≈1,5 dk). Eğitim, çapraz entropi kaybının minimizasyonudur. Derin modelimizde P ≈ 4,9·10⁵ parametre var; Hessian P² eleman, yani float64’’te yaklaşık 1,9 TB. Bu nedenle ikinci mertebe yöntemler doğrudan uygulanamaz. Üç nümerik zorluğu sırayla vurgulayın: adım boyu seçimi (kararlılık), gradyan sönmesi (derinlik) ve kayan nokta hassasiyeti (doğrulama). Sunumun geri kalanı bu üç soruya yanıt veriyor.'

  # ---------------- 3. YÖNTEM ----------------
  $s = $pres.Slides.Add(3, 12); Set-Bg $s $PAPER2
  [void](Add-Head $s 'YÖNTEM · DÖRT ALGORİTMA' 'Dört güncelleme kuralı, hazır sınıf kullanılmadan formüllerinden yazıldı')
  $rows = @(
    @('Yöntem', 'Güncelleme kuralı (g: mini-batch gradyanı)', 'İlk adımın büyüklüğü (t = 1)'),
    @('SGD', 'θ ← θ − η g', 'η|g|  (gradyana bağlı)'),
    @('Momentum (μ = 0,9)', 'v ← μv + g,   θ ← θ − η v', 'η|g|  (gradyana bağlı)'),
    @('RMSprop (α = 0,99)', 's ← αs + (1−α) g²,   θ ← θ − η g / (√s + ε)', 'η / √(1−α) = 10η'),
    @('Adam (β₁ = 0,9, β₂ = 0,999)', 'm̂ = m/(1−β₁ᵗ),  v̂ = v/(1−β₂ᵗ),   θ ← θ − η m̂ / (√v̂ + ε)', 'η'),
    @('Adam, düzeltmesiz (ablasyon)', 'm ve v doğrudan kullanılır', 'η(1−β₁)/√(1−β₂) ≈ 3,16η'))
  [void](Add-Table $s 48 140 @(190, 444, 230) $rows 13 @(2, 3))
  [void](Add-Rect $s 48 304 864 92 $CARD)
  [void](Add-Rect $s 48 304 5 92 $BLUE)
  [void](Add-Text $s 70 316 828 72 '**Uyarlamalı yöntemlerin ilk adımı gradyanın ölçeğinden bağımsızdır.** Bu sayede gradyanı sönmüş katmanlar da η mertebesinde güncellenir; ancak büyük η’’da küçük gradyan artık fren işlevi görmez. Dört uygulama da torch.optim referansıyla kayan nokta hassasiyetinde eşleşiyor.' 14 $BODY $SANS 0 1 0 $INK)
  Add-Foot $s 3
  Set-Notes $s 'Yöntem (≈1,5 dk). Dört kuralı kısaca okuyun. Asıl vurgu son sütun: t = 1’’de Adam’’ın adımı tam η, yanlılık düzeltmesi kaldırılırsa ≈3,16η, RMSprop’’unki ise 10η. Bu büyüklükler gradyanın ölçeğinden bağımsız; birazdan göreceğimiz iki sonucu açıklıyor: uyarlamalı yöntemler sönmüş derin ağda öğreniyor, ama büyük η’’da RMSprop ilk adımda patlıyor. Doğrulama: her optimizer için elle hesaplanan değerler ve torch.optim ile karşılaştıran birim testleri var.'

  # ---------------- 4. DENEY TASARIMI ----------------
  $s = $pres.Slides.Add(4, 12); Set-Bg $s $PAPER
  [void](Add-Head $s 'YÖNTEM · DENEY TASARIMI' 'Altı deney, her biri tek bir soruyu yanıtlıyor')
  $mods = @(@('ShallowCNN', '3 konvolüsyon · 458 570 parametre'), @('DeepCNN', '8 konvolüsyon, normalizasyon yok · 493 114 parametre'), @('DeepCNN+BN', 'Her konvolüsyondan sonra BatchNorm (ablasyon)'))
  for ($i = 0; $i -lt 3; $i++) {
    $x = 48 + $i * 296
    [void](Add-Rect $s $x 124 272 58 $CARD)
    [void](Add-Rect $s $x 124 4 58 $(if ($i -eq 1) { $ACC } else { $BLUE }))
    [void](Add-Text $s ($x + 16) 132 248 20 $mods[$i][0] 15 $INK $HEAD 1)
    [void](Add-Text $s ($x + 16) 154 248 24 $mods[$i][1] 11.5 $BODY)
  }
  $rows = @(
    @('No', 'Soru', 'Yapılandırma'),
    @('1', 'Yakınsama hızı ve tohum kararlılığı', '2 model × 4 yöntem × 5 tohum × 10 epoch'),
    @('2', 'Öğrenme oranı duyarlılığı, nümerik patlama', 'DeepCNN; η ∈ {0,5; 0,1; 0,01; 10⁻³; 10⁻⁴} × 3 tohum × 3 epoch'),
    @('3', 'Gradyan doğrulaması, sonlu fark hatası', '25 adım boyu h ∈ [10⁻¹², 1]; float64 ve float32'),
    @('4', 'CPU–GPU performansı', 'B = 128; 5 ısınma + 50 ölçüm iterasyonu'),
    @('5', 'Hessian spektrumu, kararlılık sınırı', 'Kuvvet yöntemi; Nₛ = 1024 örnek; 0–5. epoch; 3 tohum'),
    @('6', 'Ablasyon: BatchNorm, yanlılık düzeltmesi', 'Aynı 5 tohum; eşleştirilmiş t-testi'))
  [void](Add-Table $s 48 200 @(44, 330, 490) $rows 13)
  [void](Add-Text $s 48 392 864 16 'Tohumlar 42, 100, 2024, 7, 99 · cuDNN deterministik · Colab GPU: T4 (Deney 1, 3, 4), L4 (Deney 2, 5, 6)' 11 $MUTED)
  Add-Foot $s 4
  Set-Notes $s 'Deney tasarımı (≈1 dk). İki ana model: sığ ağ ve bilerek normalizasyon katmanı konmamış 8 konvolüsyonlu derin ağ; ablasyon için BatchNorm’’lu sürüm. Aynı tohumla tüm yöntemler aynı başlangıç ağırlıklarını ve aynı mini-batch sırasını görür; bu da Deney 6’’daki eşleştirilmiş t-testini mümkün kılar. Öğrenme oranları her yöntemin literatürdeki varsayılanı: SGD ve Momentum 0,01, RMSprop ve Adam 0,001.'

  # ---------------- 5. YAKINSAMA ----------------
  $s = $pres.Slides.Add(5, 12); Set-Bg $s $PAPER
  [void](Add-Head $s 'SONUÇ 1 · YAKINSAMA (DENEY 1)' 'Sığ ağda dört yöntem de öğreniyor; derin ağda SGD takılıyor')
  [void](Add-Pic $s 'loss_convergence.png' 48 140 580 2.07 'Eğitim kaybının epoch’’a göre değişimi, ShallowCNN ve DeepCNN, dört yöntem')
  [void](Add-Text $s 48 425 580 30 'Eğitim kaybı (log ölçek). Çizgi 5 tohumun ortalaması, gölge ±1 standart sapma.' 11 $MUTED)
  Add-Stat $s 656 140 256 '8 kat' 'Adam %85 doğruluğa SGD’’den 8 kat az iterasyonla ulaşıyor (sığ ağ: 469’’a karşı 3752).'
  Add-Stat $s 656 240 256 '%18,7 ± 9,3' 'Derin ağda SGD’’nin test doğruluğu; 5 tohumun hiçbiri %85’’e ulaşamadı.'
  Add-Stat $s 656 340 256 '%91,5 ± 0,4' 'Derin ağda Adam. RMSprop %91,2, Momentum %90,4: uyarlamalı yöntemler derinlikten etkilenmiyor.' $BLUE $BODY $LINE 62
  Add-Foot $s 5
  Set-Notes $s 'Yakınsama (≈1,5 dk). Sol panel sığ ağ: dört yöntem de iniyor; Momentum, RMSprop ve Adam %91,3–91,8 bandında, farklar tohumlar arası sapma mertebesinde. Asıl fark hızda: Adam %85’’e ilk epoch’’ta, SGD ancak 8. epoch’’ta ulaşıyor. Sağ panel derin ağ: SGD kaybı ln 10 ≈ 2,303 seviyesinden hiç ayıramıyor. Momentum 1–2 epoch platoda kalıp tohuma göre aniden kurtuluyor; geniş gölge bundan. Soru: SGD neden takılıyor? Sonraki slayt.'

  # ---------------- 6. GRADYAN SÖNMESİ + ABLASYON ----------------
  $s = $pres.Slides.Add(6, 12); Set-Bg $s $PAPER2
  [void](Add-Head $s 'SONUÇ 2 · GRADYAN SÖNMESİ VE ABLASYON (DENEY 6)' 'Neden: gradyan her katmanda zayıflıyor, BatchNorm bunu gideriyor')
  Add-Stat $s 48 128 272 '1/6' 'Varsayılan başlatmada katman başına varyans kazancı: Var(h⁽ˡ⁾) ≈ ⅙ Var(h⁽ˡ⁻¹⁾).'
  Add-Stat $s 344 128 272 '≈ 300 kat' 'DeepCNN–SGD: ilk konvolüsyonun gradyan normu 1,2·10⁻⁴, fc1’’inki 3,8·10⁻².'
  Add-Stat $s 640 128 272 '+72 puan' 'BatchNorm ile SGD %18,7 → %90,9 (eşleştirilmiş t-testi, p = 5,9·10⁻⁵).' $BLUE
  [void](Add-Pic $s 'ablation.png' 96 240 768 3.31 'Ablasyon: SGD ve Momentum için BatchNorm etkisi, Adam yanlılık düzeltmesinin etkisi')
  [void](Add-Text $s 96 474 768 28 '(a) SGD ve (b) Momentum için BatchNorm etkisi, (c) Adam yanlılık düzeltmesi (fark −0,86 puan, p = 0,067: anlamlı değil).' 11 $MUTED)
  Add-Foot $s 6
  Set-Notes $s 'Gradyan sönmesi (≈2 dk). Teori: varsayılan kaiming-uniform (a = √5) başlatmada Var(W) = 1/(3 n_in); ReLU yarıyı sıfırladığı için katman başına varyans 1/6’’ya iniyor, yani genlikte ≈2,4 kat. Sekiz katmanda bu birikiyor. Ölçüm: ilk konvolüsyonla fc1 arasında ≈300 kat fark; ilk/son katman oranı SGD için 10 epoch boyunca 0,01–0,07. Neden-sonuç kanıtı ablasyon: yalnızca BatchNorm eklenince SGD %18,7’’den %90,9’’a çıkıyor, p = 5,9·10⁻⁵. Momentum’’da plato tamamen kayboluyor (+1,39 puan, p = 1,5·10⁻³). Yanlılık düzeltmesini kaldırmanın etkisi küçük ve anlamlı değil, çünkü η = 0,001’’de 3,16η hâlâ küçük.'

  # ---------------- 7. KARARLILIK ----------------
  $s = $pres.Slides.Add(7, 12); Set-Bg $s $PAPER
  [void](Add-Head $s 'SONUÇ 3 · HESSIAN SPEKTRUMU VE KARARLILIK (DENEY 2, 5)' 'Iraksama, η kuramsal sınır 2/λmax’’ı aştığında başlıyor')
  Add-Stat $s 48 128 272 '0,1–0,5 → 61–99' 'DeepCNN’’de λmax: platoda yüzey neredeyse düz, 3. epoch’’ta keskin bölgeye geçiş.'
  Add-Stat $s 344 128 272 'kritik η: 11 → 0,02' 'SGD η = 0,5 ilk epoch’’u geçiyor, sınır düşünce 2. epoch’’ta NaN/Inf üretiyor.'
  Add-Stat $s 640 128 272 '10η = 5' 'RMSprop η = 0,5: ilk adım her ağırlıkta 5 birim, 2. iterasyonda NaN.' $BLUE
  [void](Add-Pic $s 'sharpness.png' 104 240 752 3.09 'Kuvvet yönteminin yakınsaması, en büyük özdeğerin değişimi ve kararlılık sınırı')
  [void](Add-Text $s 104 485 752 16 '(a) Kuvvet yönteminin yakınsaması, (b) λmax’’ın eğitim boyunca değişimi, (c) 2/λmax ve Deney 2’’deki ıraksama sınırı.' 11 $MUTED)
  Add-Foot $s 7
  Set-Notes $s 'Kararlılık (≈2 dk). Kuadratik yaklaşımda gradyan inişi e_(k+1) = (I − ηH) e_k; kararlılık için η < 2/λmax gerekir. H’’yi kuramayız; Hessian–vektör çarpımını iki geri yayılımla hesaplayıp kuvvet yöntemiyle λmax’’ı buluyoruz. (a) Başlangıçta spektral boşluk küçük, 76 ± 35 iterasyon; eğitimden sonra 10–13 iterasyon yetiyor. (b) DeepCNN platoda düz (λmax 0,1–0,5), platodan çıkınca 61–99. (c) Bu yüzden SGD η = 0,5 ilk epoch’’u geçiyor ama sınır 0,02–0,07’’ye düşünce 2. epoch’’ta ıraksıyor. Not: SGD η = 0,1 sınırın üstünde olduğu hâlde 2/3 tohumda öğrendi; bu, literatürdeki kararlılık kıyısı (edge of stability) davranışıyla uyumlu. Ayrıca bazı koşular NaN üretmeden ölüyor (ReLU’’lar kalıcı sıfır), yani yalnızca NaN denetimi yetmez.'

  # ---------------- 8. SONLU FARK ----------------
  $s = $pres.Slides.Add(8, 12); Set-Bg $s $PAPER2
  [void](Add-Head $s 'SONUÇ 4 · GRADYAN DOĞRULAMASI (DENEY 3)' 'Sonlu fark hatası, kuramın öngördüğü V eğrisini izliyor')
  [void](Add-Pic $s 'fd_error.png' 48 136 520 2.29 'Merkezi sonlu fark göreli hatasının adım boyuna göre değişimi, float64 ve float32')
  [void](Add-Text $s 48 368 520 30 'Göreli hata (log-log). Kesikli çizgiler ε/h ve h² eğimleri, noktalı dikey çizgiler kuramsal h*.' 11 $MUTED)
  [void](Add-Rect $s 592 136 320 72 $CARD)
  [void](Add-Rect $s 592 136 5 72 $BLUE)
  [void](Add-Text $s 610 148 296 64 ('|Dₕ − ∇L·v| ≈ (h²/6)|L‴| + ε|L|/h' + "`n" + '⇒  h* ∼ ∛ε') 15 $INK $MATH 0 1 6)
  $rows = @(@('Aritmetik', 'Kuram h*', 'En iyi h', 'En küçük hata'),
            @('float64', '6·10⁻⁶', '10⁻⁵', '1,6·10⁻⁸'),
            @('float32', '5·10⁻³', '3,2·10⁻²', '1,7·10⁻²'))
  [void](Add-Table $s 592 224 @(72, 70, 70, 108) $rows 12 @())
  [void](Add-Text $s 592 312 320 140 ('DeepCNN değerleri. float64’’te otomatik türev **8–9 anlamlı basamakta** doğrulandı.' + "`n" + 'float32’’de h ≤ 10⁻⁶ için θ ± hv yuvarlamada θ’’ya eşit çıkıyor ve Dₕ = 0 bulunuyor.') 13 $BODY $SANS 0 1 8 $INK)
  Add-Foot $s 8
  Set-Notes $s 'Sonlu fark (≈1,5 dk). Merkezi farkın kesme hatası O(h²), yuvarlama hatası ε|L|/h; toplam hata h* ∼ ε^(1/3)''te en küçük. float64 için h* ≈ 6·10⁻⁶, float32 için ≈ 5·10⁻³. Grafikte V biçimi açıkça görülüyor: sağ kol eğim +2, sol kol −1. DeepCNN’’de en iyi adım 10⁻⁵, kurama çok yakın. ShallowCNN’’de h ≥ 3·10⁻⁴''te ani sıçrama var; bu kesme hatası değil, ReLU/MaxPool kırılma noktalarının aşılması. float32’’de ulaşılabilen en iyi hata float64’’ten 5–6 mertebe kötü. Sonuç: geri yayılımımız doğru.'

  # ---------------- 9. TEST VE DEMO ----------------
  $s = $pres.Slides.Add(9, 12); Set-Bg $s $PAPER
  [void](Add-Head $s 'TEST SÜREÇLERİ VE CANLI DEMO' '23 birim testinin tamamı geçiyor; tüm hat yaklaşık bir dakikada çalışıyor')
  $cw = 272
  [void](Add-Rect $s 48 140 $cw 260 $CARD); [void](Add-Rect $s 48 140 $cw 4 $BLUE)
  [void](Add-Text $s 66 156 ($cw - 36) 22 'Ne test edildi' 17 $INK $HEAD 1)
  $b = Add-Text $s 66 188 ($cw - 36) 260 ('Optimizer adımları: elle hesap ve torch.optim ile karşılaştırma' + "`n" + 'Sonlu fark: float64’’te otomatik türev = sonlu fark' + "`n" + 'Kuvvet yöntemi: bilinen spektrumda λmax = 5, oran (3/5)²' + "`n" + 'Determinizm, NaN yakalama, koşu durumu sınıflaması') 13 $BODY $SANS 0 1 8
  Set-Bullets $b 0 $BLUE
  $x = 48 + $cw + 24
  [void](Add-Rect $s $x 140 $cw 260 $CARD); [void](Add-Rect $s $x 140 $cw 4 $ACC)
  [void](Add-Text $s ($x + 18) 156 ($cw - 36) 22 'Bulunan 3 hata' 17 $INK $HEAD 1)
  $b = Add-Text $s ($x + 18) 188 ($cw - 36) 260 ('**Kuvvet yöntemi testi:** yakınsama oranı ilk iterasyonlarda değil, asimptotik bölgede ölçülmeliydi.' + "`n" + '**Negatif özdeğer (−0,19):** plato bölgesinde kritik η tanımsız sayıldı, grafikten çıkarıldı.' + "`n" + '**Log eksende bant:** ortalama − std negatif oluyordu; log-simetrik bant kullanıldı.') 13 $BODY $SANS 0 1 8 $INK
  Set-Bullets $b 1 $ACC
  $x = 48 + 2 * ($cw + 24)
  [void](Add-Rect $s $x 140 $cw 260 $INK)
  [void](Add-Text $s ($x + 18) 156 ($cw - 36) 22 'Canlı demo (Colab)' 17 $CREAM $HEAD 1)
  [void](Add-Text $s ($x + 18) 194 ($cw - 36) 70 ('python tests.py' + "`n" + 'python experiments.py --quick') 12.5 $CREAM $MONO 0 1 6)
  [void](Add-Text $s ($x + 18) 250 ($cw - 36) 130 '6000 örneklik alt kümede tüm deney hattı yaklaşık bir dakikada çalışır. Aynı deneyler PyTorch’’un hazır optimizer’’larıyla da tekrarlanabilir (‑‑optim‑impl torch).' 13 $NIGHT)
  Add-Foot $s 9
  Set-Notes $s 'Test ve demo (≈2,5 dk, demo dahil). Testler unittest ile, veri indirmeden ≈15 saniyede çalışıyor. Bulunan hatalardan en öğreticisi kuvvet yöntemi testi: ilk sürüm oranı 5.–10. iterasyonda ölçüyor ve kuramsal 0,36⁵ yerine 0,33 buluyordu; çünkü (|λ₂/λ₁|)^2k oranı yalnızca asimptotik bölgede geçerli. Demo: Colab’’da önce tests.py (23/23 OK), sonra experiments.py --quick. Demo sırasında ağ veya GPU sorunu olursa önceden kaydedilmiş çıktıları (results/tests_output.txt) gösterin.'

  # ---------------- 10. SONUÇ VE GÖREV DAĞILIMI ----------------
  $s = $pres.Slides.Add(10, 12); Set-Bg $s $INK
  [void](Add-Head $s 'SONUÇ VE GÖREV DAĞILIMI' 'Asıl farkı derinlik ve kararlılık belirliyor' 1)
  $b = Add-Text $s 48 140 520 330 ('Sığ ağda yöntemler benzer doğruluğa ulaşıyor; fark hızda: **Adam SGD’’den 8 kat az iterasyon** gerektiriyor.' + "`n" + 'Normalizasyonsuz derin ağda gradyan sönmesi SGD’’yi işlevsiz kılıyor; **BatchNorm +72 puan** kazandırıyor.' + "`n" + 'Uyarlamalı yöntemler sönmeye dayanıklı ama büyük η’’da kırılgan; **2/λmax** ıraksamanın zamanlamasını açıklıyor.' + "`n" + 'float64 sonlu fark otomatik türevi **8–9 basamakta** doğruluyor; en iyi adım **h* ∼ ∛ε**.') 14.5 $NIGHT $SANS 0 1 12 $CREAM
  Set-Bullets $b 1 $ACC_D
  [void](Add-Rect $s 600 140 312 300 (C '#243042'))
  [void](Add-Text $s 620 156 272 22 'Görev dağılımı' 17 $CREAM $HEAD 1)
  $people = @(@('[Ad Soyad]', '[ör. optimizer’’lar, birim testleri]'), @('[Ad Soyad]', '[ör. Deney 1–4, gradyan analizi]'), @('[Ad Soyad]', '[ör. Hessian, kuvvet yöntemi, rapor]'))
  for ($i = 0; $i -lt 3; $i++) {
    $y = 196 + $i * 84
    $ln = $s.Shapes.AddLine(620, $y, 892, $y); $ln.Line.ForeColor.RGB = (C '#3A4757'); $ln.Line.Weight = 1
    [void](Add-Text $s 620 ($y + 10) 272 20 $people[$i][0] 14.5 $CREAM $SANS 1)
    [void](Add-Text $s 620 ($y + 34) 272 40 $people[$i][1] 12.5 $NMUT)
  }
  Add-Foot $s 10 1
  Set-Notes $s 'Sonuç (≈1,5 dk). Dört bulguyu okuyun. Öneriler (sorulursa): He başlatması ve artık bağlantıların BatchNorm’’a alternatif olarak karşılaştırılması; öğrenme oranı ısınması ve gradyan kırpma; λmax’’ın her yöntemin kendi yörüngesinde ölçülmesi; Hessian’’sız Newton yöntemi. Sınırlılıklar: tek veri kümesi, 10 epoch, n = 5 tohum. Görev dağılımını her üye kendi kısmı için bir cümleyle söylesin. Teşekkür edip soruları alın.'

  $pres.SaveAs($Out)
  $pdf = [IO.Path]::ChangeExtension($Out, '.pdf')
  $pres.SaveAs($pdf, 32)
  if ($PngDir) {
    New-Item -ItemType Directory -Force $PngDir | Out-Null
    for ($i = 1; $i -le $pres.Slides.Count; $i++) { $pres.Slides.Item($i).Export((Join-Path $PngDir ("s{0:00}.png" -f $i)), 'PNG', 1600, 900) }
  }
  "OK: $Out ($($pres.Slides.Count) slayt)"
}
finally {
  if ($pres) { $pres.Close(); [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($pres) }
  $app.Quit()
  [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($app)
  [GC]::Collect(); [GC]::WaitForPendingFinalizers()
}
