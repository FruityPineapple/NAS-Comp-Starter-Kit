# Architektur des NAS

Stand: 30. Juli 2026

## Aktiver Controller-Stand nach der Log-Analyse

Dieser Abschnitt ersetzt bei Widerspruechen die aelteren Beschreibungen
weiter unten:

- Architektur und Trainingsrecipe sind entkoppelt. Alle Architekturen werden
  zuerst mit demselben neutralen `architecture_probe` verglichen.
- Familienquoten enthalten bewusst verschiedene Parametergroessen. Proxies
  dienen nur dem Pre-Screen und gehen nicht mehr in die labelbasierte Utility
  ein.
- AdamW-Zustaende bleiben ueber Fidelity-Runden erhalten. Eine kleine
  Plateau-Regel senkt dabei die Search-LR ohne die Momente zu verwerfen.
- Die Full-Data-Fidelity ist adaptiv. Faire weitere Paesse laufen, solange
  mindestens ein Finalist noch messbar lernt und beide denselben Schrittumfang
  erhalten koennen.
- Die Architekturentscheidung kombiniert Full Validation, die vorherige
  Fidelity und Rangstabilitaet ueber die Lernkurve.
- Erst danach vergleicht ein separates Recipe-Turnier Recipes auf Klonen
  desselben Architektur-Checkpoints.
- Axis-Encoder besitzen eine parameterfreie absolute Positionskodierung und
  mehrere geordnete Pooling-Bins statt eines einzigen globalen Mittelwerts.
- Der finale Trainer verwendet bei einem ausgeschoepften Versuch keinen
  Low-LR-Neustart im selben Minimum. Er startet einen unabhaengigen Versuch
  aus dem gemeinsamen NAS-Checkpoint mit neuer Recipe, neuem Seed und neuem
  Optimizer. Unterhalb des Benchmarks darf eine weitere noch ungetestete
  Recipe folgen. Das global beste Checkpoint bleibt geschuetzt.
- Liegt die Validation unter dem gelieferten Benchmark, wird verbleibende Zeit
  bevorzugt fuer diesen unabhaengigen Versuch eingesetzt. Das Competition-
  Zeitlimit und die Prediction-Reserve bleiben harte Grenzen.

Dieses Dokument beschreibt die aktive Competition-Submission. Änderungen an
Datenverarbeitung, Suchraum, Controller, Training oder Packaging müssen hier
im selben Arbeitsschritt dokumentiert werden.

## 1. Ziel und Competition-Schnittstelle

Für jedes unbekannte Dataset muss die Submission innerhalb eines variablen
Zeitlimits:

1. NumPy-Daten in DataLoader überführen,
2. eine Architektur suchen,
3. das gewählte Modell trainieren,
4. für sämtliche Testbeispiele Vorhersagen erzeugen.

Der offizielle Evaluator erwartet:

- `DataProcessor.process() -> train_loader, valid_loader, test_loader`
- `NAS.search() -> torch.nn.Module`
- `Trainer.train() -> torch.nn.Module`
- `Trainer.predict(test_loader) -> predictions`

Die aktive Lösung ist deshalb ein hierarchisches NAS-Portfolio: Während des
Search werden mehrere unabhängige Kandidaten verglichen, anschließend wird
genau ein Modell zurückgegeben und final trainiert. Es gibt kein DARTS, kein
einspace, kein Supernet-Weight-Sharing und kein finales Ensemble.

Alle Entscheidungen basieren auf Dateninhalt, Tensorform, Klassenverteilung,
gemessener Laufzeit und Validation-Ergebnissen. Dataset-Codenamen werden nicht
verwendet.

## 2. Motivation der Überarbeitung

Der vorherige Controller hatte vier zentrale Schwächen:

- Eine einzelne boolesche Categorical-Grid-Erkennung entschied, ob überhaupt
  ein Axis-Modell gesucht wurde. LaMelo fiel wegen seiner niedrigen
  One-Hot-Spaltenquote aus diesem Pfad, obwohl die andere Achse strukturiert
  sein kann.
- Der aktive Suchraum bestand ansonsten nur aus homogenen Residual-CNNs mit
  Global Average Pooling. Damit fehlten positionssensitive, achsenorientierte
  und effizient factorisierte Alternativen.
- Zero-Cost-Proxies beeinflussten zu stark, welche Kandidaten labelbasiertes
  Training erhielten. Die Search-Zeit und Datenabdeckung waren sehr klein.
- Auf Gutenberg beendete Early Stopping das Training bereits nach 28 Epochen,
  während die Learning Rate noch fast am Maximum lag und über 25 Minuten
  ungenutzt blieben.

Die aktuelle Architektur behebt diese Fehler auf Controller-Ebene, ohne auf
einen unbeschränkten Grammar-Search oder ein differentielles Supernet
umzubauen.

## 3. Gesamtpipeline

```text
Raw arrays
    |
    v
DataProcessor
  - mehrere Repräsentationshypothesen
  - Normalisierung und konservative Augmentierung
  - Klassenstatistik und Loader
    |
    v
Hierarchischer NAS-Controller
  - kleines Portfolio starker Familien
  - Architektur + Trainingsrezept
  - Proxy-Pre-Screen ohne harte Gate-Wirkung
  - labelbasierte Zeitquanten
  - Full-Validation Champion/Challenger
    |
    v
Ein warm gestartetes Gewinnermodell
    |
    v
Trainer
  - gemessener Durchsatz
  - Warmup, Reduce-on-Plateau, Zeit-Cooldown
  - bestes Checkpoint
  - sichere Prediction-Reserve
```

## 4. Mehrhypothesen-Datenprofil

`helpers.inspect_data_properties()` analysiert höchstens 256 deterministisch
gezogene Trainingsbeispiele. Ermittelt werden:

- tatsächliche Kanalzahl, Höhe und Breite,
- Wertebereich, Mittelwert, Standardabweichung und räumliche Varianz,
- Grayscale-, Small-, Square- und Standardized-Eigenschaften,
- Anteil binärähnlicher Werte und Aktivierungsdichte,
- Anteil von Spalten mit genau einer Aktivierung,
- Anteil von Zeilen mit genau einer Aktivierung,
- Klassenungleichgewicht und geglättete inverse Klassenhäufigkeiten.

Statt eines einzigen Routes werden Konfidenzen ausgegeben:

| Hypothese | Bedeutung |
|---|---|
| `spatial` | Lokale zweidimensionale Struktur bleibt immer möglich |
| `position_sensitive` | Absolute oder grobe Position kann relevant sein |
| `sequence_width` | Zeilen/Kanäle sind Features, Spalten bilden die Sequenz |
| `sequence_height` | Spalten/Kanäle sind Features, Zeilen bilden die Sequenz |
| `factorized` | Effiziente separable räumliche Verarbeitung ist plausibel |

Eine starke Spaltenstruktur liefert daher nicht nur einen harten Route-Wert.
Sie erhöht die Width- und schwächer auch die Height-/Dual-Axis-Hypothese.
Dasselbe gilt symmetrisch für zeilencodierte Daten. Eine räumliche Alternative
bleibt immer erhalten.

## 5. Datenverarbeitung

### 5.1 Normalisierung und Speicher

- NumPy-Arrays werden mit `torch.from_numpy` eingebunden.
- Fehlende Kanaldimensionen werden ergänzt.
- Tatsächliche Tensorformen ersetzen unzuverlässige nominale Metadaten.
- Channel-Mittelwerte und Standardabweichungen werden deterministisch auf
  höchstens 10.000 Beispielen berechnet.
- Testdaten werden weder geshuffelt noch abgeschnitten.

### 5.2 Augmentierung

Augmentierung bleibt konservativ:

- Starke Sequence-/Grid-Hypothesen erhalten keine geometrische Transformation.
- Sehr kleine Inputs werden nur normalisiert.
- Strukturierte Grayscale-Daten erhalten eine kleine Translation nur mit
  Wahrscheinlichkeit 0,5; Identity-Beispiele bleiben erhalten.
- Natural-Image-artige Daten können labelgeprüfte Flips, Random Crop und bei
  größeren Inputs vorsichtiges Random Erasing erhalten.
- Rotation wird für strukturierte Inputs nicht verwendet.

Die Search-Rezepte variieren zusätzlich Regularisierung, Label Smoothing,
MixUp und gegebenenfalls Class Weighting. Damit ist Augmentation nicht mehr
die einzige Regularisierungsquelle.

## 6. Architekturportfolio

Eine Spezifikation enthält weiterhin wenige robuste Makrodimensionen:

- zwei bis vier Stages, soweit die Auflösung dies erlaubt,
- 16, 32 oder 64 Initialkanäle,
- ein bis drei Blöcke pro Stage,
- Basic- oder Bottleneck-Konfiguration,
- Kernel 3 oder 5,
- optionale Squeeze-and-Excitation,
- Stem-Kernel 3, 5 oder 7,
- Modellfamilie.

Aktive Familien:

| Familie | Zweck |
|---|---|
| `spatial` | bewährter Residual-CNN-Sicherheitsanker mit Global Pooling |
| `spatial_pyramid` | kombiniert globales Pooling mit grobem 2x2-Layout |
| `factorized` | depthwise-separable Residualblöcke mit GroupNorm |
| `axis_width` | 1D-Residualpfad entlang der Breite plus Zeilenfrequenzen |
| `axis_height` | symmetrischer 1D-Pfad entlang der Höhe |
| `dual_axis` | fusioniert unabhängige Width- und Height-Encoder |

Spatial, Spatial Pyramid und Factorized sind standardmäßig aktiv. Die
Axis-Familien werden anhand der Hypothesen aktiviert. Dadurch bleibt das
Portfolio auf gewöhnlichen Bildern kompakt und wird bei strukturierten Inputs
gezielt breiter.

GroupNorm in factorisierten und Axis-Modellen reduziert die Abhängigkeit von
unbekannten oder OOM-bedingt verkleinerten Batchgrößen. Alle Kandidaten sind
eigenständige `nn.Module`-Objekte. Parameterzahlen werden analytisch berechnet.

## 7. Trainingsrezepte

Hinweis: Die unten noch beschriebene gemeinsame Auswahl ist historisch. Aktiv
ist die oben dokumentierte Trennung aus neutralem Architekturvergleich und
anschliessendem Recipe-Turnier auf identischen Checkpoint-Klonen.

Architecture Search und Trainingspolicy werden in einem kleinen,
kontrollierten Produktraum gemeinsam gewählt:

| Rezept | Eigenschaften |
|---|---|
| `stable` | AdamW, mittleres Weight Decay, wenig Smoothing |
| `regularized` | mehr Weight Decay/Smoothing, vorsichtiges MixUp |
| `balanced` | geglättete Class Weights bei erkennbarer Imbalance |
| `fast_fit` | etwas höhere LR und weniger Regularisierung bei vielen Daten |

Es werden nie alle denkbaren Hyperparameter kombiniert. Je nach
Klassenverteilung sind höchstens drei Rezepte aktiv. Das hält den Search
berechenbar und vermeidet dataset-spezifisches Tuning.

Das gewählte Rezept wird am Gewinnermodell gespeichert und vom separaten
`Trainer` übernommen.

## 8. Hierarchischer Multi-Fidelity-Controller

### 8.1 Zeit-Tiers

| Tier | Zeit | Search |
|---|---:|---|
| 1 | mindestens 15 Minuten | 54 Pre-Screen-Kandidaten, 12 labelbasierte Finalisten |
| 2 | 5 bis 15 Minuten | 30 Pre-Screen-Kandidaten, 7 Finalisten |
| 3 | unter 5 Minuten | direkter Portfolio-Anker |

Tier 1 darf höchstens 18 Prozent beziehungsweise 360 Sekunden verbrauchen.
Tier 2 darf höchstens 12 Prozent beziehungsweise 120 Sekunden verbrauchen.
Eine harte Restzeitgrenze schützt finales Training und Prediction.

### 8.2 Portfolio-Anker und Sampling

Aktiv sind drei Groessenanker pro Familie und Groessenstrata in den Quoten;
die folgende Beschreibung der Recipe-Rotation ist historisch.

Jede aktive Familie erhält zuerst einen deterministischen, mittelgroßen
Sicherheitsanker. Weitere Spezifikationen werden familienbalanciert und über
Größenquartile gezogen. Kandidaten sind duplikatfrei.

Vor der ersten Fidelity-Runde werden die verfügbaren Plätze explizit
gleichmäßig auf die im Proxy-Screen erreichten Familien verteilt. Bei drei
Familien und zwölf Plätzen erhält jede vier Kandidaten; bei sechs Familien
erhält jede zwei. Nur wenn eine Familie ihr Kontingent wegen Parameter- oder
Laufzeitfiltern nicht füllen kann, werden Restplätze global nach Proxy-Rang
vergeben.

Die Trainingsrezepte rotieren innerhalb jeder Familie unabhängig. Dadurch ist
beispielsweise Spatial nicht fest mit `stable` und Factorized nicht fest mit
`regularized` gekoppelt. Wenn das Familienkontingent groß genug ist, erreicht
jede Familie die labelbasierte Runde unter mehreren Rezepten.

Parametergrenzen dienen ausschließlich der Machbarkeit. Die Competition
bestraft Parameterzahl nicht direkt; sie ist daher kein Accuracy-Proxy.

### 8.3 Rolle der Zero-Cost-Proxies

Aktiv ist der Proxy ausschliesslich im Pre-Screen. Ab dem labelbasierten
Training besitzt er keinen Utility-Tie-Break mehr.

SynFlow, Jacobian-Correlation und NASWOT sehen denselben Calibration-Batch.
Gemessene Proxy-Latenz wird als weiteres schwaches Signal verwendet.

Die Proxies sind nur ein Pre-Screen. Vor dem labelbasierten Training stellt der
Controller sicher, dass:

- jede im Pre-Screen erreichte Modellfamilie vertreten ist,
- jedes aktive Trainingsrezept vertreten ist,
- übrige Plätze nach Proxy-Rang aufgefüllt werden.

Damit kann ein Proxy-Bias keine vollständige Repräsentationsfamilie mehr
eliminieren.

### 8.4 Labelbasierte Fidelity-Runden

Aktiv sind erhaltene AdamW-Zustaende, adaptive faire Full-Data-Paesse und die
robuste Kombination aus Full Validation, vorheriger Fidelity und Rangstabilitaet.
Der unten genannte feste Drei-Pass-Deckel ist historisch.

Bis zu 128 Trainingsbatches beziehungsweise 128 MiB werden einmal
materialisiert. Alle Kandidaten sehen exakt dieselben Daten.

Pro Runde erhält jeder Kandidat:

- eine maximale Datenabdeckung,
- zusätzlich ein gleiches Wall-Clock-Quantum,
- dieselbe klassenbalancierte Validation-Teilmenge.

Langsame Kandidaten erhalten damit nicht unbegrenzt mehr Suchzeit, schnelle
Kandidaten können im gleichen Quantum mehr Updates durchführen. Utility
kombiniert:

- Accuracy,
- bei Imbalance einen kleinen Balanced-Accuracy-Anteil,
- Learning-Curve-Gewinn,
- gemessene Schrittzeit und projizierte finale Epochen,
- Proxy-Rang nur als sehr kleinen Tie-Breaker.

Die letzten beiden Kandidaten werden auf derselben vollständigen Validation
verglichen. Teilmengen- und Full-Validation-Werte werden nicht direkt
gegeneinander gerankt. Das Gewinnermodell behält seine Gewichte.

Vor diesem Vergleich wird die nach den breiten Fidelity-Runden noch freie
Search-Zeit auf beide Finalisten gleich verteilt. Sie trainieren in derselben
deterministischen Reihenfolge auf dem vollständigen Trainingssatz statt auf
dem Cache. Pro Kandidat sind höchstens drei zusätzliche vollständige
Durchläufe erlaubt; die Wall-Clock-Grenze bleibt vorrangig. Dadurch erhöht die
Finalrunde Datenabdeckung und Warm-Start-Qualität, ohne das geschützte finale
Trainingsbudget anzutasten.

## 9. Finaler Trainer

### 9.1 Durchsatz und Zeitreserve

Vor dem Training misst der Trainer reale Train- und Validation-Schrittzeiten.
Aus der exakten Zahl der Testbatches wird eine Prediction-Reserve zwischen 30
und 180 Sekunden abgeleitet. Vor jeder Epoche und periodisch innerhalb einer
Epoche wird die Competition-Clock geprüft.

### 9.2 Optimierung

- AdamW mit dem gewählten Rezept,
- batchgrößenabhängige Base-LR,
- Label Smoothing zwischen 0,03 und 0,10,
- optional MixUp oder Class Weights,
- AMP auf CUDA,
- Gradient Clipping bei Norm 5.

### 9.3 Learning-Rate-Steuerung

Die folgende Fine-Tuning-Restart-Beschreibung ist historisch. Aktiv sind
unabhaengige Versuche vom gemeinsamen NAS-Checkpoint mit neuer Recipe, neuem
Seed und neuem Optimizer; das global beste Checkpoint bleibt geschuetzt.

Nach einem kurzen linearen Warmup steuert
`ReduceLROnPlateau` die Learning Rate. Plateaus führen zuerst zu echten
LR-Absenkungen; sie beenden das Training nicht bei hoher LR.

Parallel begrenzt ein monoton fallender, wall-clock-basierter Cosine-Cap die
maximal zulässige LR. Selbst bei ständig leicht steigender Validation gelangt
das Training dadurch vor Ablauf des Budgets in eine Fine-Tuning-Phase. Ein
später neu geschätzter Epochenhorizont kann die LR nicht wieder erhöhen.

Erreicht der erste Optimierungsversuch nach mindestens zehn Epochen ohne
Verbesserung den LR-Floor, wird genau einmal das beste Checkpoint geladen.
Optimizer-Momente werden verworfen, Weight Decay halbiert und ein
kontrollierter Fine-Tuning-Versuch mit höchstens acht Prozent der Base-LR
gestartet. Dieser protokollierte Neustart darf die LR einmalig anheben, bleibt
aber unter dem wall-clock-basierten Cap. Erreicht auch der zweite Versuch den
Floor ohne Verbesserung, endet das Training früh. Das beste Validation-
Checkpoint schützt in beiden Fällen vor Regression.

### 9.4 OOM-Verhalten

- OOM-Kandidaten werden im Search verworfen.
- OOM während der Trainer-Kalibrierung halbiert die Batchgröße bis mindestens
  vier.
- Ein später OOM kann ebenfalls einen kleineren Loader auslösen.
- Prediction-Batches werden bei OOM rekursiv geteilt.

Ein ungewöhnlich großes verborgenes Dataset soll dadurch schlechter, aber
nicht vollständig fehlschlagen.

## 10. Sicherheitsinvarianten

1. Keine Entscheidung verwendet den Dataset-Codenamen.
2. NAS gibt genau ein PyTorch-Modell zurück.
3. Es gibt kein finales Ensemble, DARTS, einspace oder Supernet.
4. Testdaten werden vollständig und in fester Reihenfolge vorhergesagt.
5. Search- und Prediction-Zeit sind explizit geschützt.
6. Jede aktive Familie erhält nach Möglichkeit labelbasierte Evidenz.
7. Full Validation entscheidet nur gegen Full Validation.
8. Das beste Checkpoint wird vor Prediction wiederhergestellt.
9. LR steigt innerhalb eines Optimierungsversuchs nicht wieder an; erlaubt ist
   ausschließlich der einmalige protokollierte Fine-Tuning-Neustart vom besten
   Checkpoint.
10. Fehlende Proxywerte oder einzelne OOM-Kandidaten stoppen nicht die
    gesamte Pipeline.

## 11. Validierung

Lokal geprüft:

- Python-Syntax aller Submission-Dateien,
- Zeilen- und Spalten-One-Hot-Fingerprints,
- Aktivierung der passenden Axis-Hypothesen,
- Portfolio-Größe auf natürlichen und strukturierten Inputs,
- exakte analytische Parameterzahl für 36 Kombinationen über alle sechs
  Modellfamilien,
- unveränderte Competition-API,
- Archivstruktur ohne Evaluator oder Testdateien.

Ein vollständiger Accuracy- und CUDA-Lauf bleibt in Colab beziehungsweise der
Competition-Umgebung erforderlich, da die lokale Python-Installation kein
PyTorch enthält.

## 12. Offene empirische Kalibrierung

Die folgenden Werte sind robuste Startpunkte und müssen mit mehreren Seeds
auf historischen Datasets überprüft werden:

- Hypothesen-Schwellen für Axis-Familien,
- Anteil des Search-Budgets,
- Zahl der labelbasierten Finalisten,
- MixUp-Stärke,
- Gewichte der Utility-Komponenten,
- Plateau-Patience und LR-Reduktionsfaktor.

Die relevante Auswertung ist Leave-one-Dataset-out: Regeln werden auf mehreren
historischen Datasets gewählt und jeweils auf einem nicht zur Auswahl
verwendeten Dataset bewertet. Optimiert werden Median, Worst Case und
Fehlerrate, nicht der Codename eines einzelnen Tests.

## 13. Änderungsprotokoll

### 30. Juli 2026 - Controller-Nachschärfung nach Teilrun

- Gleichmäßige Familienquoten für die erste labelbasierte Runde eingeführt.
- Trainingsrezepte innerhalb jeder Familie unabhängig rotiert.
- Ungenutzte Search-Zeit in eine faire Full-Data-Finalistenrunde investiert.
- Einmaligen Fine-Tuning-Neustart vom besten Checkpoint ergänzt.
- Erfolgsloses zweites Low-LR-Plateau beendet Training früh.

### 30. Juli 2026 - Hierarchisches Portfolio

- Boolesches Routing durch mehrere Repräsentationshypothesen ersetzt.
- One-Hot-Struktur auf beiden Achsen ergänzt.
- Spatial-Pyramid-, Factorized-, Height-Axis- und Dual-Axis-Familien ergänzt.
- GroupNorm für factorisierte und achsenorientierte Modelle eingeführt.
- Architecture/Recipe-Suche mit Stable-, Regularized-, Balanced- und
  Fast-Fit-Rezepten ergänzt.
- Zero-Cost-Proxies auf Pre-Screen/Tie-Break beschränkt.
- Familien- und Rezeptabdeckung vor labelbasiertem Halving garantiert.
- Gleiche Wall-Clock-Quanten und stabilen Validation-Vergleich eingeführt.
- Search-Budget auf maximal 18 Prozent erhöht.
- Cosine-Horizon/Early-Stopping durch Plateau-Scheduler plus monotonen
  Zeit-Cooldown ersetzt.
- Class Weights, MixUp und OOM-Fallback integriert.

### 30. Juli 2026 - Vorherige Compute-Korrekturen

- Duplikatfreies Sampling und feste Calibration-Batches.
- Korrigierte Proxy-Implementierungen.
- Warm-Start des Search-Gewinners.
- Reale Durchsatzmessung und Prediction-Reserve.
- AMP, Gradient Clipping und bestes Validation-Checkpoint.
