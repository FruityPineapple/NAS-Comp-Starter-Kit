# Architektur des NAS

Stand: 30. Juli 2026

Dieses Dokument ist die verbindliche Architekturbeschreibung der Competition-
Submission. Bei künftigen Änderungen an Datenverarbeitung, Search Space,
Architecture Search, Training, Zeitsteuerung oder Packaging muss es im selben
Änderungsschritt aktualisiert werden.

## 1. Ziel und Randbedingungen

Die Submission muss für ein vollständig unbekanntes Bildklassifikations-
Dataset innerhalb eines dataset-spezifischen Zeitlimits:

1. die Daten analysieren und DataLoader erzeugen,
2. eine geeignete Architektur suchen,
3. das ausgewählte Modell trainieren,
4. für alle Testbeispiele Vorhersagen erzeugen.

Zur Laufzeit sind insbesondere Auflösung, Kanalzahl, Klassenanzahl,
Datenverteilung, Bildsemantik und Zeitbudget unbekannt. Die Pipeline darf
deshalb weder auf Dataset-Codenamen noch auf fest bekannte Datasets
spezialisiert sein.

Der Competition-Score normalisiert die Test-Accuracy relativ zu einem
dataset-spezifischen Benchmark. Ein Fehler oder Time-out ist besonders teuer.
Zuverlässige Vorhersagen und eine sichere Prediction-Reserve haben daher
Vorrang vor einem maximal umfangreichen Search.

## 2. Erkenntnisse aus der initialen Analyse

Die erste Version verwendete einen zufälligen Macro-Search über 648
ResNet-artige Architekturen:

- 200 Kandidaten wurden mit Zurücklegen gesampelt.
- SynFlow, Jacob-Covariance und NASWOT bildeten ein Borda-Ranking.
- Die zehn besten Kandidaten wurden jeweils für zwei kurze Epochen trainiert.
- Der Kandidat mit der höchsten kurzfristigen Validation-Accuracy wurde neu
  initialisiert und anschließend vollständig trainiert.

Die Auswertung zeigte mehrere strukturelle Schwächen:

- Sampling mit Zurücklegen erzeugte im Mittel ungefähr 28 Duplikate pro
  Search. Dieselbe Architektur wurde tatsächlich mehrfach evaluiert.
- Verschiedene Kandidaten sahen unterschiedliche zufällige Batches und
  Augmentierungen. Ihre Proxy- und Learning-Curve-Werte waren daher nicht
  direkt vergleichbar.
- Zero-Cost-Proxies, insbesondere SynFlow, bevorzugten große Modelle.
- Parameterzahl und reale Trainingsgeschwindigkeit waren nicht Teil der
  Auswahlentscheidung.
- Auf demselben Dataset wurden dadurch einmal ungefähr 48 Millionen und
  einmal 1,91 Millionen Parameter ausgewählt. Das große Modell schaffte nur
  wenige Epochen; das kleinere Modell erreichte eine wesentlich höhere
  Accuracy.
- Die im Low-Fidelity-Search trainierten Gewichte wurden verworfen.
- Die Epochenzeit wurde mit einer festen Heuristik geschätzt.
- Der LR-Warmup erreichte die konfigurierte Base-LR nicht.
- Der Cosine-Scheduler lief über seinen geplanten Horizont hinaus und ließ die
  Learning Rate nach dem Minimum wieder steigen.
- Die Stoppbedingung reservierte Zeit für zwei weitere Epochen und ließ bei
  langsamen Modellen mehrere Minuten ungenutzt.
- Augmentierungen wurden aus wenigen, teilweise skalenabhängigen Heuristiken
  abgeleitet.

Die aktuelle Architektur wurde so entworfen, dass diese Fehlerklassen
systematisch vermieden werden.

## 3. Gesamtpipeline

```text
Raw NumPy Arrays
        |
        v
DataProcessor
  - Datenfingerprint
  - tatsächliche C/H/W-Dimensionen
  - Normalisierung
  - label-aware Augmentierungswahl
  - DataLoader
        |
        v
NAS
  - Zeit-Tier bestimmen
  - Search Space enumerieren
  - Parametergrenzen anwenden
  - eindeutige Kandidaten auswählen
  - feste Proxy-Batches
  - Successive Halving
  - Compute/Accuracy-Ranking
        |
        v
Warm gestartetes Gewinnermodell
        |
        v
Trainer
  - realen Durchsatz messen
  - sichere Epochenzahl bestimmen
  - Warmup + monotones Cosine Decay
  - AMP-Training
  - bestes Validation-Checkpoint
        |
        v
Prediction
```

Die aktive Pipeline benötigt nur PyTorch, torchvision und NumPy.

## 4. Zeitsteuerung

Der Evaluator stellt eine `Clock` mit der verbleibenden Zeit bereit. Der NAS
arbeitet in drei Tiers:

| Tier | Verbleibende Zeit | Strategie |
|---|---:|---|
| 1 | mindestens 15 Minuten | 72 Kandidaten, vollständiger Proxy-Screen und mehrstufiges Successive Halving |
| 2 | 5 bis 15 Minuten | 32 Kandidaten, verkürzter Search |
| 3 | unter 5 Minuten | sofortiges, input-adaptives Fallback-Modell |

Für Tier 1 werden höchstens 12 Prozent beziehungsweise 240 Sekunden für den
Search verwendet. Tier 2 verwendet höchstens 8 Prozent beziehungsweise
75 Sekunden. Der Search besitzt eine harte Restzeitgrenze; die geschützte Zeit
wird dem finalen Training überlassen.

Der Trainer misst vor dem Training:

- Zeit pro realem Trainingsbatch einschließlich Data Loading,
- Zeit pro Validation-Batch,
- geschätzte Zeit für eine vollständige Epoche,
- notwendige Prediction-Reserve.

Aus diesen Messungen wird die sichere Zahl vollständiger Epochen abgeleitet.
Vor jeder Epoche und periodisch innerhalb einer Epoche wird die Clock erneut
geprüft. Ist kein sicheres Training mehr möglich, wird direkt zur Prediction
übergegangen.

## 5. DataProcessor

Implementierung: `data_processor.py` und `helpers.py`

### 5.1 Datenrepräsentation

- NumPy-Arrays werden nach Möglichkeit mit `torch.from_numpy` ohne zusätzliche
  vollständige Speicherkopie eingebunden.
- Nicht-Float32-Eingaben werden einmalig in Float32 konvertiert.
- Fehlende Kanaldimensionen werden von `[N,H,W]` nach `[N,1,H,W]` ergänzt.
- Für Kanalzahl und räumliche Dimensionen gelten die tatsächlichen
  Tensorformen. Nominale oder veraltete Metadata-Dimensionen werden nicht für
  den Modellbau verwendet.

### 5.2 Datenfingerprint

Auf einer deterministischen Teilmenge werden unter anderem ermittelt:

- Kanalzahl, Höhe und Breite,
- Grayscale-, Square- und Small-Flags,
- rohe und normalisierte räumliche Varianz,
- Wertebereich, Mittelwert und Standardabweichung,
- Hinweis auf bereits standardisierte beziehungsweise strukturierte Daten,
- Hinweis auf niedrigvariante RGB-Bilder,
- Klassenungleichgewicht.

### 5.3 Normalisierung

Per-Channel-Mittelwert und -Standardabweichung werden auf höchstens 10.000
gleichmäßig verteilten Trainingsbeispielen berechnet. Dadurch ist die
Berechnung deterministisch und benötigt keine Float32-Kopie des kompletten
Trainingssets.

### 5.4 Augmentierung

Augmentierungen sind bewusst konservativ:

- Eingaben bis 8x8 werden ausschließlich normalisiert.
- Für strukturierte Grayscale-Daten wird eine kleine Translation verwendet;
  Rotation und Spiegelung werden vermieden.
- Random Crop verwendet nur moderates Padding.
- Random Erasing wird ausschließlich für große, natürlich wirkende Bilder
  eingesetzt.
- Horizontal und Vertical Flip werden nicht allein aus Kanalzahl oder
  Auflösung abgeleitet.

Für Flips wird ein kleiner label-aware Invarianztest ausgeführt. Ein
Nearest-Centroid-Klassifikator vergleicht unveränderte und gespiegelte
Trainingsbeispiele. Eine Spiegelung wird nur freigegeben, wenn sie die
Klassenzuordnung weitgehend erhält. Vertical Flip bleibt zusätzlich auf
quadratische, niedrigvariante RGB-Daten beschränkt.

### 5.5 DataLoader

- Testdaten werden nie geshuffelt und nie mit `drop_last=True` geladen.
- Auf CUDA werden zwei persistente Worker und Pinned Memory verwendet.
- Auf CPU bleibt `num_workers=0` für Kompatibilität.
- Der Trainingsloader besitzt einen festen Generator-Seed.
- Die Batchgröße wird aus Inputgröße und verfügbarer GPU-RAM abgeleitet.

## 6. Search Space

Implementierung: `search_space.py`

Der Search Space enthält eigenständige residuale CNNs. Eine Architektur wird
durch folgende Dimensionen beschrieben:

| Dimension | Werte |
|---|---|
| Stages | 2, 3, 4; bei kleinen Inputs sicher begrenzt |
| Initiale Kanäle | 16, 32, 64 |
| Blöcke pro Stage | 1, 2, 3 |
| Blocktyp | BasicBlock, BottleneckBlock |
| Kernel | 3x3, 5x5 |
| SE Attention | an, aus |
| Stem-Kernel | 3x3, 5x5, 7x7 |

Damit entstehen bei vier zulässigen Stages 648 eindeutige Architekturen.

Jedes Modell besteht aus:

1. Conv-BatchNorm-ReLU-Stem,
2. zwei bis vier residualen Stages,
3. Downsampling im ersten Block jeder neuen Stage,
4. Adaptive Average Pooling,
5. größenabhängigem Dropout,
6. linearem Klassifikationskopf.

Der Search Space kann vollständig und duplikatfrei enumeriert werden. Die
Parameterzahl wird analytisch berechnet, ohne jedes der 648 Modelle bauen und
initialisieren zu müssen.

## 7. Compute-aware NAS

Implementierung: `nas.py`

### 7.1 Reproduzierbarkeit

Der Seed wird ausschließlich aus Inputkanälen, Auflösung und Klassenanzahl
abgeleitet. Damit bleibt das Verfahren unabhängig vom Dataset-Codenamen, aber
für identische Aufgaben reproduzierbar.

Python, NumPy, PyTorch und CUDA werden konsistent geseedet. Kandidaten werden
ohne Zurücklegen ausgewählt.

### 7.2 Parametergrenzen

Vor dem Proxy-Screen werden Architekturen ausgeschlossen, deren Größe nicht
zum Input und zum Zeitbudget passt:

| Kleinste Dimension | Mindestparameter | Maximalparameter |
|---:|---:|---:|
| bis 8 | 20.000 | 4.000.000 |
| bis 32 | 80.000 | 6.000.000 |
| bis 64 | 200.000 | 4.000.000 |
| größer als 64 | 150.000 | 2.500.000 |

Die Grenzen verhindern insbesondere, dass ein kurzfristig gut geranktes, aber
praktisch nicht trainierbares Modell das gesamte Zeitbudget verbraucht.

Die Kandidaten werden gleichmäßig aus Parametergrößen-Quartilen gezogen. Ein
robustes, input-adaptives Fallback-Modell wird als Anchor einbezogen.

### 7.3 Proxy-Screen

Alle Kandidaten sehen denselben gecachten Calibration-Batch. Verwendet werden:

- SynFlow mit logarithmierter Saliency,
- sample-basierte Jacobian-Correlation,
- NASWOT mit Aktivierungs- und Inaktivierungsübereinstimmung.

NASWOT akkumuliert seine Kernelbeiträge direkt in ReLU-Hooks. Dadurch werden
auch mehrfach verwendete ReLU-Module korrekt berücksichtigt, ohne riesige
Aktivierungstensoren zu konkatenieren.

Das Ranking ist ein gewichteter, tie-aware Borda-Count:

| Signal | Gewicht |
|---|---:|
| SynFlow | 0,25 |
| Jacobian-Correlation | 0,75 |
| NASWOT | 1,00 |
| Effizienz | 0,50 |

Fehlgeschlagene oder nicht-finite Proxies erhalten den schlechtesten Rang und
keine zufällige Position. Das Effizienzsignal dämpft die bekannte
Größenpräferenz von SynFlow.

### 7.4 Successive Halving

Die besten Proxy-Kandidaten werden mit identischen, bereits materialisierten
Trainingsbatches verglichen:

- Tier 1: zunächst 12 Kandidaten, danach 6, 2 und 1 Kandidat,
- Tier 2: zunächst 6 Kandidaten, danach 3 und 1 Kandidat.

Die Tier-1-Runden trainieren bis zu 32, 64 und weitere 64 Schritte. Tier 2
verwendet 20 und 40 Schritte. Die Gewichte überlebender Kandidaten bleiben
zwischen den Runden erhalten.

Die Utility kombiniert:

- Low-Fidelity-Validation-Accuracy,
- Parameterstrafe,
- gemessene Zeit pro Trainingsschritt,
- erwartete Zahl vollständiger finaler Trainingsepochen.

Wenn mindestens ein Kandidat voraussichtlich zwölf vollständige Epochen
schafft, werden langsamere Kandidaten verworfen. Dies verhindert die in der
initialen Analyse beobachtete Auswahl extrem großer Modelle.

Das Gewinnermodell wird nicht neu initialisiert. Seine Low-Fidelity-Gewichte
werden an den Trainer übergeben.

## 8. Trainer

Implementierung: `trainer.py`

### 8.1 Optimierung

- Optimizer: AdamW
- Base-LR: ungefähr `1e-3`, mit konservativer Quadratwurzel-Skalierung anhand
  der Batchgröße
- Weight Decay: `1e-2`, bei mindestens drei Millionen Parametern `2e-2`
- Loss: Cross Entropy mit Label Smoothing `0,1`
- Gradient Clipping: Norm `5,0`
- CUDA: Automatic Mixed Precision mit GradScaler

### 8.2 Learning-Rate-Schedule

Der Schedule wird in Optimizer-Schritten statt nur in Epochen definiert:

1. linearer Warmup von 10 Prozent auf exakt 100 Prozent der Base-LR,
2. monotones Cosine Decay bis zum Ende der sicheren Schritte,
3. kein Cosine-Restart und kein Wiederanstieg nach dem Minimum.

Ein vom NAS warm gestartetes Modell erhält einen kürzeren Warmup.

### 8.3 Checkpoints und Early Stopping

- Nach jeder vollständigen oder sicher beendeten Epoche wird auf dem
  Validation-Set evaluiert.
- Das beste `state_dict` wird auf CPU gesichert.
- Die Patience wird aus der sicheren Epochenzahl abgeleitet und liegt zwischen
  8 und 20 Epochen.
- Vor der Prediction wird das beste Validation-Checkpoint wiederhergestellt.
- Accuracy-Zähler werden direkt aus Tensoren berechnet; große Python-Listen
  und sklearn-Aufrufe im inneren Trainingspfad entfallen.

## 9. Fallback und Fehlertoleranz

Wenn weniger als fünf Minuten verfügbar sind, wird kein Search durchgeführt.
Statt eines festen ResNet-18 wird ein kompaktes residuales Modell gewählt,
dessen Stage-Zahl, Stem, Kernel und Attention an Auflösung und
Datenfingerprint angepasst sind.

Wenn Proxies fehlschlagen oder der Search seine Zeitgrenze erreicht, wird
entweder der beste bereits bewertete Kandidat oder das robuste Fallback
verwendet. Wenn nicht einmal eine sichere Trainingsepoche möglich ist, darf
der Trainer ohne weitere Epoche direkt Vorhersagen erzeugen.

## 10. Packaging und Competition-API

Der offizielle Evaluator erwartet:

- `DataProcessor.process() -> train_loader, valid_loader, test_loader`
- `NAS.search() -> torch.nn.Module`
- `Trainer.train() -> torch.nn.Module`
- `Trainer.predict(test_loader) -> Predictions`

`submission/main.py` ist absichtlich nicht Bestandteil der Submission. Die
Datei würde durch die Kopierreihenfolge des Makefiles den offiziellen
Evaluator überschreiben. Der Evaluator bleibt außerhalb der Submission und
wird durch die Competition bereitgestellt.

`ensemble.py` ist derzeit nur als Kompatibilitätsbaustein vorhanden. Der
aktive NAS gibt bewusst ein einzelnes compute-aware ausgewähltes Modell
zurück, weil das Training mehrerer Ensemble-Mitglieder unter kurzen,
unbekannten Zeitlimits die Robustheit reduziert.

## 11. Sicherheitsinvarianten

Folgende Eigenschaften dürfen bei künftigen Änderungen nicht unbeabsichtigt
verletzt werden:

1. Es werden ausschließlich Dateien unter `submission/` geändert.
2. Der offizielle Evaluator wird nicht mitgeliefert oder überschrieben.
3. Testdaten werden weder geshuffelt noch abgeschnitten.
4. Der Search verwendet eine begrenzte Zeit und schützt Trainingszeit.
5. Kandidaten werden duplikatfrei und auf vergleichbaren Daten bewertet.
6. Architekturgröße und gemessene Trainingszeit beeinflussen die Auswahl.
7. Die Prediction besitzt eine gemessene Sicherheitsreserve.
8. Der LR-Schedule ist nach dem Warmup monoton fallend.
9. Nicht-finite Losses oder Proxies führen nicht zum Absturz der Pipeline.
10. Entscheidungen verwenden keine Dataset-Codenamen.

## 12. Validierung

Für den aktuellen Stand wurden statisch beziehungsweise isoliert geprüft:

- Syntax aller Python-Dateien,
- vollständige Competition-API,
- 648 eindeutige Search-Space-Spezifikationen,
- Sampling von 200 Architekturen ohne Duplikate,
- analytische Parameterzählung gegen drei bekannte Evaluator-Werte,
- tie- und NaN-robustes Proxy-Ranking,
- Datenfingerprints für standardisierte, Grayscale- und niedrigvariante
  RGB-Eingaben,
- exakt erreichter Warmup-Peak und monoton fallendes Cosine Decay,
- keine geänderten Dateien außerhalb von `submission/`.

Ein vollständiger lokaler Competition-Lauf benötigt eine Umgebung mit
PyTorch, torchvision, CUDA und den Competition-Datasets. Nach jeder
wesentlichen Architekturänderung muss mindestens ein vollständiger
`make submission=submission all`-Lauf durchgeführt und dessen Accuracy,
Parameterzahl, Phasenlaufzeit, Epochenzahl und verbleibende Prediction-Zeit
verglichen werden.

## 13. Bekannte offene Punkte

- Die tatsächliche Korrelation jedes Zero-Cost-Proxys mit finaler Accuracy muss
  über mehrere historische Datasets und Seeds weiter gemessen werden.
- Parametergrenzen und Proxy-Gewichte sind robuste Startwerte, aber noch nicht
  durch eine vollständige Ablationsstudie optimiert.
- Die label-aware Flip-Erkennung ist ein günstiger Proxy für Invarianz und kann
  bei sehr vielen Klassen oder schwacher Centroid-Struktur unentschieden sein.
- Der Prediction-Zeitbedarf wird aus der Validation-Laufzeit geschätzt, da der
  Testloader dem Trainer erst bei `predict()` übergeben wird.
- Das vorhandene `ensemble.py` ist nicht Teil des aktiven Suchpfads.

## 14. Änderungsprotokoll

### 30. Juli 2026 – Compute-aware Überarbeitung

- Sampling mit Zurücklegen durch eindeutige, stratifizierte Kandidatenwahl
  ersetzt.
- Feste Calibration- und Low-Fidelity-Batches eingeführt.
- Jacobian-Correlation und NASWOT korrigiert.
- Tie- und fehlerrobustes gewichtetes Proxy-Ranking eingeführt.
- Parametergrenzen, Laufzeitmessung und projizierte Epochen in die
  Architekturwahl aufgenommen.
- Fixed Top-10-Evaluation durch Successive Halving ersetzt.
- Warm-Start des Gewinnermodells eingeführt.
- Feste Epochenzeitheuristik durch reale Durchsatzmessung ersetzt.
- Warmup und Cosine-Scheduler korrigiert.
- AMP, Gradient Clipping, bestes Checkpoint und dynamische Prediction-Reserve
  integriert.
- Tatsächliche Inputdimensionen und speicherschonende Tensorerzeugung
  eingeführt.
- Datenfingerprints und label-aware Flip-Erkennung ergänzt.
- `submission/main.py` entfernt.

