# stereo_from_depth.py — Anleitung

Dieses Dokument beschreibt, wie du das kleine Hilfs-Skript `stereo_from_depth.py` verwendest, das aus einem einzelnen RGB-Bild und einer zugehörigen Depth-Map ein Stereo-Paar (linkes und rechtes Auge) erzeugt.

Inhalt
- Zweck
- Voraussetzungen
- Installation (Conda empfohlen)
- Alternative Installation mit pip
- Ausführung / Parameter
- Beispiele
- Troubleshooting
- Hinweise zur Anpassung

## Zweck

`stereo_from_depth.py` ist ein leichtgewichtiges Werkzeug, das aus einer RGB-Aufnahme und einer Depth-Map (z.B. `.npy` oder Bild) zwei neue Bilder generiert: `*_left.png` und `*_right.png`. Die Methode ist einfach: es wird eine invertierte, normalisierte Depth-Map in eine horizontale Disparität (Pixelversatz) übersetzt und das Bild entsprechend remapped. Löcher durch disocclusion werden mit OpenCV-Inpainting gefüllt.

## Voraussetzungen

- Python 3.11 empfohlen (funktioniert auch mit 3.10). Auf Apple Silicon (M1/M2/M4) ist eine conda/miniforge-Umgebung empfohlen, weil dort native arm64-Pakete verfügbar sind.
- Benötigte Python-Pakete (aktualisiert in `requirements.txt`):
  - numpy (>=1.24)
  - opencv-python (>=4.7.0) oder `opencv` aus conda-forge
  - scikit-image, vispy, moviepy, transforms3d, networkx, cynetworkx (falls du Teile des Repos zusätzlich nutzt)

## Installation (empfohlen: conda / conda-forge)

1. Öffne ein Terminal und aktiviere/erstelle deine conda-Umgebung (beispielhaft python 3.11):

```bash
conda create -n stereo python=3.11 -y
conda activate stereo
```

2. Installiere die benötigten Pakete (conda-forge liefert stabile arm64-Binaries auf Apple Silicon):

```bash
conda install -c conda-forge numpy=1.24 opencv=4.7 scikit-image moviepy transforms3d vispy networkx -y
python -m pip install cynetworkx
```

3. (Optional) Alternativ kannst du die komplette `requirements.txt` mit pip installieren:

```bash
# Falls du pip bevorzugst, zuerst sicherstellen, dass pip up-to-date ist
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Hinweis: Auf Apple Silicon ist `conda install -c conda-forge opencv` oft zuverlässiger als `pip install opencv-python`, weil conda native arm64-Builds bereitstellt.

## Ausführung / Parameter

Wechsle ins Projektverzeichnis und rufe das Skript auf. Beispiel (verwende die im Repo enthaltenen Dateien):

```bash
cd /path/to/3d-photo-inpainting-for-Video/3d-photo-inpainting-for-Video
python3 stereo_from_depth.py --image image/moon.jpg --depth depth/moon.png --out_dir out --max_shift 40 --fill_method inpaint
```

Parameter
- `--image` (string, required): Pfad zur RGB-Datei (z. B. `image/moon.jpg`). Unterstützte Formate: jpg, png.
- `--depth` (string, required): Pfad zur Depth-Map. Unterstützte Formate:
  - `.npy` (NumPy-Array)
  - Bildformate wie `.png`/`.jpg` (als Graustufen interpretiert)
- `--out_dir` (string, default `out`): Zielverzeichnis für die Ausgabedateien.
- `--max_shift` (float, default `40`): Maximale horizontale Verschiebung in Pixeln, die für die nächsten Objekte angewendet wird. Größere Werte → stärkere Stereo-Effekte, aber mehr Löcher.
- `--fill_method` (choice: `inpaint`|`morph`, default `inpaint`): Methode zum Füllen der Löcher nach dem Warping. `inpaint` verwendet OpenCV-Inpainting (optisch besser), `morph` benutzt morphologische Schließung (schneller, weniger akkurat).

Ausgabe
- Das Skript schreibt zwei PNG-Dateien in `--out_dir`: `<basename>_left.png` und `<basename>_right.png`.

## Beispiele

- Subtiler Stereo-Effekt:

```bash
python3 stereo_from_depth.py --image image/moon.jpg --depth depth/moon.npy --out_dir out --max_shift 20
```

- Starker Stereo-Effekt (mehr Löcher möglich):

```bash
python3 stereo_from_depth.py --image image/moon.jpg --depth depth/moon.npy --out_dir out --max_shift 80 --fill_method inpaint
```

## Troubleshooting

- Depth `.npy` zu groß / ungewöhnliche Werte:
  - Prüfe die Statistik:

```bash
python3 - <<'PY'
import numpy as np
d = np.load('depth/moon.npy')
print('dtype', d.dtype, 'min', d.min(), 'max', d.max(), 'shape', d.shape)
PY
```
  - Wenn Werte nicht im erwarteten Bereich liegen, normalisiere und speichere als PNG:

```bash
python3 - <<'PY'
import numpy as np, cv2
d = np.load('depth/moon.npy').astype('float32')
d = d - d.min()
d = d / max(1e-8, d.max())
cv2.imwrite('depth/moon_converted.png', (d*255).astype('uint8'))
print('wrote depth/moon_converted.png')
PY
```
  - Verwende dann `--depth depth/moon_converted.png`.

- Auflösungs-Inkompatibilität (Bild und Depth unterschiedliche Dimensionen):
  - Das Script versucht, die Depth-Map automatisch auf die Bildgröße zu skalieren. Du kannst sie aber vorher manuell anpassen:

```bash
python3 - <<'PY'
import cv2
img = cv2.imread('image/moon.jpg')
d = cv2.imread('depth/moon_converted.png', cv2.IMREAD_UNCHANGED)
d2 = cv2.resize(d, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
cv2.imwrite('depth/moon_resized.png', d2)
print('wrote depth/moon_resized.png')
PY
```

- Probleme beim Installieren von `opencv-python` auf Apple Silicon:
  - Nutze statt `pip` die conda-Variante: `conda install -c conda-forge opencv`.
  - Alternativ: `pip install --pre opencv-python` (kann experimentelle Wheels nutzen).

## Hinweise zur Anpassung

- Die aktuelle Disparitätszuordnung ist proportional zur normalisierten inversen Depth (nächste Punkte bekommen größte Pixelverschiebung). Für realistischere Ergebnisse kann man Kameraparameter (FOV, Baseline) berücksichtigen und Disparität physikalisch berechnen.
- Wenn Löcher stören, kannst du Depth vorab glätten (z. B. `cv2.GaussianBlur`) oder `--max_shift` reduzieren.

## Kontakt

Wenn du möchtest, erweitere ich das Skript um:
- automatisierten Testlauf + CI-Skript
- einstellbare Kamera-Parameter (Baseline, focal length)
- bessere Hole-Filling-Methoden (PatchMatch / exemplar-based)

Viel Erfolg! Falls du möchtest, erstelle ich noch eine kleine Testdatei `run_stereo_test.sh` im Repo, die die oben genannten Schritte automatisiert.

Beispiel-Skript
---------------

Im Repo gibt es jetzt `run_video_example.sh`, ein kleines Shell-Skript, das zeigt, wie du aus `depth/moon.npy` eine Depth-Frame-Sequenz erzeugst und `stereo_from_depth_video.py` im Pipe-Modus ausführst. Stelle sicher, dass das Skript ausführbar ist:

```bash
chmod +x run_video_example.sh
./run_video_example.sh
```

