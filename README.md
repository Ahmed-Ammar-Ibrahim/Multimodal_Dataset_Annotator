# Multimodal Emotion Annotator

## Run on Windows/macOS/Linux

1. Install Python 3.10 or newer.
2. Open a terminal in this folder and run:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python multimodal_emotion_annotator.py
```

Open the local Gradio link shown in the terminal. Data is saved beside the script in `emotion_annotation_data/`: videos, `annotations.sqlite3`, and exported files.

## Google Colab

Run these cells:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content
!pip install -q "gradio>=4.44,<7"
import os
os.environ['EMOTION_ANNOTATION_ROOT'] = '/content/drive/MyDrive/emotion_annotation_data'
!python /content/multimodal_emotion_annotator.py
```

Upload `multimodal_emotion_annotator.py` to `/content` first, or use the file upload panel. Colab will print a public Gradio URL. The database, uploaded videos, and exports remain in Google Drive.

## Workflow

Add a video in **Videos**, select it, then create timestamped segments in **Annotations**. Enter transcript, independently select visual/audio/text emotion, rate intensity and confidence from 1–5, and provide an annotator ID. Click a table row or enter its ID to edit/delete. **Export** creates CSV or JSON files.

Emotion codes: `NEU`, `HAP`, `SAD`, `ANG`, `FEA`, `DIS`, `SUR`, `UNCERTAIN`.

While watching, use **Mark Start from video** and **Mark End from video** to capture the exact playback time, including fractional seconds. Use the nudge buttons for 0.1-second adjustments. Saving a new segment creates `dataset_segments/segment_XXXX/video.mp4`, `audio.wav`, and `text.txt`, in addition to the database row.

Install FFmpeg and put it on your PATH to enable clip extraction. On Windows, install it with `winget install Gyan.FFmpeg` or download a build from ffmpeg.org. Without FFmpeg, annotation and database saving still work, but media clips are not extracted.
