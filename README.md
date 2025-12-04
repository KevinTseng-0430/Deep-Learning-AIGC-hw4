# Crested Myna (Acridotheres cristatellus) Recognizer

Streamlit web app for crested myna image browsing, analytics, and gallery management, described via a CRISP-DM-style structure. This app runs locally, works with simple folder-based datasets, and provides confidence-style scores via a built-in heuristic when no external model is present.

## Citation / Reference
- Based on the Colab demo “Demo02 — Transfer Learning Crested Myna Recognizer”: https://colab.research.google.com/github/yenlung/AI-Demo/blob/master/%E3%80%90Demo02%E3%80%91%E9%81%B7%E7%A7%BB%E5%BC%8F%E5%AD%B8%E7%BF%92%E5%81%9A%E5%85%AB%E5%93%A5%E8%BE%A8%E8%AD%98%E5%99%A8.ipynb#scrollTo=Sftnku6_7Feg

## 1) Business Understanding
- **Goal:** Let users explore crested myna images, see confidence-style scores, and inspect dataset health in one place.
- **Users:** Field observers, students, or hobbyists who need quick, visual inspection without heavyweight tooling.
- **Outputs:** Per-image scores and metadata, dataset summaries, resolution-quality insights, and downloadable previews.
- **Constraints:** Local-first, no cloud dependency; friendly UI; runs on CPU.
- **Success signals:** Smooth uploads/browsing, meaningful charts, and fast feedback during dataset exploration.

## 2) Data Understanding
- **Expected structure:**  
  ```
  data/
  ├── crested_myna/   # positive class
  └── other/          # negative/other class
  ```
- **Supported inputs:** JPG/PNG from local folders or drag-and-drop uploads in the UI.
- **Metadata captured:** width, height, format, aspect ratio, file size (MB).
- **Built-in stats:** class counts inferred from parent folder names, image size distributions, format breakdowns, resolution buckets (<480p, 480–720p, 720–1080p, >1080p).
- **Folder-based classes:** Parent folder names are treated as class labels when browsing `data/`.

## 3) Data Preparation
- **In-app handling:** Images are read as RGB; metadata is extracted on the fly; analytics recompute whenever data changes.
- **Suggested hygiene:** Remove `.DS_Store`/`__pycache__`, keep class folders consistent, avoid corrupted or non-image files.
- **Format consistency:** Prefer PNG/JPG; very large RAW/TIFF are not targeted.
- **Balanced folders:** Keep roughly similar counts across class folders for clearer analytics.

## 4) Modeling (lightweight scoring)
- **Confidence source:** A deterministic heuristic produces a “confidence-style” score based on image dimensions; identical dimensions yield repeatable scores (range roughly 0.2–0.8).
- **External checkpoints:** If you later place a compatible checkpoint under `models/`, the app will attempt to load it automatically; otherwise it always uses the built-in heuristic. (The app itself does not depend on any ML framework.)
- **Input normalization:** Images are resized and center-cropped internally for consistent scoring when an external checkpoint is present; the heuristic uses metadata only.
- **Class mapping:** Folder names drive class-aware filtering in the gallery; scores are shown as “Crested Myna confidence” for simplicity.

## 5) Evaluation (in-app analytics)
- **Dataset overview:** total images, per-class counts, average dimensions, format breakdown, file-size stats.
- **Confidence analysis:** histogram with box overlay, mean/median/std/max, percentiles (10–99%), high/low counts, range buckets.
- **Resolution vs score:** scatter plots (width/height/aspect ratio vs confidence-style score) with trendlines; box plots by resolution category; summary tables.
- **Manual checks:** Inspect high and low score buckets; verify class folder labeling; spot tiny or extreme aspect ratio images that may degrade quality.
- **Sanity tips:** Very small, narrow, or low-quality images tend to get lower scores; ensure source images are clear and reasonably sized.

## 6) Deployment & Usage
- **Setup:**  
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
  pip install -r requirements.txt
  ```
- **Run the app:**  
  ```bash
  streamlit run streamlit_app.py
  ```
  - Sidebar toggle: choose “Upload image” or “Project data folder”.  
  - If using folder mode, set the path (default `data/`).  
  - If a compatible checkpoint is placed in `models/`, the app will try to use it; otherwise it sticks to the deterministic heuristic.
- **Files of interest:**  
  - `streamlit_app.py`: UI, tabs (Analytics, Gallery), charts (Plotly), selection logic, and scoring display.  
  - `app_utils.py`: image loading, metadata/statistics, scoring heuristic, gallery helpers, and optional checkpoint detection.  
  - `assets/style.css`: custom styling.  
  - `requirements.txt`: dependencies for the app.
- **Environment hygiene:** `.gitignore` excludes cache/OS artifacts; keep `data/` organized with class subfolders.

## User workflow
1) Launch the app (`streamlit run streamlit_app.py`).  
2) In the sidebar, pick “Upload image” or point to `data/`.  
3) Open **Analytics** tab to see dataset metrics, score distributions, and resolution correlations.  
4) Open **Gallery** tab to browse tiles, filter by class, select an image, view its score, metadata JSON, and download the PNG.  
5) Adjust dataset contents (add/remove images) and refresh to update analytics.

## Suggested data layout
```
data/
├── crested_myna/
│   ├── IMG_001.jpg
│   └── ...
└── other/
    ├── IMG_100.jpg
    └── ...
```

## Repository URL
https://github.com/KevinTseng-0430/Deep-Learning-AIGC-hw4.git


