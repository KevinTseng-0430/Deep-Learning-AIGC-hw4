# Crested Myna (Acridotheres cristatellus) Recognizer

Streamlit web app for browsing and analyzing crested myna images, with optional PyTorch model loading and training support.

## What’s included
- `streamlit_app.py`: UI for uploads or `data/` browsing, dataset analytics, confidence charts, resolution vs confidence plots, gallery with downloads.
- `app_utils.py`: image loading, metadata extraction, stub predictor, dataset stats, and auto-detecting a PyTorch checkpoint in `models/`.
- `train_pytorch.py`: optional PyTorch training script (ResNet/EfficientNet backbones) with augmentation, early stopping, checkpoints, and CLI args.
- `assets/style.css`: custom styling.
- `requirements.txt`: dependencies.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run the Streamlit app
```bash
streamlit run streamlit_app.py
```
- Place images under `data/crested_myna/` and `data/other/`, or upload via the UI.
- Checkpoints in `models/` ending with `.pt`/`.pth` are auto-detected; otherwise a demo predictor is used.

## Train a model (PyTorch, optional)
```bash
python train_pytorch.py --data_dir ./data --arch resnet50 --epochs 12 --batch_size 32 --output models/crested_myna_model.pth --use_cuda
```
Key flags:
- `--arch` choices: `resnet18`, `resnet34`, `resnet50` (default), `resnet101`, `efficientnet_b0`
- `--lr`, `--weight_decay`, `--patience`, `--val_split`, `--num_workers`
- Saves checkpoint with class mapping and best weights by validation accuracy.
Notes:
- The app runs without a trained checkpoint (falls back to a demo predictor).
- Use the script only if you want to fine-tune or refresh a model; place outputs in `models/`.

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
