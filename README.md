# Crested Myna (Acridotheres cristatellus) Recognizer

Streamlit app + PyTorch training pipeline for crested myna image recognition, plus LLM-ready prompt templates for future extensions (audio, TFLite, detection, active learning, explainability).  
Reference Colab demo: https://colab.research.google.com/github/yenlung/AI-Demo/blob/master/%E3%80%90Demo02%E3%80%91%E9%81%B7%E7%A7%BB%E5%BC%8F%E5%AD%B8%E7%BF%92%E5%81%9A%E5%85%AB%E5%93%A5%E8%BE%A8%E8%AD%98%E5%99%A8.ipynb

## What’s implemented
- Streamlit analytics/gallery UI (`streamlit_app.py`) with uploads, project `data/` browsing, detailed confidence analytics, resolution vs confidence plots, and downloads.
- Image utilities (`app_utils.py`) for loading, metadata extraction, stub predictions, dataset stats, and model auto-detection.
- PyTorch training script (`train_pytorch.py`) with transfer learning (ResNet18/34/50/101, EfficientNet-B0), augmentations, early stopping, checkpoints, and CLI args.
- Styling (`assets/style.css`) and dependency list (`requirements.txt`).

## What’s provided as prompts (not implemented code)
The sections below are LLM prompts to generate code/pipelines for: audio call detector, TFLite/mobile export, active learning loop, Grad-CAM/audio saliency, and multi-species detection. Use them as templates; only the image classifier + Streamlit app are present in the repo.

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
- Place images under `data/crested_myna/` and `data/other/`, or upload via UI.
- Models in `models/` ending with `.pt`/`.pth` are auto-detected; otherwise a demo predictor is used.

## Train a model (PyTorch, optional)
```bash
python train_pytorch.py --data_dir ./data --arch resnet50 --epochs 12 --batch_size 32 --output models/crested_myna_model.pth --use_cuda
```
Key flags:
- `--arch` choices: `resnet18`, `resnet34`, `resnet50` (default), `resnet101`, `efficientnet_b0`
- `--lr`, `--weight_decay`, `--patience`, `--val_split`, `--num_workers`
- Saves checkpoint with class mapping and best weights by val accuracy.
Notes:
- The Streamlit app will run without a trained model, falling back to a demo predictor.
- Use this script only if you want to fine-tune or refresh a model; place outputs in `models/`.

## Quick overview of feature proposals

1. Image-based fine-tune prompt (transfer learning) — guide an LLM to produce training scripts/hparams for crested myna image classifier.
2. Audio-based call detector prompt — spectrogram pipeline + classifier for myna calls.
3. TFLite / Mobile export prompt — export/quantization workflow for on-device inference.
4. Active learning data-collector prompt — human-in-the-loop labeling and selection strategy.
5. Explainability prompt — Grad-CAM and audio saliency visualizations.
6. Multi-species detection / bounding-box prompt — move to object detection (YOLO/Detectron2).

---

## Prompts (English) — copy & paste ready

Below are complete prompts. Each starts with a one-line role assignment, followed by detailed instructions, required input and output formats, constraints, evaluation criteria, and an example.

### 1) Image-based fine-tune prompt (transfer learning)

Role: You are an expert ML engineer who writes reproducible PyTorch or TensorFlow training scripts and step-by-step instructions.

Task: Produce a complete, ready-to-run training script (and a short README usage snippet) to fine-tune a convolutional neural network for binary image classification: Crested Myna (Acridotheres cristatellus) vs. Not-Crested-Myna.

Input format (to the LLM):
- Dataset location: path or URL containing two folders: `crested_myna/` and `other/` with JPG/PNG images.
- Desired output model: `saved_model` (TensorFlow) or `model.pt` (PyTorch).
- Target hardware constraints (optional): e.g., "max model size 10 MB" or "train on a single GPU with 8GB RAM".
- Preferred framework: `tensorflow` or `pytorch`.

Output required from the LLM:
1. A single-file training script (Python) that performs data loading, preprocessing, transfer learning (e.g., MobileNetV2/EfficientNet or ResNet), training loop + validation, and saves the best model.
2. A short usage README section with commands to run training and evaluate the model.
3. Default hyperparameters (batch size, lr schedule, epochs) and recommended augmentation strategies for small bird-image datasets.

Constraints and checks:
- Use ImageNet-pretrained backbone.
- Include early stopping and model checkpointing by validation F1 score.
- Provide deterministic seed handling and reproducible transforms.
- Keep training script parameterized via CLI args (argparse).

Evaluation metrics to report in script:
- Accuracy, Precision, Recall, F1, and confusion matrix on validation set.

Example prompt payload (how to call the LLM):
{
  "dataset_path": "/path/to/dataset",
  "framework": "pytorch",
  "max_model_size_mb": 20,
  "target_hardware": "single GPU 8GB"
}

Example expected LLM response (summary):
- `train.py` (PyTorch) created: loads dataset from `/path/to/dataset`, uses EfficientNet-b0 backbone, trains for up to 40 epochs with ReduceLROnPlateau, saves `model.pt`, prints validation F1 each epoch.

---

### 2) Audio-based crested myna call detector prompt

Role: You are an audio ML engineer experienced with bird-call detection and sound-event detection pipelines.

Task: Provide a complete plan and code snippets for a pipeline that turns raw WAV recordings into a crested myna presence detector. Include data preprocessing, spectrogram generation, model architecture suggestion, training loop, and an inference example.

Input format (to the LLM):
- Dataset: folder structure `audio/crested_myna/*.wav`, `audio/other/*.wav` or labeled CSV with start/stop annotations.
- Sampling rate expected: 16 kHz or 44.1 kHz.
- Desired detection mode: "clip-level" (presence in a clip) or "temporal" (with timestamps).

Output required from the LLM:
1. Python snippets that: load WAV files, resample, compute mel-spectrograms (log-mels), and save as numpy arrays or TFRecords.
2. Model architecture choices: (a) CNN on spectrograms, or (b) CNN + CRNN for temporal localization.
3. Training and inference code, including thresholding logic and optional post-processing (median filtering, non-max suppression for events).
4. Example evaluation script computing clip-level precision/recall, and event-level metrics (precision/recall at IoU thresholds) if temporal annotations are available.

Constraints:
- Be robust to background noise: suggest augmentation (time-shift, noise injection, frequency masking).
- Include an option to generate spectrogram patches with 1s or 2s context windows.

Example output from LLM (short):
- `preprocess_audio.py` — resamples to 16 kHz, converts to 128-bin log-mel for 2s windows with 50% overlap.
- `train_audio.py` — trains a small CNN (3 conv blocks + global pooling) with binary cross-entropy and logs validation ROC-AUC.

---

### 3) Mobile export & optimization (TFLite) prompt

Role: You are a mobile ML engineer who produces compact models and a clear export workflow.

Task: Provide a full, step-by-step export and optimization workflow to convert a trained TensorFlow model into a TFLite model optimized for on-device inference. Include quantization options and a minimal on-device benchmarking snippet.

Input format (to the LLM):
- Path to a saved TensorFlow SavedModel directory or a Keras `.h5` model.
- Target constraints: e.g., max size 5 MB, latency <50ms for classification on mid-range mobile CPU.

Output required:
1. A `export_tflite.py` script that converts the model with: float32 baseline; post-training dynamic range quantization; and full integer quantization (with a representative dataset generator).
2. Instructions to measure model size and example Python snippet to run inference with `tflite.Interpreter` for a sample image or audio spectrogram.
3. Tips on reducing model size: pruning, smaller backbone (MobileNetV2/EfficientNet-lite), layer fusion.

Constraints:
- If integer-quantization is requested, provide a small representative dataset generator function and clear steps to test correctness vs. float model.

Example expected LLM output:
- `export_tflite.py` with CLI interface: `--saved_model`, `--output_tflite`, `--quantize {none,dynamic,full}` and a `representative_data()` stub.

---

### 4) Active learning / data-collection prompt (human-in-the-loop)

Role: You are a data-engineer and ML scientist building efficient active-learning loops.

Task: Create a prompt that, given a model and unlabeled pool, returns an ordered list of samples for human labeling and provides labeling UI suggestions and annotation format.

Input format (to the LLM/system):
- Model predictions on an unlabeled pool with confidence scores (JSON list of {id, path, score, pred}).
- Budget: number of samples to label (e.g., 100 per iteration).

Output required:
1. Selection strategy code (uncertainty sampling, entropy, margin sampling, or hybrid with diversity — e.g., KMeans on features + uncertainty).
2. A suggested annotation schema (CSV/JSON) and minimal web UI design (Flask or Streamlit) for quick labeling with keyboard shortcuts.
3. Instructions to retrain the model with newly-labeled data and to repeat the loop.

Constraints:
- Prefer batch-mode selection that balances uncertainty and diversity to avoid label redundancy.

Example expected result:
- A `select_for_labeling.py` script that reads `predictions.json`, computes entropy, clusters feature embeddings to select diverse uncertain samples, and writes `to_label.csv`.

---

### 5) Explainability prompt (Grad-CAM and audio saliency)

Role: You are an explainability engineer who generates interpretable visualizations for both images and audio models.

Task: Provide scripts and clear instructions to produce Grad-CAM visualizations for image models and time-frequency saliency maps for audio spectrogram models. Include CLI examples and recommended thresholds to show salient regions.

Input format (to the LLM):
- Path to trained model and a sample image or audio clip.

Output required:
1. `gradcam_image.py` — given an image and model, output `heatmap.png` overlayed on original.
2. `audio_saliency.py` — given a WAV and spectrogram model, output a PNG of the saliency map and an annotated timestamped summary of high-saliency intervals.
3. Short diagnostic checklist: when visualizations look noisy, try smoothing, larger receptive field, or aggregating multiple samples.

Example expected output from LLM:
- `gradcam_image.py` using PyTorch hooks and OpenCV to produce overlays.

---

### 6) Multi-species detection / bounding-box prompt

Role: You are a computer-vision engineer experienced in building object detectors for wildlife monitoring.

Task: Provide a step-by-step prompt and dataset-conversion scripts to move from binary classifier to an object detector that can locate and classify crested mynas in photos (bounding boxes). Include annotation format examples (COCO VOC), trainer selection (YOLOv5/Detectron2), and a minimal training pipeline.

Input format (to the LLM/system):
- Existing image folder and optional CSV of bounding boxes, or only classification labels.

Output required:
1. `convert_to_coco.py` — script to create COCO JSON from existing annotations or generate weak boxes from classifier heatmaps for human correction.
2. A training YAML config for YOLOv5 or Detectron2 and CLI commands to start training.
3. A short suggestion for semi-automatic annotation (using classifier + Grad-CAM to propose boxes for human correction) to speed up annotation.

Evaluation:
- mAP@0.5 and mAP@[0.5:0.95] on a held-out test set.

---

## How to use these prompts

1. Pick the feature you want to add (image classifier, audio detector, mobile export, etc.).
2. Copy the corresponding prompt above and paste it into your preferred LLM interface (OpenAI, local LLM) or into an internal engineering ticket.
3. Provide the dataset path and the optional hardware constraints in the prompt payload.
4. Use the LLM response as a scaffold: verify code, run unit tests, and adapt hyperparameters to your environment.

## Suggested next steps (implementation)

- Start with the Image-based fine-tune prompt to obtain a training script and train a baseline model.
- Use the Explainability prompt to sanity-check the model and create initial annotations for object detection.
- If you have field recordings, run the Audio-based prompt to create a call detector; otherwise, focus on images first.
- Export a compact model with the Mobile export prompt for on-device testing.
- Iterate with the Active learning prompt to label new, high-value samples and improve model robustness.

## Closing notes

These prompts are intentionally explicit and prescriptive so they can be used both interactively with an LLM and as developer tickets. If you want, I can:

- Generate the actual `train.py`, `preprocess_audio.py`, and `export_tflite.py` files for one selected feature now.
- Create a minimal Streamlit app for labeling and reviewing model outputs.

Update the TODO list status after you choose which feature(s) you'd like implemented first.

---

Licensed for personal and educational use. Cite the original Colab demo when re-using parts of it.
