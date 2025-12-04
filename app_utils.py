from pathlib import Path
from PIL import Image
import random
import io
import os
from typing import Tuple, Dict, List


def is_streamlit_cloud() -> bool:
    """Detect if running on Streamlit Cloud."""
    return os.getenv("STREAMLIT_SERVER_HEADLESS") == "true"


def is_local_deployment() -> bool:
    """Detect if running locally."""
    return not is_streamlit_cloud()


def gather_dataset_stats(data_dir: Path) -> Dict:
    """Gather simple dataset statistics for images under data_dir.

    Returns dict with:
      - total_images
      - class_counts (dict: folder name -> count)
      - sizes: list of (width, height)
      - formats: dict of format counts
    """
    stats = {
        "total_images": 0,
        "class_counts": {},
        "sizes": [],
        "formats": {},
    }
    if not data_dir.exists():
        return stats

    # If images are inside subfolders, treat each subfolder as a class
    for child in data_dir.iterdir():
        if child.is_dir():
            imgs = list_images_in_folder(child)
            stats["class_counts"][child.name] = len(imgs)
            stats["total_images"] += len(imgs)
            for p in imgs:
                try:
                    with Image.open(p) as im:
                        stats["sizes"].append((im.width, im.height))
                        fmt = (im.format or "UNKNOWN").upper()
                        stats["formats"][fmt] = stats["formats"].get(fmt, 0) + 1
                except Exception:
                    continue
    # If no subfolders with images, look directly in data_dir
    if stats["total_images"] == 0:
        imgs = list_images_in_folder(data_dir)
        stats["total_images"] = len(imgs)
        stats["class_counts"] = {"all": len(imgs)}
        for p in imgs:
            try:
                with Image.open(p) as im:
                    stats["sizes"].append((im.width, im.height))
                    fmt = (im.format or "UNKNOWN").upper()
                    stats["formats"][fmt] = stats["formats"].get(fmt, 0) + 1
            except Exception:
                continue

    return stats


def extract_image_metadata(pil_img: Image.Image, path: Path = None) -> Dict:
    """Return a small metadata dict for a PIL image and optional path."""
    md = {
        "width": pil_img.width,
        "height": pil_img.height,
        "mode": pil_img.mode,
        "format": pil_img.format or "UNKNOWN",
        "aspect_ratio": round(pil_img.width / pil_img.height, 3) if pil_img.height > 0 else None,
    }
    if path is not None and path.exists():
        try:
            md["file_size_bytes"] = path.stat().st_size
            md["file_size_mb"] = round(path.stat().st_size / (1024 * 1024), 2)
        except Exception:
            pass
    return md


def compute_detailed_stats(data_dir: Path) -> Dict:
    """Compute detailed statistics including aspect ratios, file sizes, and dimension distributions."""
    stats = gather_dataset_stats(data_dir)
    
    sizes = stats.get("sizes", [])
    aspect_ratios = []
    file_sizes = []
    
    if not data_dir.exists():
        return stats
    
    # Recompute with file size and aspect ratio info
    for child in data_dir.iterdir():
        if child.is_dir():
            img_paths = list_images_in_folder(child)
            for p in img_paths:
                try:
                    with Image.open(p) as im:
                        if im.height > 0:
                            aspect_ratios.append(im.width / im.height)
                        file_size_bytes = p.stat().st_size
                        file_sizes.append(file_size_bytes / (1024 * 1024))  # MB
                except Exception:
                    continue
    
    # Fallback if no subfolders
    if not aspect_ratios and not file_sizes:
        img_paths = list_images_in_folder(data_dir)
        for p in img_paths:
            try:
                with Image.open(p) as im:
                    if im.height > 0:
                        aspect_ratios.append(im.width / im.height)
                    file_size_bytes = p.stat().st_size
                    file_sizes.append(file_size_bytes / (1024 * 1024))  # MB
            except Exception:
                continue
    
    stats["aspect_ratios"] = aspect_ratios
    stats["file_sizes_mb"] = file_sizes
    
    return stats




def list_images_in_folder(folder: Path):
    exts = ["*.jpg", "*.jpeg", "*.png"]
    files = []
    for e in exts:
        files.extend(list(folder.rglob(e)))
    # sort for consistent ordering
    return sorted(files)


def load_pil_image(source):
    # source can be a Path or an uploaded file-like object
    if isinstance(source, Path):
        return Image.open(source).convert("RGB")
    else:
        # uploaded file (BytesIO)
        return Image.open(io.BytesIO(source.getvalue() if hasattr(source, "getvalue") else source.read())).convert("RGB")


def predict_image_stub(pil_img, torch_model=None):
    """Return ("Crested Myna", confidence) tuple where confidence is 0-1 (higher = more likely Crested Myna).
    If a torch_model is provided try to use it; otherwise fallback to heuristic/random demo."""
    # Try to use real model if provided (best-effort)
    if torch_model is not None:
        try:
            # If model is a dict-like checkpoint, not a Module, skip (loader should produce Module)
            import torch
            from torchvision import transforms
            import math

            # If torch_model is an nn.Module, perform inference
            if hasattr(torch_model, "eval"):
                model = torch_model
                model.eval()
                preprocess = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                inp = preprocess(pil_img).unsqueeze(0)  # batch dim
                with torch.no_grad():
                    out = model(inp)
                    if out is None:
                        raise RuntimeError("Model returned None")
                    # ensure tensor
                    if isinstance(out, (list, tuple)):
                        out = out[0]
                    probs = torch.softmax(out, dim=1).squeeze(0).cpu().numpy()

                # Map class idx to confidence for Crested Myna
                class_to_idx = getattr(model, "class_to_idx", None)
                if class_to_idx:
                    # Try to find Crested Myna index
                    idx_to_class = {v: k for k, v in class_to_idx.items()}
                    
                    # Look for "Crested Myna" or "crested_myna" in class names
                    crested_myna_idx = None
                    for idx, name in idx_to_class.items():
                        if "crested" in name.lower() and "myna" in name.lower():
                            crested_myna_idx = idx
                            break
                    
                    if crested_myna_idx is not None:
                        # Confidence that this is Crested Myna
                        confidence = float(probs[crested_myna_idx])
                    else:
                        # Fallback: assume index 0 or 1 is Crested Myna, take max prob
                        confidence = float(probs.max())
                    
                    return "Crested Myna", confidence
                else:
                    # fallback: take max probability across classes
                    confidence = float(probs.max())
                    return "Crested Myna", confidence
        except Exception:
            # if any of the model inference steps fail, fall back to heuristic
            pass

    # Heuristic: use random but deterministic by image size
    w, h = pil_img.size
    seed = (w * 31 + h * 17) % 1000
    rnd = random.Random(seed)
    prob = rnd.random() * 0.6 + 0.2
    return "Crested Myna", prob


def infer_label_from_path(path: Path):
    """Try to infer a ground-truth label from a file Path by using its parent folder name.

    Returns normalized label string or None if not inferable.
    """
    try:
        if path is None:
            return None
        # If path is a Path-like and has a parent folder name, use that
        p = Path(path)
        parent = p.parent.name.lower()
        if parent:
            return parent.replace(" ", "_")
    except Exception:
        pass
    return None


def prepare_predictions_dataframe(images, torch_model=None):
    """Given images (list of (path_or_name, PIL.Image) tuples), run prediction and return a DataFrame.

    DataFrame columns: image, label, confidence (0-1), confidence_pct, score, true_label (optional), is_correct (optional)
    """
    import pandas as _pd

    rows = []
    for path, img in images:
        try:
            label, conf = predict_image_stub(img, torch_model)
        except Exception:
            label, conf = "Crested Myna", 0.0

        true_label = None
        try:
            # if path is a Path instance or string pointing into a folder, infer label
            if isinstance(path, (str, Path)):
                p = Path(path)
                if p.exists():
                    true_label = infer_label_from_path(p)
                else:
                    # for uploaded files we might not have a real path; skip
                    true_label = None
        except Exception:
            true_label = None

        is_correct = None
        if true_label is not None:
            # consider true if parent's label contains myna/crested keywords
            tl = str(true_label).lower()
            is_myna = ("crested" in tl and "myna" in tl) or ("myna" in tl) or ("crested_myna" in tl)
            # Our predictor only emits "Crested Myna" label
            pred_myna = (str(label).lower().find("crested") >= 0 or str(label).lower().find("myna") >= 0)
            is_correct = (is_myna and pred_myna) or ((not is_myna) and (not pred_myna))

        rows.append({
            "image": str(path),
            "label": label,
            "confidence": float(conf),
            "confidence_pct": f"{float(conf)*100:.1f}%",
            "true_label": true_label,
            "is_correct": is_correct,
        })

    df = _pd.DataFrame(rows)
    return df


def calibration_stats(df, n_bins: int = 10):
    """Compute calibration data: for bins of predicted confidence, return average confidence and empirical accuracy.

    Expects df with columns 'confidence' and 'is_correct' (boolean or None). If 'is_correct' is None/absent, accuracy per bin will be None.
    Returns a dict with 'bin_mid', 'avg_conf', 'accuracy', 'count'.
    """
    import numpy as _np

    if df is None or df.shape[0] == 0:
        return {"bin_mid": [], "avg_conf": [], "accuracy": [], "count": []}

    confs = df["confidence"].values
    bins = _np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = _np.digitize(confs, bins, right=True) - 1

    bin_mid = []
    avg_conf = []
    accuracy = []
    counts = []

    for i in range(n_bins):
        mask = bin_ids == i
        count = int(mask.sum())
        counts.append(count)
        if count == 0:
            bin_mid.append(float((bins[i] + bins[i+1]) / 2.0))
            avg_conf.append(None)
            accuracy.append(None)
            continue

        bin_mid.append(float((bins[i] + bins[i+1]) / 2.0))
        avg_conf.append(float(confs[mask].mean()))

        if "is_correct" in df.columns and df["is_correct"].notnull().any():
            acc = float(df.loc[mask, "is_correct"].mean())
            accuracy.append(acc)
        else:
            accuracy.append(None)

    return {"bin_mid": bin_mid, "avg_conf": avg_conf, "accuracy": accuracy, "count": counts}


def try_load_torch_model(models_dir: Path):
    """Attempt to find a PyTorch .pt model in models_dir and return (found:bool, name, model_object_or_none)

    This function performs a low-risk best-effort search only; it does not import torch unless a .pt file is found.
    """
    if not models_dir.exists():
        return False, None, None
    candidates = list(models_dir.glob("*.pt")) + list(models_dir.glob("*.pth"))
    if not candidates:
        return False, None, None
    candidate = sorted(candidates)[0]
    # Try to import torch lazily
    try:
        import torch
        from torchvision import models

        loaded = None
        try:
            loaded = torch.load(candidate, map_location=torch.device("cpu"))
        except Exception:
            loaded = None

        # If loaded is a checkpoint dict with architecture info
        if isinstance(loaded, dict) and "arch" in loaded and "state_dict" in loaded:
            arch = loaded.get("arch", "resnet50")
            class_to_idx = loaded.get("class_to_idx", {})
            num_classes = len(class_to_idx) if class_to_idx else 2
            # build compatible model
            try:
                if arch == "resnet18":
                    model = models.resnet18(pretrained=False)
                    in_f = model.fc.in_features
                    model.fc = torch.nn.Linear(in_f, num_classes)
                elif arch == "resnet34":
                    model = models.resnet34(pretrained=False)
                    in_f = model.fc.in_features
                    model.fc = torch.nn.Linear(in_f, num_classes)
                elif arch == "resnet50":
                    model = models.resnet50(pretrained=False)
                    in_f = model.fc.in_features
                    model.fc = torch.nn.Linear(in_f, num_classes)
                elif arch == "resnet101":
                    model = models.resnet101(pretrained=False)
                    in_f = model.fc.in_features
                    model.fc = torch.nn.Linear(in_f, num_classes)
                elif arch in ("efficientnet_b0", "efficientnetb0"):
                    try:
                        model = models.efficientnet_b0(pretrained=False)
                        in_f = model.classifier[1].in_features
                        model.classifier[1] = torch.nn.Linear(in_f, num_classes)
                    except Exception:
                        # fallback to resnet50 if efficientnet not available
                        model = models.resnet50(pretrained=False)
                        in_f = model.fc.in_features
                        model.fc = torch.nn.Linear(in_f, num_classes)
                else:
                    # unsupported arch, return raw loaded object
                    model = loaded

                if isinstance(model, torch.nn.Module):
                    model.load_state_dict(loaded["state_dict"])
                    model.eval()
                    # attach mapping for later
                    setattr(model, "class_to_idx", class_to_idx)
                    return True, candidate.name, model
                else:
                    return True, candidate.name, loaded
            except Exception:
                return True, candidate.name, loaded

        # If loaded is already a torch nn.Module (scripted or saved model)
        if isinstance(loaded, torch.nn.Module):
            try:
                loaded.eval()
            except Exception:
                pass
            return True, candidate.name, loaded

        # else, return the raw loaded object (may be None)
        return True, candidate.name, loaded
    except Exception:
        # torch not available; report presence only
        return True, candidate.name, None
