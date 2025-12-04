from pathlib import Path
from PIL import Image
import random
import io
from typing import Tuple, Dict, List


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
