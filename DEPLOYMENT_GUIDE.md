# 🚀 Deployment Guide: Crested Myna Recognizer on Streamlit Cloud

## 📌 Problem: Data Folder Not Accessible on Streamlit Cloud

When you deploy this app to **Streamlit Cloud**, the `data/` folder may not be available because:

1. **Git doesn't track empty folders** — If `data/` only has `.gitignore`, it won't be pushed
2. **Large files aren't uploaded** — GitHub LFS or too many images can cause issues
3. **Path issues** — Streamlit Cloud runs from a different working directory

---

## ✅ Solutions

### **Solution 1: Use Image Upload (⭐ Recommended for Cloud)**

This is the **easiest and most reliable** way:

1. Deploy your app to Streamlit Cloud without worrying about data
2. Users can upload images directly via the Streamlit UI
3. No data folder needed

**Advantages:**
- ✅ Works immediately on Streamlit Cloud
- ✅ No data management overhead
- ✅ Users control what images are analyzed
- ✅ Scalable and secure

**How it works:**
```
1. User opens your Streamlit Cloud app
2. Selects "Upload image" in sidebar
3. Drags and drops an image
4. App displays predictions and analytics
```

---

### **Solution 2: Commit Data to GitHub (For Small Datasets)**

If you want to use the "Project data folder" feature:

#### Step 1: Organize your data
```bash
mkdir -p data/crested_myna
mkdir -p data/other

# Copy your images (example with find)
find /path/to/crested_images -name "*.jpg" -o -name "*.png" | xargs -I {} cp {} data/crested_myna/

find /path/to/other_birds -name "*.jpg" -o -name "*.png" | xargs -I {} cp {} data/other/
```

#### Step 2: Update `.gitignore` to allow data folder
```bash
# Remove 'data/' from .gitignore if it's there
# Keep only:
cat .gitignore
# __pycache__/
# *.pyc
# .DS_Store
# *.pth
# *.pt
# models/
```

Or explicitly allow data folder:
```bash
# In .gitignore, change from:
data/

# To:
data/.DS_Store
# Now data/*.jpg files will be tracked
```

#### Step 3: Commit and push
```bash
git add data/
git commit -m "Add sample dataset for Crested Myna recognition"
git push origin main
```

#### Step 4: Redeploy on Streamlit Cloud
- Go to https://share.streamlit.io/
- Rerun your app
- Data folder should now be available

**⚠️ Important:**
- Only commit **representative sample** images (e.g., 20-50 per class)
- GitHub has file size limits (~100MB per file, ~2GB per repo)
- Large datasets should use cloud storage instead

---

### **Solution 3: Use Cloud Storage (For Large Datasets)**

For production deployments with many images:

#### Option A: Google Cloud Storage
```python
from google.cloud import storage
import streamlit as st

@st.cache_resource
def get_gcs_client():
    return storage.Client()

def load_images_from_gcs(bucket_name, prefix):
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    
    images = []
    for blob in blobs:
        if blob.name.endswith(('.jpg', '.jpeg', '.png')):
            img_bytes = blob.download_as_bytes()
            from PIL import Image
            img = Image.open(io.BytesIO(img_bytes))
            images.append((blob.name, img))
    return images

# In streamlit_app.py:
images = load_images_from_gcs("your-bucket", "crested_myna/")
```

#### Option B: AWS S3
```python
import boto3
from PIL import Image
import io

@st.cache_resource
def get_s3_client():
    return boto3.client('s3')

def load_images_from_s3(bucket, prefix):
    s3 = get_s3_client()
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    
    images = []
    for obj in response.get('Contents', []):
        key = obj['Key']
        if key.endswith(('.jpg', '.jpeg', '.png')):
            img_obj = s3.get_object(Bucket=bucket, Key=key)
            img = Image.open(io.BytesIO(img_obj['Body'].read()))
            images.append((key, img))
    return images
```

#### Option C: Direct HTTP URL
```python
import requests
from PIL import Image
import io

@st.cache_data
def load_images_from_urls(urls):
    images = []
    for url in urls:
        try:
            response = requests.get(url)
            img = Image.open(io.BytesIO(response.content))
            images.append((url, img))
        except Exception as e:
            st.warning(f"Failed to load {url}: {e}")
    return images

# In streamlit_app.py:
sample_urls = [
    "https://example.com/crested_myna_1.jpg",
    "https://example.com/crested_myna_2.jpg",
]
images = load_images_from_urls(sample_urls)
```

---

## 🔧 Setup Instructions by Environment

### **Local Development** (Your Machine)
```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/repo.git
cd repo

# 2. Create data folder
mkdir -p data/crested_myna
mkdir -p data/other

# 3. Add images
# Place your images in those folders

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run app
streamlit run streamlit_app.py

# 6. Open browser to http://localhost:8501
```

### **Streamlit Cloud Deployment** (Free Hosting)

#### Option 1: Without Data Folder (Recommended)
```bash
# 1. Push to GitHub (without data folder)
git push origin main

# 2. Go to https://share.streamlit.io/
# 3. Click "New app"
# 4. Select:
#    - Repository: YOUR_USERNAME/repo
#    - Branch: main
#    - Main file path: streamlit_app.py

# 5. Users upload images directly in the app
```

#### Option 2: With Small Sample Dataset
```bash
# 1. Add small sample data to repo
mkdir -p data/crested_myna
mkdir -p data/other
# Copy 10-20 sample images to each folder

# 2. Update .gitignore (remove 'data/' if present)

# 3. Commit and push
git add data/
git commit -m "Add sample images"
git push origin main

# 4. Deploy on Streamlit Cloud as above
```

---

## 📊 Recommended Configuration

### For Best User Experience:

```python
# streamlit_app.py configuration
import streamlit as st

st.set_page_config(
    page_title="Crested Myna Recognizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Suggest upload for cloud deployments
import os
is_cloud = os.getenv("STREAMLIT_SERVER_HEADLESS")

if is_cloud:
    st.sidebar.info("""
    🌐 **Running on Streamlit Cloud**
    
    For best results, use **Upload image** mode.
    Images process instantly and are secure.
    """)
```

---

## ❓ Troubleshooting

### "Data folder not found"
**Cause:** Folder doesn't exist in cloud  
**Solution:** Use "Upload image" mode or commit data to GitHub

### "No images found in data folder"
**Cause:** Images not committed to repo  
**Solution:** 
```bash
git add data/
git commit -m "Add images"
git push
```

### "App takes too long to load"
**Cause:** Too many/large images in data folder  
**Solution:** 
- Reduce number of sample images (< 50)
- Use cloud storage instead of GitHub
- Use image upload mode

### "Out of memory" or "App crashes"
**Cause:** Loading too many images at once  
**Solution:**
```python
# Use pagination in app
import streamlit as st

per_page = 10
page = st.number_input("Page", min_value=1)
start_idx = (page - 1) * per_page
end_idx = start_idx + per_page

displayed_images = all_images[start_idx:end_idx]
```

---

## 🎯 Recommended Deployment Path

```
Local Development
        ↓
Test with upload feature ✓
        ↓
Push to GitHub without data/
        ↓
Deploy to Streamlit Cloud
        ↓
Users upload images (✓ Works!)
        ↓
(Optional) Add cloud storage for production use
```

---

## 📚 Resources

- [Streamlit Cloud Docs](https://docs.streamlit.io/streamlit-cloud)
- [Streamlit App Gallery](https://streamlit.io/gallery)
- [Google Cloud Storage Python Client](https://cloud.google.com/python/docs/reference/storage/latest)
- [AWS S3 Boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html)

---

## 💡 Quick Reference

| Environment | Data Folder | Best Method | Setup Time |
|---|---|---|---|
| Local | ✅ Works | Use folder or upload | 5 min |
| Streamlit Cloud | ❌ Missing | Use upload | 2 min |
| Streamlit Cloud + Sample Data | ✅ Works | Commit to GitHub | 10 min |
| Production | ✅ Scalable | Cloud Storage (GCS/S3) | 30 min |

---

## 🚀 Next Steps

1. **Immediate**: Deploy using upload mode (works now!)
2. **Short-term**: Add sample images if you want folder browsing
3. **Long-term**: Integrate cloud storage for production use

Good luck! 🎉
