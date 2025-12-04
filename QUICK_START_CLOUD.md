# 🚀 Quick Start: Deploy to Streamlit Cloud in 2 Minutes

## Problem ❌
- ✗ My `data/` folder doesn't work on Streamlit Cloud
- ✗ App crashes when I try to access project data folder
- ✗ How do I deploy this app?

## Solution ✅

### Step 1: Push to GitHub (Already Done? Skip!)

```bash
# If you haven't already
git add .
git commit -m "Ready for cloud deployment"
git push origin main
```

### Step 2: Go to Streamlit Cloud

1. Open https://share.streamlit.io/ (log in with GitHub)
2. Click **"New app"**

### Step 3: Configure Your App

| Field | Value |
|-------|-------|
| **Repository** | `YOUR_USERNAME/Deep-Learning-AIGC-hw4` |
| **Branch** | `main` |
| **Main file path** | `streamlit_app.py` |

Click **"Deploy"** and wait ~30 seconds ⏳

### Step 4: Use Your App! 🎉

```
Your app is now live at:
https://share.streamlit.io/YOUR_USERNAME/YOUR_REPO_NAME
```

**That's it!** Your app is deployed to the cloud.

---

## How to Use on Cloud

### ✅ Option 1: Upload Images (Recommended ⭐)
1. In sidebar, select **"Upload image"**
2. Drag & drop your photo
3. See instant predictions and analytics

**Why this works best:**
- ✓ Works immediately on cloud
- ✓ Faster than file browsing
- ✓ Most secure approach
- ✓ No setup needed

### ⚠️ Option 2: Project Data Folder (Limited)
1. In sidebar, select **"Project data folder"**
2. Enter path: `./data`
3. This only works if:
   - ✓ You committed `data/` to GitHub
   - ✓ Folder has actual image files
   - ✓ Files are < 2GB total

---

## FAQ

### Q: Why doesn't my data folder show up?
**A:** GitHub doesn't automatically upload empty folders. Either:
- Use **image upload mode** (recommended)
- Or commit actual images to `data/` folder

### Q: Can I use my local data folder on cloud?
**A:** No. Cloud apps can only access files in the GitHub repo. To use custom data:
- Upload images through the UI, OR
- Push images to GitHub, OR  
- Use cloud storage (Google Cloud Storage / AWS S3)

See [`DEPLOYMENT_GUIDE.md`](./DEPLOYMENT_GUIDE.md) for advanced options.

### Q: How do I update my app after changes?
**A:** 
```bash
git add .
git commit -m "My changes"
git push
```
Streamlit Cloud auto-detects changes and redeploys!

### Q: My app crashed on cloud!
**A:** Check the logs:
1. Go to https://share.streamlit.io/
2. Find your app
3. Click menu (⋮) → View logs
4. See detailed error messages

Most common fixes:
- Restart the app (menu → Rerun)
- Update dependencies in `requirements.txt`
- Remove large files from repo

---

## Cloud vs Local

| Feature | Local | Cloud |
|---------|-------|-------|
| **Data folder access** | ✅ Works | ⚠️ Limited |
| **Upload images** | ✅ Works | ✅ Works |
| **PyTorch inference** | ✅ Works | ✅ Works |
| **Analytics charts** | ✅ Works | ✅ Works |
| **No setup needed** | ❌ Install Python | ✅ Auto |
| **Cost** | 💰 Your machine | 🆓 Free tier available |

---

## Next Steps

### 🎯 Immediate
1. Deploy to cloud (above)
2. Test with image upload ✓
3. Share your app link!

### 📊 Later (Optional)
- Add sample images to `data/` folder (commit to GitHub)
- Set up cloud storage for production
- Train a custom model
- See [`DEPLOYMENT_GUIDE.md`](./DEPLOYMENT_GUIDE.md)

---

## Share Your App! 🌍

Your live app URL: 
```
https://share.streamlit.io/YOUR_USERNAME/YOUR_REPO_NAME
```

Send this link to friends, colleagues, students, etc. They can:
- ✅ Upload photos instantly
- ✅ See predictions
- ✅ View analytics
- ✅ No installation needed!

---

## 📚 Full Documentation

- **Troubleshooting?** See [`DEPLOYMENT_GUIDE.md`](./DEPLOYMENT_GUIDE.md)
- **Want cloud storage?** See cloud storage sections in [`DEPLOYMENT_GUIDE.md`](./DEPLOYMENT_GUIDE.md)
- **Local setup?** See `README.md`

---

**Need help?** Check the app's sidebar for hints, or refer to the full deployment guide. Good luck! 🚀
