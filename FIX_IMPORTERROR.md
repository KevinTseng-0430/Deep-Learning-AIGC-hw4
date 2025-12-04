# 🔧 ImportError Fix Guide

## 問題
> "This app has encountered an error. The original error message is redacted..."

這通常意味著有 import 錯誤或依賴項缺失。

---

## ✅ 已修復的問題

### 1. 缺失的 `requests` 依賴
**問題:** `requests` 沒有列在 `requirements.txt` 中  
**修復:** ✅ 已添加到 `requirements.txt`

### 2. PyTorch 在 Streamlit Cloud 上的安裝問題
**問題:** `torch` 和 `torchvision` 在 Streamlit Cloud 上可能安裝失敗  
**修復:** ✅ 已移除（應用使用啟發式預測器工作）

---

## 🚀 立即行動

### 步驟 1: 驗證 requirements.txt
確保您的 `requirements.txt` 包含所有這些行：

```
streamlit>=1.20
Pillow>=9.0
numpy
pandas
plotly
seaborn
requests
```

### 步驟 2: 推送到 GitHub
```bash
git add requirements.txt
git commit -m "Fix: Update requirements with all dependencies"
git push origin main
```

### 步驟 3: 重新部署
- 在 Streamlit Cloud 上找到您的應用
- 點擊菜單 (⋮) → 選擇 "Rerun"
- 或等待自動重新部署

---

## 📋 完整的工作 requirements.txt

```
# Web Framework
streamlit>=1.20

# Image Processing
Pillow>=9.0

# Data Processing
numpy
pandas

# Visualization
plotly
seaborn

# HTTP Requests
requests
```

---

## 🧪 本地測試

在推送之前，在本地測試：

```bash
# 1. 創建虛擬環境
python -m venv .venv
source .venv/bin/activate

# 2. 安裝依賴
pip install -r requirements.txt

# 3. 運行診斷
python diagnose.py

# 4. 如果一切通過，啟動應用
streamlit run streamlit_app.py
```

---

## ❓ 常見 ImportError 原因

### 1. 缺失的依賴項
```
ImportError: No module named 'requests'
```
**解決:** 在 `requirements.txt` 中添加該模塊

### 2. 版本不兼容
```
ImportError: cannot import name 'XXX' from 'module'
```
**解決:** 檢查版本需求，更新 `requirements.txt`

### 3. 拼寫錯誤
```
ImportError: No module named 'plotyl'
```
**解決:** 檢查 `requirements.txt` 中的拼寫 (應該是 `plotly`)

### 4. 模塊名稱與包名稱不同
```
ImportError: No module named 'cv2'
```
**解決:** 需要安裝 `opencv-python`，而不是 `cv2`

---

## 🔍 診斷步驟

1. **查看應用日誌**
   - 在 Streamlit Cloud 上，點擊 "Manage app" → "View logs"
   - 查找具體的 ImportError 消息

2. **運行本地診斷**
   ```bash
   python diagnose.py
   ```

3. **測試每個導入**
   ```bash
   python -c "import requests; print('OK')"
   python -c "import plotly.express; print('OK')"
   # 等等...
   ```

4. **檢查版本兼容性**
   ```bash
   pip show streamlit
   pip show pandas
   # 確保版本匹配 requirements.txt
   ```

---

## 📞 如果問題仍然存在

1. **檢查您的 Python 版本**
   ```bash
   python --version
   # Streamlit Cloud 使用 Python 3.10+
   ```

2. **清除 Streamlit Cloud 緩存**
   - 在應用菜單 (⋮) 中，選擇 "Rerun"
   - 或點擊應用頁面上的"Always rerun"

3. **檢查是否有文件丟失**
   ```bash
   ls -la
   # 確保所有 .py 和 .md 文件都存在
   ```

4. **查看完整的 Streamlit Cloud 日誌**
   - App Settings → View logs
   - 搜索完整的錯誤堆棧跟蹤

---

## ✅ 驗證修復

部署後，檢查以下內容：

- [ ] 應用啟動無錯誤
- [ ] "Upload image" 功能可用
- [ ] 側邊欄顯示環境信息
- [ ] Analytics 標籤加載
- [ ] 幫助部分顯示

---

## 🎯 下一步

如果一切都工作正常：

1. ✅ 測試上傳功能
2. ✅ 查看分析圖表
3. ✅ 分享應用 URL
4. ✅ 參考 [`START_HERE.md`](./START_HERE.md) 了解更多

---

**需要幫助?** 查看 [`CLOUD_SOLUTION_SUMMARY.md`](./CLOUD_SOLUTION_SUMMARY.md) 了解完整的故障排除指南。
