# 快速部署命令

## 本地測試（推薦先做這個）

```bash
# 啟動應用
streamlit run streamlit_app.py

# 然後在瀏覽器中：
# 1. 選擇 "Upload image"
# 2. 上傳 2-3 張圖片
# 3. 驗證預測立即顯示
# 4. 查看批量統計
```

## 推送到 GitHub

```bash
# 添加改動
git add streamlit_app.py README.md FEATURE_MULTI_UPLOAD.md

# 提交
git commit -m "Feature: Support multi-image upload with direct predictions"

# 推送
git push origin main
```

## Streamlit Cloud 部署

1. 訪問 https://share.streamlit.io/
2. 找到您的應用
3. 等待自動部署（30 秒）
4. 刷新頁面
5. 上傳多張圖片測試

---

**就這麼簡單！** ✨
