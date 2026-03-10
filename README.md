# 🎙️ VidoSTT Colab 版 v2.1 — 影片轉繁體中文字幕（含講者分離）

**詳細介紹：** [VidoSTT 官方網站](https://sites.google.com/view/vidostt/vidostt?authuser=0)  
**線上試用 Colab：** [Google Colab 連結](https://colab.research.google.com/drive/15e7ZIJ5P74IarhK5xH-25k7N5nTLS_qv#scrollTo=x46W3Qd1gcad)  
**使用前請先開啟 T4 GPU：**（如果在 Colab）執行階段 → 變更執行階段類型 → 硬體加速器：**T4 GPU** → 儲存

> ⚠️ 免費帳號每次可用 **2～4 小時**。建議單次處理 **30 分鐘以內** 的影片。

---

### ✅ 使用步驟
1. 安裝環境：`pip install -r requirements.txt`
2. （可選）填入 HuggingFace Token 啟用說話者分離
3. 執行 `python main.py`：填入設定後轉錄，完成後自動產生檔案

### 📊 兩種模式比較
| 模式 | 說話者標記 | 速度 | 需要 Token |
|------|-----------|------|----------|
| **基本模式** | A/B 輪替（簡單） | 快 | ❌ 不需要 |
| **說話者分離模式** | 真實聲紋分析 | 慢約 2x | ✅ 需要 |

### 📄 產出檔案
- `.srt` — 帶講者標記的字幕（可匯入 DaVinci / CapCut）
- `.txt` — 純文字逐字稿
