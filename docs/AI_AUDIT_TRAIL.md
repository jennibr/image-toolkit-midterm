# AI Audit Trail（對應報告第 4 章）

**使用的 AI：** Claude（Anthropic），透過 claude.ai 對話介面
**模型版本：** 撰寫本紀錄時為 Claude Sonnet 5.5（對話期間平台可能切換模型）
**開發期間：** 2026 年 11 月
**完整對話連結：** （待補：請貼上 Claude 對話的分享連結）

記錄格式：Prompt（原文）→ AI 產出摘要 → 除錯／調整（Refinement）。
只貼截圖、沒有文字的訊息標示為「（上傳截圖）」。第 1 階段的完整原文見報告附錄 A。

---

## Module 0 – 專案架構與環境

### 階段 1：專案架構與初版程式
- **Prompt：** （貼上完整考試說明與四項核心功能需求）…
  "for your information I want the file become seperate, like there is app.py, chapter 1.py/something, chapter2.py/something, chapter3.py/something so on. Let's do the code first in vscode, in the end the PDF. I want to use python 3.11"
- **AI 產出：** Streamlit 專案；`modules/` 依功能拆成四個檔案；`app.py`、`requirements.txt`、`README.md`、`tests/`、稽核紀錄範本。
- **Refinement：** AI 的沙盒 shell 不支援大括號展開（`mkdir -p {a,b}`），檔案沒有寫入，AI 從 stderr 發現後改為明確建立資料夾。煙霧測試又發現沙盒的 OpenCV 5.x 沒有 `CascadeClassifier`，因此將 requirements 鎖定為 `opencv < 5`。

### 階段 2：Python 版本
- **Prompt：** "wait my python right now 3.14 I want to change to 3.11 first"
- **AI 產出：** 說明 3.14 與 3.11 可並存，用 `py -3.11 -m venv .venv` 建立虛擬環境，並在 VS Code 選擇該直譯器。
- **Refinement：** 無（概念澄清）。

---

## Module 1 – Image loading（功能 1）
- **Prompt：** （上傳截圖：分頁 1 的中繼資料表）
- **AI 產出：** 確認 JPG 的中繼資料正確（1280×720、RGB、3 通道、8 bit、132.97 KB），建議再補測 PNG 與 BMP。
- **Refinement：** 無程式修改；補測 PNG／BMP 的截圖列為待辦。

---

## Module 2 – Color spaces & histogram（功能 2）

### 階段 11：色彩空間與直方圖檢查
- **Prompt：** （上傳 3 張截圖：Original、RGB channels、YUV channels）
- **AI 產出：** 確認模式切換與直方圖同步運作；解釋 U／V 直方圖集中在 128 的原因，以及縱軸尺度不同的原因。
- **Refinement：** AI 提出可選的「僅供顯示的對比拉伸」，學生未採用。

---

## Module 3 – Kernel filtering & PSNR（功能 3）

### 階段 12：卷積濾波測試與名詞釐清
- **Prompt：** （上傳 3 張截圖：Gaussian、Sharpen、Edge）"gaussian, sharpen, edge btw what do you mean by 2x2 kernel and a kernel with huge values such as 1000"
- **AI 產出：** 指出 Gaussian 5×5 過曝是因為權重總和 256 且未正規化；說明 2×2 為偶數大小、沒有中心點；說明 1000 的核用來測試截斷（clip）。
- **Refinement：** Gaussian／Box blur 預設啟用權重總和正規化；未正規化且權重總和大於 1.5 時顯示亮度放大警告。

### 階段 13：邊界條件驗證
- **Prompt：** "2x2 kernel yes there is an error (Invalid kernel: Kernel size must be odd, got 2x2.) The picture is after I use a kernel with huge values"
- **AI 產出：** 解讀極端核的結果：輸出被截斷為 255，只有黑髮保留暗色，PSNR 8.61 dB；建議用權重總和正規化做對照。
- **Refinement：** 無。

---

## Module 4 – Face detection（功能 4）

### 階段 3：Haar 偵測結果檢討
- **Prompt：** （上傳截圖：Haar 在多人合照偵測到 3 個框）
- **AI 產出：** 指出 1 個誤報（筆筒 0.87）、3 人漏偵測，說明 frontal cascade 的限制。
- **Refinement：** 結果依信心度排序；加入 Haar 最低信心度滑桿；表格編號改為從 1 開始。

### 階段 4：安裝 MediaPipe 後的環境衝突
- **Prompt：** （貼上 `pip install mediapipe` 的輸出：mediapipe 1.2.2、並安裝 opencv-contrib-python 5.0.0.93）
- **AI 產出：** 發現連帶安裝的 OpenCV 5.0 會使 Haar 失效；網頁搜尋確認新版 MediaPipe 已移除舊的 `mp.solutions` API。
- **Refinement：** 改寫為 Tasks API（首次使用下載 `blaze_face_short_range.tflite`）；requirements 改為 `opencv-contrib-python>=4.9,<5`。

### 階段 5：MediaPipe 初次結果
- **Prompt：** （上傳截圖：信心度 0.10、3 個重疊框）＋ "step 3 output 4.14"
- **AI 產出：** 確認 OpenCV 為 4.14（正確）；分析分數低與框重疊的原因（模型輸入約 128×128、偏正臉）；提出 NMS。
- **Refinement：** 無（診斷）。

### 階段 6：重疊框問題
- **Prompt：** "but what should I do if overlap like that? because its suppose not overlap?"
- **AI 產出：** 實作 NMS（依 IoU 抑制重複框）與 IoU 門檻滑桿（預設 0.3）。
- **Refinement：** 新增單元測試 `test_nms_removes_overlap_keeps_separate`。

### 階段 7：沙灘照片的品質問題
- **Prompt：** （上傳截圖：沙灘照 MediaPipe 4 個偏移框）"uhmmm why like this but my teacher want this …"（附上考試的人臉偵測需求）
- **AI 產出：** 分析成因（小臉縮圖後約 10 像素、側臉、寬圖補邊）；實作切塊偵測（整張圖＋邊長為短邊與其一半的正方形視窗、步幅 60%）。
- **Refinement：** AI 無法在沙盒下載模型，明確告知切塊結果「未經實測」；學生實測後 3 張臉皆正確（0.91／0.81／0.78）。新增 `tile_origins` 與單元測試。

### 階段 8：決定以 MediaPipe 為主
- **Prompt：** "use mediapipe meanwhile when I use haar cascade, no face detected"
- **AI 產出：** 整理兩種偵測器的比較表；MediaPipe 設為預設偵測器。
- **Refinement：** NMS 移除數量改為說明文字，避免向下箭頭造成誤解。

### 階段 9：Haar 信心度篩選造成的誤解
- **Prompt：** （上傳截圖：Haar minNeighbors 2、Min confidence 0.70、0 張臉）
- **AI 產出：** 指出 0.70 的門檻會悄悄濾掉低權重候選，建議先設為 0 再比較。
- **Refinement：** 新增「找到 N 個候選，M 個被信心度濾除」的提示。

### 階段 10：Haar 放寬參數實測
- **Prompt：** （上傳 3 張截圖：Haar 預設 0 張臉；放寬後 4 個框）
- **AI 產出：** 判讀 4 個框皆為誤報（浪花 0.79／0.62／0.44、口袋 0.00）；歸納 Haar 信心度不可靠，放寬參數只增加誤報。
- **Refinement：** 無（作為報告比較表的資料來源）。

---

## Testing & deployment

### 階段 14：部署準備
- **Prompt：** （上傳截圖：分頁 1 中繼資料）
- **AI 產出：** 新增 `.gitignore`、`packages.txt`、README 部署章節，並提供 GitHub 與 Streamlit Cloud 步驟。
- **Refinement：** 無。

### 階段 15：部署錯誤排除
- **Prompt：**
  (a) （貼上雲端 `import cv2` 的 Traceback，兩次皆缺最後一行錯誤）
  (b) "To https://github.com/jennibr/image-toolkit-midterm.git ! [rejected] main -> main (fetch first) …"
  (c) "File … app.py, line 113 … default_norm = 1 if spatial_filter.PRESET_NORMALIZATION.get(preset) …"
  (d) "Detection failed: libEGL.so.1: cannot open shared object file: No such file or directory"
- **AI 產出：** (a) 日誌被截斷，AI 在 `app.py` 加入 import 失敗時顯示實際錯誤的診斷，並檢查 `packages.txt`；(b) 遠端有本機沒有的提交，用 `git pull --rebase origin main` 再 push；(c) 版本不同步（只更新了 app.py，沒更新 spatial_filter.py）；(d) MediaPipe 缺少 EGL 系統函式庫。
- **Refinement：** (a) 修正後 cv2 可載入；(c) 同步 `spatial_filter.py` 後解決；(d) `packages.txt` 改為 libgl1、libglib2.0-0、libegl1、libgles2，重新部署後 MediaPipe 可用。

### 階段 16：製作報告
- **Prompt：** "1. C113151163 陳甄妮　2.（線上展示連結）　4. traditional chinese"
- **AI 產出：** 以學生提供的截圖製作繁體中文報告；缺少的截圖以標示框註明。
- **Refinement：** 無。

### 階段 17：Word 版與截圖清單
- **Prompt：** "btw can you give me docx instead pdf and tell me what pic I should put inside"
- **AI 產出：** 同一份內容輸出為 `.docx`，列出已放入與還要補拍的圖片。
- **Refinement：** Word 驗證器發現 XML 元素順序不符規範，改為依 schema 順序插入後通過驗證。

### 階段 18：對照評分標準表（Rubric）
- **Prompt：** （上傳評分標準表截圖）"this is the rubric"
- **AI 產出：** 逐項對照找出缺口；新增兩項獨立驗證測試（卷積對照雙迴圈參考實作、PSNR 對照 `cv2.PSNR`）、範例影像選單、參數說明提示，並補強報告。
- **Refinement：** 單元測試增為 12 項且全數通過。