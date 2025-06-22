# Haar Training Project 3 - 路燈偵測器

本專案使用 OpenCV 的 Haar 級聯分類器 (Cascade Classifier) 技術，訓練一個可辨識路燈的分類器。整個流程涵蓋資料標註、樣本生成、模型訓練與測試展示。

---

## 📁 專案結構說明

| 資料夾 / 檔案名稱         | 說明 |
|--------------------------|------|
| `annotations/`           | 使用 LabelImg 標註後產出的 XML 標註檔 |
| `pos/`                   | 正樣本影像（含路燈） |
| `neg/`                   | 負樣本影像（不含路燈） |
| `test_data/`             | 測試用影像資料夾 |
| `model/`                 | 訓練完成的 Haar 級聯模型（`cascade.xml` 等） |
| `generate_pos_txt.py`    | 產生 `pos.txt`（訓練正樣本標註資料） |
| `generate_neg_txt.py`    | 產生 `neg.txt`（列出所有負樣本圖片） |
| `pos.txt`                | 儲存正樣本位置與標註框資訊的檔案 |
| `neg.txt`                | 儲存負樣本圖片路徑的檔案 |
| `pos.vec`                | 使用 `opencv_createsamples` 工具轉換出的向量格式正樣本 |
| `train_command.txt`      | 訓練模型時在command line 要輸入的指令 |
| `haar_lamp_detector.py`  | 測試用的偵測腳本，讀取 `cascade.xml` 並在圖片中畫出辨識結果 |
| `成果展示影片.mp4`        | 模型應用展示影片 |

---

## 🛠️ 使用方式

### 1. 產生訓練資料檔
```
python generate_pos_txt.py
python generate_neg_txt.py
```
### 2. 產生 .vec 檔（OpenCV 訓練格式）
```
opencv_createsamples -info pos.txt -num 218 -w 11 -h 55 -vec pos.vec
```
### 3. 開始訓練 Haar 級聯分類器
```
opencv_traincascade -data model -vec pos.vec -bg neg.txt -numPos 210 -numNeg 375 -numStages 30 -w 11 -h 55 -featureType HAAR
```
### 4. 執行測試程式
```
python haar_lamp_detector.py
按空白鍵切換下一張測試影像。
```
## 成果展示
請參見 成果展示影片.mp4，示範模型成功偵測路燈的表現。
