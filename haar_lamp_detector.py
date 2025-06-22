import cv2
import os

# === 設定參數 ===
CASCADE_PATH = "model/cascade.xml"
TEST_DIR = "test_data"
WINDOW_NAME = "Lamp Detection"

# 載入分類器
cascade = cv2.CascadeClassifier(CASCADE_PATH)
if cascade.empty():
    raise IOError("無法載入 cascade.xml，請確認路徑與檔案是否存在")

# 取得所有圖片路徑
image_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
image_files.sort()  # 保持順序

index = 0
while index < len(image_files):
    filename = image_files[index]
    img_path = os.path.join(TEST_DIR, filename)
    img = cv2.imread(img_path)

    if img is None:
        print(f"無法載入圖片: {filename}")
        index += 1
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 執行偵測（參數之後可以再調整）
    boxes = cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=2,
        minSize=(50, 300),
        # maxSize=(200, 400)  # 可依圖片大小再調整
    )
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    for (x, y, w, h) in boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 5)

    cv2.imshow(WINDOW_NAME, img)
    print(f"顯示圖片：{filename}，共偵測到 {len(boxes)} 個路燈")
    key = cv2.waitKey(0)

    if key == 27:  # ESC 鍵退出
        break
    elif key == 32:  # 空白鍵下一張
        index += 1

cv2.destroyAllWindows()
