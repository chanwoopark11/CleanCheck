import cv2
from ultralytics import YOLO
from collections import deque, Counter

# YOLO 모델 불러오기 (너의 손 동작 인식용 모델 경로로 수정)
model = YOLO(r"CleanCheck/models/Train_v1_best.pt")

# 최근 예측 결과를 저장할 deque
history = deque(maxlen=10)  # 최근 10프레임 저장

# 클래스 이름 리스트 (너의 6가지 손동작 이름으로 수정)
class_names = ["0.Palm to Palm", "1.Back of Hands", "2.Interlaced Fingers", "3.Backs of Fingers", "4.Thumbs", "5.Fingertips and Nails"]

# 웹캠 열기
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 예측
    results = model.predict(source=frame, conf=0.5, verbose=False)
    boxes = results[0].boxes

    predicted_class = None

    # 가장 큰 box 하나만 사용 (보통 손 하나 기준)
    if boxes:
        # class id를 가져옴 (가장 큰 박스 기준)
        largest_box = max(boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
        class_id = int(largest_box.cls[0])
        predicted_class = class_names[class_id]
        history.append(predicted_class)

        # bbox 시각화
        x1, y1, x2, y2 = map(int, largest_box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(frame, predicted_class, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

    # 후처리: 최근 history 기반 다수결
    if history:
        final_prediction = Counter(history).most_common(1)[0][0]
        cv2.putText(frame, f"Smoothed: {final_prediction}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("YOLO + Gesture Smoothing", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




