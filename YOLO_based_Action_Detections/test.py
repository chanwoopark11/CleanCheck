import cv2
from ultralytics import YOLO
from collections import deque, Counter

# YOLO 모델 로드 (학습된 모델(model_best.pt) 경로로 수정)
model = YOLO(r"D:\Download\best.pt")  # 사용자 지정 경로

# 최근 프레임의 예측 결과를 저장할 deque (최대 10개 저장)
history = deque(maxlen=10)

# 클래스 ID에 대응하는 제스처 이름 리스트
class_names = [
    "0.Palm to Palm",
    "1.Back of Hands",
    "2.Interlaced Fingers",
    "3.Backs of Fingers",
    "4.Thumbs",
    "5.Fingertips and Nails"
]

# 웹캠 캡처 객체 생성 (0은 기본 카메라를 의미)
cap = cv2.VideoCapture(0)

# 프레임 처리 루프
while cap.isOpened():
    ret, frame = cap.read()  # 프레임 읽기
    if not ret:
        break  # 프레임 읽기 실패 시 루프 종료

    # YOLO 모델을 이용한 객체 탐지
    results = model.predict(source=frame, conf=0.5, verbose=False)
    boxes = results[0].boxes  # 탐지된 bounding box 목록

    predicted_class = None  # 현재 프레임의 예측 클래스 초기화

    # 최소 하나의 박스가 탐지되었을 경우
    if boxes:
        # 가장 큰 박스를 기준으로 대표 클래스 선택 
        largest_box = max(
            boxes, 
            key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1])
        )

        class_id = int(largest_box.cls[0])  # 클래스 ID 추출
        predicted_class = class_names[class_id]  # 클래스 이름 매핑
        history.append(predicted_class)  # 최근 결과 히스토리에 추가

        # 박스 시각화
        x1, y1, x2, y2 = map(int, largest_box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

        # 클래스 이름을 바운딩 박스 위에 표시
        cv2.putText(frame, predicted_class, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

    # 다수결 후처리: 최근 10프레임 기준으로 가장 많이 나온 제스처 왼쪽 상단에 표시
    if history:
        final_prediction = Counter(history).most_common(1)[0][0]
        cv2.putText(frame, f"Smoothed: {final_prediction}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2) 

    # 결과 프레임 화면에 표시
    cv2.imshow("YOLO + Gesture Smoothing", frame)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()

