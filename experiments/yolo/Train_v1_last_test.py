from ultralytics import YOLO  # YOLOv8용 라이브러리
import cv2

# 모델 로드 (YOLOv8 방식)
MODEL_PATH = r"CleanCheck/models/Train_v1_last.pt"  # 모델 경로
model = YOLO(MODEL_PATH)  # YOLOv8 방식 모델 로드

# 웹캠 열기
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 모델로 예측 수행
    results = model(frame)

    # 결과 시각화
    annotated_frame = results[0].plot()

    # 화면 출력
    cv2.imshow("YOLO Hand Gesture Detection", annotated_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 정리
cap.release()
cv2.destroyAllWindows()
