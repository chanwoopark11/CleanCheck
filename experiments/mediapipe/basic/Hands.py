import cv2
import mediapipe as mp
import numpy as np

# ----------------- MediaPipe Hands 초기화 -----------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Hands 객체 생성
# max_num_hands: 최대 인식할 손의 개수
# min_detection_confidence: 탐지 신뢰도
# min_tracking_confidence: 추적 신뢰도
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ----------------- 웹캠 설정 -----------------
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # 1. 웹캠에서 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("카메라를 찾을 수 없습니다.")
        break

    # 2. 이미지 좌우 반전 및 BGR에서 RGB로 변환
    # 웹캠은 보통 좌우 반전되어 있으므로 사용자 시점과 맞추기 위해 flip 사용
    # MediaPipe는 RGB 이미지를 입력으로 받으므로 cvtColor로 색상 공간 변환
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 3. MediaPipe Hands를 이용해 손 처리
    results = hands.process(image_rgb)

    # 4. 결과 시각화
    # multi_hand_landmarks: 프레임에서 감지된 모든 손의 랜드마크
    if results.multi_hand_landmarks:
        # 감지된 손의 개수만큼 반복
        for hand_landmarks in results.multi_hand_landmarks:
            # 손 랜드마크와 연결선을 프레임에 그림
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(245, 117, 66), thickness=2, circle_radius=4),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(245, 66, 230), thickness=2, circle_radius=2)
            )

    # 5. 화면에 결과 출력
    cv2.imshow('MediaPipe Hands Basic', frame)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# ----------------- 자원 해제 -----------------
cap.release()
hands.close()
cv2.destroyAllWindows()