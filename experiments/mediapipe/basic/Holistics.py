import cv2
import mediapipe as mp

# ----------------- MediaPipe Holistic 초기화 -----------------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Holistic 객체 생성
# min_detection_confidence: 탐지 신뢰도
# min_tracking_confidence: 추적 신뢰도
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
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
    # MediaPipe 처리를 위해 이미지 성능 향상을 위해 쓰기 불가(non-writable)로 설정
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False

    # 3. MediaPipe Holistic을 이용해 전신 처리
    results = holistic.process(image_rgb)

    # 4. 결과 시각화
    # 다시 화면에 그리기 위해 이미지를 쓰기 가능(writable)으로 변경
    image_rgb.flags.writeable = True
    frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) # 원본 BGR 프레임으로 복원

    # 4-1. 얼굴 랜드마크 그리기 (Tesselation)
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=results.face_landmarks,
        connections=mp_holistic.FACEMESH_TESSELATION,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=(80, 110, 10), thickness=1, circle_radius=1),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(80, 256, 121), thickness=1, circle_radius=1)
    )

    # 4-2. 포즈(전신) 랜드마크 그리기
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=results.pose_landmarks,
        connections=mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=(80, 22, 10), thickness=2, circle_radius=4),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(80, 44, 121), thickness=2, circle_radius=2)
    )

    # 4-3. 왼손 랜드마크 그리기
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=results.left_hand_landmarks,
        connections=mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=(121, 22, 76), thickness=2, circle_radius=4),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(121, 44, 250), thickness=2, circle_radius=2)
    )

    # 4-4. 오른손 랜드마크 그리기
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=results.right_hand_landmarks,
        connections=mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=(245, 117, 66), thickness=2, circle_radius=4),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(245, 66, 230), thickness=2, circle_radius=2)
    )

    # 5. 화면에 결과 출력
    cv2.imshow('MediaPipe Holistic Basic', frame)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# ----------------- 자원 해제 -----------------
cap.release()
holistic.close()
cv2.destroyAllWindows()