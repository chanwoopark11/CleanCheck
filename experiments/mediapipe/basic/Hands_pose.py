import cv2
import mediapipe as mp

# ----------------- MediaPipe 초기화 -----------------
# 공통으로 사용할 드로잉 유틸리티
mp_drawing = mp.solutions.drawing_utils

# 1. Pose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 2. Hands 모델 초기화
mp_hands = mp.solutions.hands
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

    # 2. 이미지 전처리 (좌우 반전 및 색상 변환)
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 성능 향상을 위해 이미지를 쓰기 불가(non-writable)로 설정
    image_rgb.flags.writeable = False

    # 3. 이미지에서 포즈와 손 동시 처리
    pose_results = pose.process(image_rgb)
    hands_results = hands.process(image_rgb)

    # 4. 결과 시각화
    # 다시 화면에 그리기 위해 이미지를 쓰기 가능(writable)으로 변경
    image_rgb.flags.writeable = True
    frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) # 원본 BGR 프레임으로 복원

    # 4-1. 포즈 랜드마크 그리기
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=pose_results.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(80, 22, 10), thickness=2, circle_radius=4),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(80, 44, 121), thickness=2, circle_radius=2)
        )

    # 4-2. 손 랜드마크 그리기
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
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
    cv2.imshow('MediaPipe Pose and Hands Combined', frame)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# ----------------- 자원 해제 -----------------
cap.release()
pose.close()
hands.close()
cv2.destroyAllWindows()