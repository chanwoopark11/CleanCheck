import cv2
import mediapipe as mp
import numpy as np
import time

# ----------- 유틸 함수 -----------

# 두 점 사이의 유클리드 거리 계산
def distance(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

# 손바닥 방향을 추정 (손바닥인지 손등인지 구분)
def estimate_palm_direction(landmarks, label):
    if 0 in landmarks and 5 in landmarks and 17 in landmarks:
        wrist = np.array(landmarks[0])
        index = np.array(landmarks[5])
        pinky = np.array(landmarks[17])
        v1 = index - wrist
        v2 = pinky - wrist
        normal = np.cross(np.array([*v1, 0]), np.array([*v2, 0]))
        z = normal[2]

        # 오른손은 z > 0일 때 손바닥, 왼손은 반대 방향
        if label == "Left":
            return z < 0
        else:
            return z > 0
    return True  # 정보가 부족할 경우 기본값 True (손바닥)

# 칼만 필터 초기화 (입력: 초기 좌표)
def create_kalman_filter(x, y):
    kf = cv2.KalmanFilter(4, 2)  # 상태 4차원 (x, y, vx, vy), 측정 2차원 (x, y)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], dtype=np.float32)
    kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
    kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
    return kf

# 칼만 필터로 손 관절 위치 보정 및 추정
def update_kalman_filters(kalman_dict, detection, last_landmarks):
    if detection is not None:
        for idx, (cx, cy) in detection.items():
            if idx not in kalman_dict:
                kalman_dict[idx] = create_kalman_filter(cx, cy)
            kf = kalman_dict[idx]
            kf.predict()
            kf.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))
        predicted = {idx: (int(kf.statePost[0]), int(kf.statePost[1])) for idx, kf in kalman_dict.items()}
        last_landmarks = predicted
    else:
        predicted = last_landmarks if last_landmarks is not None else {}
    return predicted, last_landmarks

# ----------- 시각화 및 상태 저장 -----------

# 세정 여부 상태 저장: 손의 각 관절별로 손바닥/손등 기준
cleansed = {
    "Left": {"palm": [False]*21, "back": [False]*21},
    "Right": {"palm": [False]*21, "back": [False]*21}
}
# 접촉 시작 시간을 저장하여 지속 시간 측정
contact_timer = {
    "Left": {"palm": [0]*21, "back": [0]*21},
    "Right": {"palm": [0]*21, "back": [0]*21}
}

# 손가락끼리 일정 시간 이상 접촉했을 경우 "씻음 처리"
def update_cleansed_state(pred_left, pred_right, facing_left, facing_right):
    now = time.time()
    p1 = "palm" if facing_left else "back"
    p2 = "palm" if facing_right else "back"
    for i in range(21):
        if i in pred_left and i in pred_right:
            d = distance(pred_left[i], pred_right[i])
            if d < 80:  # 일정 거리 이하일 경우 접촉으로 간주
                if contact_timer["Left"][p1][i] == 0:
                    contact_timer["Left"][p1][i] = now
                if contact_timer["Right"][p2][i] == 0:
                    contact_timer["Right"][p2][i] = now
                # 1초 이상 접촉 시 세정 완료 처리
                if now - contact_timer["Left"][p1][i] > 1:
                    cleansed["Left"][p1][i] = True
                if now - contact_timer["Right"][p2][i] > 1:
                    cleansed["Right"][p2][i] = True
            else:
                # 접촉이 멈추면 타이머 초기화
                contact_timer["Left"][p1][i] = 0
                contact_timer["Right"][p2][i] = 0

# 손 관절 시각화: 세정된 관절은 초록/파랑, 안 된 관절은 빨간색
def draw_landmarks_with_contact(img, landmarks, label, facing):
    part = "palm" if facing else "back"
    for i, (x, y) in landmarks.items():
        if cleansed[label][part][i]:
            color = (0, 255, 0) if part == "palm" else (255, 0, 0)  # 손바닥=초록, 손등=파랑
        else:
            color = (0, 0, 255)  # 미세정 = 빨강
        cv2.circle(img, (x, y), 10, color, -1)
    if 0 in landmarks:
        cv2.putText(img, f"{label} - {part}", landmarks[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

# ----------- 초기화 -----------

kalman_left, kalman_right = {}, {}  # 칼만 필터 저장
last_left_landmarks, last_right_landmarks = None, None
last_left_wrist, last_right_wrist = None, None

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# Mediapipe 초기화
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)  # 웹캠 열기

# ----------- 메인 루프 -----------

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # 좌우 반전 (거울 모드)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    # 포즈에서 손목 좌표 추출 (보간에 사용)
    left_wrist_xy, right_wrist_xy = None, None
    pose_results = pose.process(image_rgb)
    if pose_results.pose_landmarks:
        lw = pose_results.pose_landmarks.landmark[15]
        rw = pose_results.pose_landmarks.landmark[16]
        left_wrist_xy = (int(lw.x * w), int(lw.y * h))
        right_wrist_xy = (int(rw.x * w), int(rw.y * h))

    results = hands.process(image_rgb)
    predicted_left, predicted_right = {}, {}

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[i].classification[0].label  # 'Left' or 'Right'
            landmarks = {idx: (int(lm.x * w), int(lm.y * h)) for idx, lm in enumerate(hand_landmarks.landmark)}
            facing = estimate_palm_direction(landmarks, label)

            if label == "Left":
                predicted_left, last_left_landmarks = update_kalman_filters(kalman_left, landmarks, last_left_landmarks)
                update_cleansed_state(predicted_left, predicted_right, facing, True)
                draw_landmarks_with_contact(frame, predicted_left, "Left", facing)
            else:
                predicted_right, last_right_landmarks = update_kalman_filters(kalman_right, landmarks, last_right_landmarks)
                update_cleansed_state(predicted_left, predicted_right, True, facing)
                draw_landmarks_with_contact(frame, predicted_right, "Right", facing)

    # -------- 보간 처리: 손이 사라진 경우 이전 위치 + 손목 이동량 추정 --------

    if not predicted_left and left_wrist_xy and last_left_landmarks and last_left_wrist:
        dx = left_wrist_xy[0] - last_left_wrist[0]
        dy = left_wrist_xy[1] - last_left_wrist[1]
        predicted_left = {idx: (pt[0] + dx, pt[1] + dy) for idx, pt in last_left_landmarks.items()}
        last_left_landmarks = predicted_left
        draw_landmarks_with_contact(frame, predicted_left, "Left", True)

    if not predicted_right and right_wrist_xy and last_right_landmarks and last_right_wrist:
        dx = right_wrist_xy[0] - last_right_wrist[0]
        dy = right_wrist_xy[1] - last_right_wrist[1]
        predicted_right = {idx: (pt[0] + dx, pt[1] + dy) for idx, pt in last_right_landmarks.items()}
        last_right_landmarks = predicted_right
        draw_landmarks_with_contact(frame, predicted_right, "Right", True)

    # 마지막 손목 위치 갱신
    if left_wrist_xy:
        last_left_wrist = left_wrist_xy
    if right_wrist_xy:
        last_right_wrist = right_wrist_xy

    cv2.imshow("Hand Wash Visualizer (Enhanced)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------- 리소스 정리 -----------

cap.release()
hands.close()
pose.close()
cv2.destroyAllWindows()

