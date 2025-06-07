import cv2
import mediapipe as mp
import numpy as np
import math

# ----------------- 거리 계산 함수 -----------------
def distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

# ----------------- Pose 기반 손 할당 함수 -----------------
def assign_hand_detections_pose_based(detections, left_wrist_xy, right_wrist_xy):
    """
    Pose 모듈로부터 얻은 왼손목, 오른손목 좌표를 사용하여,
    detections 안의 손목(랜드마크 0)이 왼손인지 오른손인지 할당합니다.
    """
    assigned_left = None
    assigned_right = None

    if len(detections) == 2:
        d0_wrist = detections[0][0]
        d1_wrist = detections[1][0]

        dist_0_lw = distance(d0_wrist, left_wrist_xy)
        dist_0_rw = distance(d0_wrist, right_wrist_xy)
        dist_1_lw = distance(d1_wrist, left_wrist_xy)
        dist_1_rw = distance(d1_wrist, right_wrist_xy)

        if (dist_0_lw + dist_1_rw) <= (dist_0_rw + dist_1_lw):
            assigned_left = detections[0]
            assigned_right = detections[1]
        else:
            assigned_left = detections[1]
            assigned_right = detections[0]

    elif len(detections) == 1:
        d_wrist = detections[0][0]
        if distance(d_wrist, left_wrist_xy) < distance(d_wrist, right_wrist_xy):
            assigned_left = detections[0]
        else:
            assigned_right = detections[0]

    return assigned_left, assigned_right

# ----------------- Kalman Filter 생성 함수 -----------------
def create_kalman_filter(x, y):
    """
    2차원 좌표 (x, y) 추적을 위한 칼만 필터 생성.
    상태: [x, y, dx, dy], 측정: [x, y]
    """
    kf = cv2.KalmanFilter(4, 2, 0, cv2.CV_32F)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], dtype=np.float32)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    kf.statePost = np.array([[np.float32(x)],
                             [np.float32(y)],
                             [0],
                             [0]], dtype=np.float32)
    return kf

# ----------------- Kalman Filter 업데이트 함수 -----------------
def update_kalman_filters(kalman_dict, detection, last_landmarks):
    """
    각 손의 칼만 필터를 업데이트합니다.
    detection: 검출된 랜드마크 딕셔너리 (있을 경우)
    last_landmarks: 이전 프레임의 예측 결과 (검출 실패 시 활용)
    반환: (predicted, updated_last_landmarks)
    """
    if detection is not None:
        for idx, (cx, cy) in detection.items():
            if idx not in kalman_dict:
                kalman_dict[idx] = create_kalman_filter(cx, cy)
            else:
                kf = kalman_dict[idx]
                kf.predict()
                measurement = np.array([[np.float32(cx)], [np.float32(cy)]], dtype=np.float32)
                kf.correct(measurement)
        predicted = {}
        for idx, kf in kalman_dict.items():
            pred_x = int(kf.statePost[0])
            pred_y = int(kf.statePost[1])
            predicted[idx] = (pred_x, pred_y)
        last_landmarks = predicted
    else:
        predicted = last_landmarks if last_landmarks is not None else {}
    return predicted, last_landmarks

# ----------------- 전역 변수 -----------------
kalman_left = {}
kalman_right = {}
last_left_landmarks = None
last_right_landmarks = None
last_left_wrist = None
last_right_wrist = None

# Mediapipe 초기화: Hands와 Pose를 별도로 사용
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_h, image_w, _ = frame.shape

    # ----------------- Pose 처리: 왼손목/오른손목 좌표 획득 -----------------
    pose_results = pose.process(image_rgb)
    left_wrist_xy, right_wrist_xy = None, None
    if pose_results.pose_landmarks:
        plm = pose_results.pose_landmarks.landmark
        lw = plm[15]  # 왼손목 (인덱스 15)
        rw = plm[16]  # 오른손목 (인덱스 16)
        left_wrist_xy = (int(lw.x * image_w), int(lw.y * image_h))
        right_wrist_xy = (int(rw.x * image_w), int(rw.y * image_h))

    # ----------------- Hands 검출: 손 랜드마크 추출 -----------------
    hands_results = hands.process(image_rgb)
    detections = []
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            landmarks_dict = {}
            for idx, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * image_w), int(lm.y * image_h)
                landmarks_dict[idx] = (cx, cy)
            detections.append(landmarks_dict)

    # ----------------- Pose 기반 손 할당 (왼손/오른손 구분) -----------------
    assigned_left, assigned_right = None, None
    if (left_wrist_xy is not None) and (right_wrist_xy is not None) and (len(detections) > 0):
        assigned_left, assigned_right = assign_hand_detections_pose_based(detections, left_wrist_xy, right_wrist_xy)

    # ----------------- 왼손 Kalman Filter 업데이트 -----------------
    predicted_left, last_left_landmarks = update_kalman_filters(kalman_left, assigned_left, last_left_landmarks)
    # ----------------- 오른손 Kalman Filter 업데이트 -----------------
    predicted_right, last_right_landmarks = update_kalman_filters(kalman_right, assigned_right, last_right_landmarks)

    # ----------------- 손 검출 실패 시, 손목 이동 기반 보간 -----------------
    if assigned_left is None and left_wrist_xy is not None and last_left_landmarks is not None and last_left_wrist is not None:
        dx = left_wrist_xy[0] - last_left_wrist[0]
        dy = left_wrist_xy[1] - last_left_wrist[1]
        predicted_left = {idx: (pt[0] + dx, pt[1] + dy) for idx, pt in last_left_landmarks.items()}
        last_left_landmarks = predicted_left

    if assigned_right is None and right_wrist_xy is not None and last_right_landmarks is not None and last_right_wrist is not None:
        dx = right_wrist_xy[0] - last_right_wrist[0]
        dy = right_wrist_xy[1] - last_right_wrist[1]
        predicted_right = {idx: (pt[0] + dx, pt[1] + dy) for idx, pt in last_right_landmarks.items()}
        last_right_landmarks = predicted_right

    # ----------------- 현재 손목 좌표 저장 (다음 프레임 대비) -----------------
    if left_wrist_xy is not None:
        last_left_wrist = left_wrist_xy
    if right_wrist_xy is not None:
        last_right_wrist = right_wrist_xy

    # ----------------- 시각화: 왼손 스켈레톤 -----------------
    if predicted_left:
        for connection in mp_hands.HAND_CONNECTIONS:
            if connection[0] in predicted_left and connection[1] in predicted_left:
                cv2.line(frame, predicted_left[connection[0]], predicted_left[connection[1]], (121, 22, 76), 2)
        for pt in predicted_left.values():
            cv2.circle(frame, pt, 4, (121, 44, 250), -1)

    # ----------------- 시각화: 오른손 스켈레톤 -----------------
    if predicted_right:
        for connection in mp_hands.HAND_CONNECTIONS:
            if connection[0] in predicted_right and connection[1] in predicted_right:
                cv2.line(frame, predicted_right[connection[0]], predicted_right[connection[1]], (245, 117, 66), 2)
        for pt in predicted_right.values():
            cv2.circle(frame, pt, 4, (245, 66, 230), -1)

    # ----------------- (옵션) Pose 시각화 -----------------
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("손 스켈레톤 (Kalman Filter + 보간)", frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
pose.close()
cv2.destroyAllWindows()
