import cv2
import mediapipe as mp
import numpy as np

# ----------------- 거리 계산 함수 -----------------
def distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

# ----------------- Pose 기반 손 할당 함수 (다중 검출 지원) -----------------
def assign_hand_detections_pose_based(detections, left_wrist_xy, right_wrist_xy):
    assigned_left = None
    assigned_right = None
    if len(detections) >= 2:
        best_sum = float('inf')
        best_pair = None
        # 두 검출 결과 중 왼손/오른손을 결정하는 최적의 조합 찾기
        for i in range(len(detections)):
            for j in range(i+1, len(detections)):
                d1_wrist = detections[i][0]  # 각 검출 결과의 첫번째 랜드마크(손목)
                d2_wrist = detections[j][0]
                sum_dist = distance(d1_wrist, left_wrist_xy) + distance(d2_wrist, right_wrist_xy)
                if sum_dist < best_sum:
                    best_sum = sum_dist
                    best_pair = (detections[i], detections[j])

                # 스왑한 경우도 체크
                sum_dist_swapped = distance(d1_wrist, right_wrist_xy) + distance(d2_wrist, left_wrist_xy)
                if sum_dist_swapped < best_sum:
                    best_sum = sum_dist_swapped
                    best_pair = (detections[j], detections[i])
        
        if best_pair is not None:
            assigned_left, assigned_right = best_pair

    elif len(detections) == 1:
        d_wrist = detections[0][0]
        if distance(d_wrist, left_wrist_xy) < distance(d_wrist, right_wrist_xy):
            assigned_left = detections[0]
        else:
            assigned_right = detections[0]

    return assigned_left, assigned_right

# ----------------- Kalman Filter 생성 함수 -----------------
def create_kalman_filter(x, y):
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

# ----------------- 전역 변수(칼만 필터 및 직전 좌표) -----------------
kalman_left = {}
kalman_right = {}
last_left_landmarks = None
last_right_landmarks = None
last_left_wrist = None
last_right_wrist = None

# ----------------- 손가락 연결 정보 (Mediapipe Hands 참고) -----------------
HAND_CONNECTIONS = frozenset([
    (0,1), (1,2), (2,3), (3,4),
    (0,5), (5,6), (6,7), (7,8),
    (5,9), (9,10), (10,11), (11,12),
    (9,13), (13,14), (14,15), (15,16),
    (13,17), (17,18), (18,19), (19,20)
])

# Mediapipe 초기화: 홀리스틱 모듈만 사용
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_h, image_w, _ = frame.shape

    # 1. Holistic 모듈을 통해 한번에 pose+손+얼굴 등 처리
    holistic_results = holistic.process(image_rgb)

    # 2. Pose 결과에서 왼/오른손 손목 좌표 추출
    left_wrist_xy, right_wrist_xy = None, None
    if holistic_results.pose_landmarks:
        plm = holistic_results.pose_landmarks.landmark
        lw = plm[15]  # 왼손목(포즈 랜드마크에서 index=15)
        rw = plm[16]  # 오른손목(포즈 랜드마크에서 index=16)
        left_wrist_xy = (int(lw.x * image_w), int(lw.y * image_h))
        right_wrist_xy = (int(rw.x * image_w), int(rw.y * image_h))

    # 3. 홀리스틱에서 제공하는 왼손/오른손 랜드마크를 detections 리스트에 저장
    detections = []
    if holistic_results.left_hand_landmarks:
        landmarks_dict = {}
        for idx, lm in enumerate(holistic_results.left_hand_landmarks.landmark):
            cx, cy = int(lm.x * image_w), int(lm.y * image_h)
            landmarks_dict[idx] = (cx, cy)
        detections.append(landmarks_dict)

    if holistic_results.right_hand_landmarks:
        landmarks_dict = {}
        for idx, lm in enumerate(holistic_results.right_hand_landmarks.landmark):
            cx, cy = int(lm.x * image_w), int(lm.y * image_h)
            landmarks_dict[idx] = (cx, cy)
        detections.append(landmarks_dict)

    # 4. Pose 기반 손 할당 (왼손/오른손 분류)
    assigned_left, assigned_right = None, None
    if (left_wrist_xy is not None and right_wrist_xy is not None) and (len(detections) > 0):
        assigned_left, assigned_right = assign_hand_detections_pose_based(detections, left_wrist_xy, right_wrist_xy)

    # 5. 각 손에 대해 Kalman Filter 업데이트
    predicted_left, last_left_landmarks = update_kalman_filters(kalman_left, assigned_left, last_left_landmarks)
    predicted_right, last_right_landmarks = update_kalman_filters(kalman_right, assigned_right, last_right_landmarks)

    # 6. 손 검출 실패 시, Pose 손목 좌표 이동량으로 보간
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

    # 7. 현재 손목 좌표 기록
    if left_wrist_xy is not None:
        last_left_wrist = left_wrist_xy
    if right_wrist_xy is not None:
        last_right_wrist = right_wrist_xy

    # 8. 시각화 (손 스켈레톤만 그리기) - 직접 정의한 HAND_CONNECTIONS 사용
    if predicted_left:
        for connection in HAND_CONNECTIONS:
            if connection[0] in predicted_left and connection[1] in predicted_left:
                cv2.line(frame, predicted_left[connection[0]], predicted_left[connection[1]], (121, 22, 76), 2)
        for pt in predicted_left.values():
            cv2.circle(frame, pt, 4, (121, 44, 250), -1)

    if predicted_right:
        for connection in HAND_CONNECTIONS:
            if connection[0] in predicted_right and connection[1] in predicted_right:
                cv2.line(frame, predicted_right[connection[0]], predicted_right[connection[1]], (245, 117, 66), 2)
        for pt in predicted_right.values():
            cv2.circle(frame, pt, 4, (245, 66, 230), -1)

    cv2.imshow("Hand Skeleton (Holistic Only + Kalman Filter + Interpolation)", frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
holistic.close()
cv2.destroyAllWindows()
