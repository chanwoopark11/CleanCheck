import cv2
from ultralytics import YOLO
from collections import deque, Counter
import time
import numpy as np
import mediapipe as mp

# --- 설정 변수 ---
HANDWASH_MODEL_PATH = r"CleanCheck/models/Train_v1_best.pt"
CLASS_NAMES_HANDWASH = ["0.Palm to Palm", "1.Back of Hands", "2.Interlaced Fingers", "3.Backs of Fingers", "4.Thumbs", "5.Fingertips and Nails"]

MAX_HISTORY_LEN = 20
MIN_FRAMES_FOR_STABLE_ACTION = 10
MIN_ACTION_DURATION_FOR_COUNT = 1.5

MIN_WRIST_MOVEMENT_THRESHOLD_MP = 10
MIN_WRIST_VISIBILITY_MP = 0.5
MAX_MEASUREMENT_DURATION_SEC = 120

# 새로운 기능 관련 설정
NO_MOVEMENT_THRESHOLD_SEC = 0.4 # 민감도 증가 (기존 0.5 -> 0.4)
RECOMMENDATION_TARGET_DURATION_SEC = 5.0
RECOMMENDATION_INTERVAL_SEC = 3.0 # 추천 업데이트 간격도 약간 줄임 (기존 5.0 -> 3.0)

# --- 초기화 ---
handwash_model = YOLO(HANDWASH_MODEL_PATH)
mp_holistic = mp.solutions.holistic
holistic_detector = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

action_durations = {name: 0.0 for name in CLASS_NAMES_HANDWASH}
action_counts = {name: 0 for name in CLASS_NAMES_HANDWASH}

history_handwash_action = deque(maxlen=MAX_HISTORY_LEN)
current_stable_handwash_action = None
current_stable_handwash_action_start_time = 0.0

prev_mp_left_wrist_pt = None
prev_mp_right_wrist_pt = None

measurement_start_time_total = time.time()
is_measuring = True
last_frame_time = time.time()

last_mp_movement_time = time.time()
is_user_actively_washing = True
show_wash_hands_prompt = False
current_recommendation_text = ""
last_recommendation_update_time = 0.0
# recommendation_idx_handwash 변수 제거

# --- 유틸리티 함수 ---
def calculate_euclidean_distance(pt1, pt2):
    if pt1 is None or pt2 is None: return float('inf')
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def determine_stable_action_class(current_history, class_list, min_frames_threshold):
    if not current_history or len(current_history) < min_frames_threshold: return None
    most_common_list = Counter(current_history).most_common(1)
    if not most_common_list: return None
    action_candidate, count = most_common_list[0]
    if action_candidate is not None and count >= min_frames_threshold: return action_candidate
    return None

def update_recommendations():
    global current_recommendation_text, last_recommendation_update_time
    
    if time.time() - last_recommendation_update_time < RECOMMENDATION_INTERVAL_SEC:
        return

    found_recommendation = False
    # CLASS_NAMES_HANDWASH 순서대로 5초 미만인 첫 번째 동작을 찾음
    for action_name in CLASS_NAMES_HANDWASH:
        if action_durations[action_name] < RECOMMENDATION_TARGET_DURATION_SEC:
            current_recommendation_text = f"추천: {action_name.split('.')[1].strip()} ({action_durations[action_name]:.1f}s)"
            found_recommendation = True
            break 
    
    if not found_recommendation:
        current_recommendation_text = "모든 단계 완료!" # 메시지 변경
    
    last_recommendation_update_time = time.time()


print("손 씻기 측정을 시작합니다. (민감도↑, 추천 방식 변경) 'q'를 누르면 종료.")
print(f"{MAX_MEASUREMENT_DURATION_SEC}초 동안 측정합니다...")

# --- 메인 루프 ---
while cap.isOpened() and is_measuring:
    ret, frame = cap.read()
    if not ret: print("웹캠 오류"); break

    current_time = time.time()
    delta_time = current_time - last_frame_time
    last_frame_time = current_time

    frame_height, frame_width, _ = frame.shape
    elapsed_time_total = current_time - measurement_start_time_total

    if elapsed_time_total >= MAX_MEASUREMENT_DURATION_SEC:
        is_measuring = False
        if current_stable_handwash_action is not None and current_stable_handwash_action_start_time > 0:
            stable_duration_for_count = current_time - current_stable_handwash_action_start_time
            if stable_duration_for_count >= MIN_ACTION_DURATION_FOR_COUNT:
                 action_counts[current_stable_handwash_action] += 1
        break

    # 1. 손씻기 동작 분류 (YOLO)
    current_frame_handwash_prediction = None
    handwash_results = handwash_model.predict(source=frame, conf=0.5, verbose=False)
    boxes_handwash = handwash_results[0].boxes
    if boxes_handwash:
        largest_box = max(boxes_handwash, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
        class_id = int(largest_box.cls[0])
        if 0 <= class_id < len(CLASS_NAMES_HANDWASH):
            current_frame_handwash_prediction = CLASS_NAMES_HANDWASH[class_id]
    
    history_handwash_action.append(current_frame_handwash_prediction)
    new_stable_handwash_action = determine_stable_action_class(history_handwash_action, CLASS_NAMES_HANDWASH, MIN_FRAMES_FOR_STABLE_ACTION)

    # 2. 손목 움직임 감지 (MediaPipe Holistic)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False
    mp_holistic_results = holistic_detector.process(frame_rgb)
    frame_rgb.flags.writeable = True

    movement_this_frame_mp = False
    current_mp_left_wrist_pt, current_mp_right_wrist_pt = None, None
    if mp_holistic_results.pose_landmarks:
        pose_landmarks = mp_holistic_results.pose_landmarks.landmark
        left_wrist_landmark = pose_landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value]
        right_wrist_landmark = pose_landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value]

        if left_wrist_landmark.visibility > MIN_WRIST_VISIBILITY_MP:
            current_mp_left_wrist_pt = (int(left_wrist_landmark.x * frame_width), int(left_wrist_landmark.y * frame_height))
            if prev_mp_left_wrist_pt:
                dist_left = calculate_euclidean_distance(prev_mp_left_wrist_pt, current_mp_left_wrist_pt)
                if MIN_WRIST_MOVEMENT_THRESHOLD_MP < dist_left < (frame_width / 3) : 
                    movement_this_frame_mp = True
        
        if not movement_this_frame_mp and right_wrist_landmark.visibility > MIN_WRIST_VISIBILITY_MP:
            current_mp_right_wrist_pt = (int(right_wrist_landmark.x * frame_width), int(right_wrist_landmark.y * frame_height))
            if prev_mp_right_wrist_pt:
                dist_right = calculate_euclidean_distance(prev_mp_right_wrist_pt, current_mp_right_wrist_pt)
                if MIN_WRIST_MOVEMENT_THRESHOLD_MP < dist_right < (frame_width / 3):
                     movement_this_frame_mp = True
    
    if movement_this_frame_mp:
        last_mp_movement_time = current_time
        is_user_actively_washing = True
        show_wash_hands_prompt = False 
        current_recommendation_text = "" 
    else:
        if current_time - last_mp_movement_time > NO_MOVEMENT_THRESHOLD_SEC:
            is_user_actively_washing = False
            show_wash_hands_prompt = True
            update_recommendations()

    prev_mp_left_wrist_pt = current_mp_left_wrist_pt if current_mp_left_wrist_pt else prev_mp_left_wrist_pt
    prev_mp_right_wrist_pt = current_mp_right_wrist_pt if current_mp_right_wrist_pt else prev_mp_right_wrist_pt

    # 3. 손씻기 동작 상태 변화 및 '시간' 누적
    if new_stable_handwash_action != current_stable_handwash_action:
        if current_stable_handwash_action is not None and current_stable_handwash_action_start_time > 0:
            stable_duration_for_count = current_time - current_stable_handwash_action_start_time
            if stable_duration_for_count >= MIN_ACTION_DURATION_FOR_COUNT:
                action_counts[current_stable_handwash_action] += 1
                print(f"*** HANDWASH COUNT: {current_stable_handwash_action} (for {stable_duration_for_count:.1f}s) ***")
        
        current_stable_handwash_action = new_stable_handwash_action
        if current_stable_handwash_action is not None:
            current_stable_handwash_action_start_time = current_time
            print(f"[{int(elapsed_time_total)}s] Stable Handwash: {current_stable_handwash_action}")
        else:
            current_stable_handwash_action_start_time = 0
            print(f"[{int(elapsed_time_total)}s] Stable Handwash: None / Unstable")
            if not is_measuring : # 측정 종료시가 아니라면, 안정동작 없을때도 추천
                 show_wash_hands_prompt = True 
                 update_recommendations()

    if current_stable_handwash_action is not None and is_user_actively_washing:
        action_durations[current_stable_handwash_action] += delta_time
            
    # --- 시각화 ---
    display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(display_frame, f"Time: {int(elapsed_time_total)}s / Total: {MAX_MEASUREMENT_DURATION_SEC}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(display_frame, f"Handwash: {current_stable_handwash_action.split('.')[1].strip() if current_stable_handwash_action else 'N/A'}", 
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1) # y위치, 크기 조정
    
    y_offset = 80 # 동작별 시간 표시 시작 y 위치
    for i, name in enumerate(CLASS_NAMES_HANDWASH):
        text = f"{name.split('.')[1].strip()}: {action_durations[name]:.1f}s"
        color = (255,255,255)
        if name == current_stable_handwash_action:
            color = (0,255,255) if is_user_actively_washing else (0,165,255)
        cv2.putText(display_frame, text, (10, y_offset + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1) # 크기, 간격 조정

    # 경고 및 추천 메시지
    if show_wash_hands_prompt:
        prompt_text = "!! 움직이며 손을 씻으세요 !!"
        (w_prompt, h_prompt), _ = cv2.getTextSize(prompt_text, cv2.FONT_HERSHEY_TRIPLEX, 0.6, 2)
        cv2.putText(display_frame, prompt_text, (frame_width//2 - w_prompt//2, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 255), 2) # 중앙 상단
        
        if current_recommendation_text:
            # 추천 메시지 오른쪽 상단 표시
            (w_rec, h_rec), _ = cv2.getTextSize(current_recommendation_text, cv2.FONT_HERSHEY_TRIPLEX, 0.5, 1)
            cv2.putText(display_frame, current_recommendation_text, (frame_width - w_rec - 10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 200, 255), 1)

    if movement_this_frame_mp:
        cv2.putText(display_frame, "MOVING", (frame_width - 100, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255),1) # 오른쪽 하단
    
    if current_mp_left_wrist_pt: cv2.circle(display_frame, current_mp_left_wrist_pt, 7, (0,255,0), -1)
    if current_mp_right_wrist_pt: cv2.circle(display_frame, current_mp_right_wrist_pt, 7, (0,0,255), -1)

    cv2.imshow("Hand Wash Monitor", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        if current_stable_handwash_action is not None and current_stable_handwash_action_start_time > 0:
            stable_duration_for_count = current_time - current_stable_handwash_action_start_time
            if stable_duration_for_count >= MIN_ACTION_DURATION_FOR_COUNT:
                 action_counts[current_stable_handwash_action] += 1
        is_measuring = False; print("사용자에 의해 측정이 중단되었습니다.")
        break

# --- 종료 및 결과 출력 ---
cap.release()
cv2.destroyAllWindows()
if holistic_detector: holistic_detector.close()

print("\n--- 최종 측정 결과 ---")
print(f"총 측정 시간: {elapsed_time_total:.2f} 초")
print("\n[손씻기 동작별 (YOLO Handwash Model)]")
print("각 동작별 총 유효 지속 시간 (움직임 있을 때만):")
for action_name in CLASS_NAMES_HANDWASH:
    print(f"- {action_name}: {action_durations.get(action_name, 0.0):.2f} 초")
print(f"\n각 동작별 수행 횟수 (최소 안정 시간 {MIN_ACTION_DURATION_FOR_COUNT}초 이상):")
for action_name in CLASS_NAMES_HANDWASH:
    print(f"- {action_name}: {action_counts.get(action_name, 0)} 회")