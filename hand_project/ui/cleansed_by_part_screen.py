# ui/cleansed_by_part_screen.py
import customtkinter
import cv2
import mediapipe as mp
import numpy as np
import time
from PIL import Image

class CleansedByPartScreen(customtkinter.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0) # Title
        self.grid_rowconfigure(1, weight=1) # Video
        self.grid_rowconfigure(2, weight=0) # Percentage Labels
        self.grid_rowconfigure(3, weight=0) # Controls

        self.title_label = customtkinter.CTkLabel(self, text="부위별 손씻기 실시간 분석",
                                                  font=customtkinter.CTkFont(size=24, weight="bold"))
        self.title_label.grid(row=0, column=0, pady=(10,5), padx=20, sticky="n")

        self.video_label = customtkinter.CTkLabel(self, text="웹캠 로딩 중...")
        self.video_label.grid(row=1, column=0, sticky="nsew", padx=20, pady=5)

        # 퍼센테이지 표시를 위한 프레임 및 레이블
        self.percentage_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        self.percentage_frame.grid(row=2, column=0, pady=(0,5), padx=20, sticky="ew")
        self.percentage_frame.grid_columnconfigure((0,1), weight=1)
        self.left_hand_percentage_label = customtkinter.CTkLabel(self.percentage_frame, text="왼손: 0.0%",
                                                                 font=customtkinter.CTkFont(size=16))
        self.left_hand_percentage_label.grid(row=0, column=0, sticky="e", padx=10)
        self.right_hand_percentage_label = customtkinter.CTkLabel(self.percentage_frame, text="오른손: 0.0%",
                                                                  font=customtkinter.CTkFont(size=16))
        self.right_hand_percentage_label.grid(row=0, column=1, sticky="w", padx=10)


        self.controls_frame = customtkinter.CTkFrame(self)
        self.controls_frame.grid(row=3, column=0, pady=(0,10), padx=20, sticky="ew")
        self.controls_frame.grid_columnconfigure(0, weight=1)

        self.stop_button = customtkinter.CTkButton(self.controls_frame, text="분석 중지 및 결과 보기",
                                                   command=self.stop_and_show_results)
        self.stop_button.grid(row=0, column=0, padx=5, pady=5)

        self.cap = None
        self.mp_holistic = None
        self.holistic_detector = None
        self.is_running = False
        self.start_time = 0

        # 상태 변수 초기화
        self._initialize_state_variables()

    def _initialize_state_variables(self):
        self.kalman_left, self.kalman_right = {}, {}
        self.last_left_landmarks, self.last_right_landmarks = None, None
        self.last_left_wrist, self.last_right_wrist = None, None
        self.last_known_facing_left_is_palm = True
        self.last_known_facing_right_is_palm = True

        self.cleansed = {
            "Left": {"palm": [False]*21, "back": [False]*21},
            "Right": {"palm": [False]*21, "back": [False]*21}
        }
        self.contact_timer = {
            "Left": {"palm": [0.0]*21, "back": [0.0]*21},
            "Right": {"palm": [0.0]*21, "back": [0.0]*21}
        }
        self.left_percentage = 0.0
        self.right_percentage = 0.0


    def on_show(self):
        print("CleansedByPartScreen: 화면 표시 및 분석 시작")
        self.is_running = True
        self._initialize_state_variables()
        self.start_time = time.time()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.video_label.configure(text="웹캠을 열 수 없습니다.")
            self.is_running = False
            return

        self.mp_holistic = mp.solutions.holistic
        self.holistic_detector = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.left_hand_percentage_label.configure(text="왼손: 0.0%")
        self.right_hand_percentage_label.configure(text="오른손: 0.0%")
        self.process_video_frame()

    def stop_measurement(self):
        print("CleansedByPartScreen: 분석 중지")
        self.is_running = False
        final_duration = time.time() - self.start_time if self.start_time > 0 else 0

        if self.cap:
            self.cap.release()
            self.cap = None
        if self.holistic_detector:
            self.holistic_detector.close()
            self.holistic_detector = None

        self._calculate_and_update_percentages() # 최종 퍼센테이지 계산

        results = {
            "type": "cleansed_by_part",
            "left_percentage": self.left_percentage,
            "right_percentage": self.right_percentage,
            "total_time": final_duration
        }
        self.controller.set_handwash_results(results) # App 클래스에 결과 전달
        return results

    def stop_and_show_results(self):
        results = self.stop_measurement()
        self.controller.handle_action("손씻기 결과 실행 요청", data=results)


    def _distance(self, pt1, pt2):
        return np.linalg.norm(np.array(pt1) - np.array(pt2))

    def _estimate_palm_direction(self, landmarks_dict, label):
        if not landmarks_dict or not (0 in landmarks_dict and 5 in landmarks_dict and 17 in landmarks_dict):
            return True

        wrist = np.array(landmarks_dict[0])
        index_mcp = np.array(landmarks_dict[5])
        pinky_mcp = np.array(landmarks_dict[17])
        v1 = index_mcp - wrist
        v2 = pinky_mcp - wrist
        normal_z = v1[0] * v2[1] - v1[1] * v2[0]
        if label == "Left":
            return normal_z < 0
        else: # Right
            return normal_z > 0

    def _create_kalman_filter(self, x, y):
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
        kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        kf.statePost = np.array([[x],[y],[0],[0]], dtype=np.float32)
        return kf

    def _update_kalman_filters(self, kalman_dict, detection, last_landmarks_input):
        current_kalman_output = {}
        updated_last_landmarks_for_next_frame = last_landmarks_input if last_landmarks_input is not None else {}

        if detection is not None:
            for idx, (cx, cy) in detection.items():
                if idx not in kalman_dict:
                    kalman_dict[idx] = self._create_kalman_filter(cx, cy)
                kf = kalman_dict[idx]
                kf.predict()
                kf.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))
                current_kalman_output[idx] = (int(kf.statePost[0]), int(kf.statePost[1]))
            updated_last_landmarks_for_next_frame = current_kalman_output
        else:
            if kalman_dict:
                temp_output = {}
                for idx, kf in kalman_dict.items():
                    kf.predict()
                    temp_output[idx] = (int(kf.statePost[0]), int(kf.statePost[1]))
                current_kalman_output = temp_output
                updated_last_landmarks_for_next_frame = temp_output
        return current_kalman_output, updated_last_landmarks_for_next_frame


    def _update_cleansed_state(self, final_left_lm, final_right_lm, facing_left_is_palm, facing_right_is_palm):
        now = time.time()
        p1_orientation = "palm" if facing_left_is_palm else "back"
        p2_orientation = "palm" if facing_right_is_palm else "back"
        contact_dist_threshold = 40
        contact_time_threshold = 0.3

        for i in range(21):
            left_lm_exists = final_left_lm and i in final_left_lm
            right_lm_exists = final_right_lm and i in final_right_lm

            if left_lm_exists and right_lm_exists:
                d = self._distance(final_left_lm[i], final_right_lm[i])
                if d < contact_dist_threshold:
                    if self.contact_timer["Left"][p1_orientation][i] == 0:
                        self.contact_timer["Left"][p1_orientation][i] = now
                    if now - self.contact_timer["Left"][p1_orientation][i] >= contact_time_threshold:
                        self.cleansed["Left"][p1_orientation][i] = True
                    
                    if self.contact_timer["Right"][p2_orientation][i] == 0:
                        self.contact_timer["Right"][p2_orientation][i] = now
                    if now - self.contact_timer["Right"][p2_orientation][i] >= contact_time_threshold:
                        self.cleansed["Right"][p2_orientation][i] = True
                else:
                    self.contact_timer["Left"][p1_orientation][i] = 0
                    self.contact_timer["Right"][p2_orientation][i] = 0
            else:
                self.contact_timer["Left"]["palm"][i] = 0
                self.contact_timer["Left"]["back"][i] = 0
                self.contact_timer["Right"]["palm"][i] = 0
                self.contact_timer["Right"]["back"][i] = 0
        
        # 매 프레임 세척 상태 업데이트 후 퍼센테이지 다시 계산 및 UI 업데이트
        self._calculate_and_update_percentages()


    def _calculate_and_update_percentages(self):
        num_landmarks_per_hand = 21
        # 각 랜드마크당 손바닥, 손등 2개의 면을 고려
        total_countable_surfaces_per_hand = num_landmarks_per_hand * 2 # 42

        # Left hand
        left_cleansed_surfaces_count = 0
        for i in range(num_landmarks_per_hand): # 0부터 20까지의 랜드마크 인덱스
            if self.cleansed["Left"]["palm"][i]:
                left_cleansed_surfaces_count += 1
            if self.cleansed["Left"]["back"][i]:
                left_cleansed_surfaces_count += 1
        
        if total_countable_surfaces_per_hand > 0:
            self.left_percentage = (left_cleansed_surfaces_count / total_countable_surfaces_per_hand) * 100
        else:
            self.left_percentage = 0.0

        # Right hand
        right_cleansed_surfaces_count = 0
        for i in range(num_landmarks_per_hand): # 0부터 20까지의 랜드마크 인덱스
            if self.cleansed["Right"]["palm"][i]:
                right_cleansed_surfaces_count += 1
            if self.cleansed["Right"]["back"][i]:
                right_cleansed_surfaces_count += 1

        if total_countable_surfaces_per_hand > 0:
            self.right_percentage = (right_cleansed_surfaces_count / total_countable_surfaces_per_hand) * 100
        else:
            self.right_percentage = 0.0

        # UI 레이블 업데이트
        self.left_hand_percentage_label.configure(text=f"왼손: {self.left_percentage:.1f}%")
        self.right_hand_percentage_label.configure(text=f"오른손: {self.right_percentage:.1f}%")


    def _draw_landmarks_with_contact(self, img, landmarks_dict, label, is_palm_facing):
        if not landmarks_dict: return img
        part_orientation = "palm" if is_palm_facing else "back"

        for i, (x, y) in landmarks_dict.items():
            if i >= 21: continue
            color = (0,0,0)
            if self.cleansed[label][part_orientation][i]:
                color = (0, 255, 0) if part_orientation == "palm" else (255, 165, 0)
            else:
                color = (0, 0, 255)
            cv2.circle(img, (x, y), 7, color, -1)
            cv2.circle(img, (x,y), 7, (200,200,200), 1)
        return img

    def process_video_frame(self):
        if not self.is_running or not self.cap or not self.cap.isOpened() or not self.holistic_detector:
            return

        ret, frame = self.cap.read()
        if not ret:
            if self.is_running: self.after(10, self.process_video_frame)
            return

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        results_holistic = self.holistic_detector.process(image_rgb)

        current_left_wrist_xy, current_right_wrist_xy = None, None
        detected_landmarks_left_this_frame, detected_landmarks_right_this_frame = None, None
        current_facing_left_is_palm = self.last_known_facing_left_is_palm
        current_facing_right_is_palm = self.last_known_facing_right_is_palm

        if results_holistic.pose_landmarks:
            lw = results_holistic.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_WRIST]
            if lw.visibility > 0.1: current_left_wrist_xy = (int(lw.x * w), int(lw.y * h))
            rw = results_holistic.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.RIGHT_WRIST]
            if rw.visibility > 0.1: current_right_wrist_xy = (int(rw.x * w), int(rw.y * h))

        if results_holistic.left_hand_landmarks:
            landmarks_mp = results_holistic.left_hand_landmarks
            detected_landmarks_left_this_frame = {idx: (int(lm.x*w), int(lm.y*h)) for idx, lm in enumerate(landmarks_mp.landmark)}
            current_facing_left_is_palm = self._estimate_palm_direction(detected_landmarks_left_this_frame, "Left")
            self.last_known_facing_left_is_palm = current_facing_left_is_palm

        if results_holistic.right_hand_landmarks:
            landmarks_mp = results_holistic.right_hand_landmarks
            detected_landmarks_right_this_frame = {idx: (int(lm.x*w), int(lm.y*h)) for idx, lm in enumerate(landmarks_mp.landmark)}
            current_facing_right_is_palm = self._estimate_palm_direction(detected_landmarks_right_this_frame, "Right")
            self.last_known_facing_right_is_palm = current_facing_right_is_palm

        kalman_output_left, next_last_left_landmarks = self._update_kalman_filters(self.kalman_left, detected_landmarks_left_this_frame, self.last_left_landmarks)
        kalman_output_right, next_last_right_landmarks = self._update_kalman_filters(self.kalman_right, detected_landmarks_right_this_frame, self.last_right_landmarks)

        final_left_landmarks_for_frame = kalman_output_left
        final_right_landmarks_for_frame = kalman_output_right

        if detected_landmarks_left_this_frame is None and current_left_wrist_xy and final_left_landmarks_for_frame and self.last_left_wrist:
            dx, dy = current_left_wrist_xy[0] - self.last_left_wrist[0], current_left_wrist_xy[1] - self.last_left_wrist[1]
            final_left_landmarks_for_frame = {idx: (pt[0]+dx, pt[1]+dy) for idx, pt in final_left_landmarks_for_frame.items()}
            next_last_left_landmarks = final_left_landmarks_for_frame

        if detected_landmarks_right_this_frame is None and current_right_wrist_xy and final_right_landmarks_for_frame and self.last_right_wrist:
            dx, dy = current_right_wrist_xy[0] - self.last_right_wrist[0], current_right_wrist_xy[1] - self.last_right_wrist[1]
            final_right_landmarks_for_frame = {idx: (pt[0]+dx, pt[1]+dy) for idx, pt in final_right_landmarks_for_frame.items()}
            next_last_right_landmarks = final_right_landmarks_for_frame

        self.last_left_landmarks = next_last_left_landmarks
        self.last_right_landmarks = next_last_right_landmarks

        # 세척 상태 업데이트는 유효한 랜드마크가 있을 때만 시도
        if final_left_landmarks_for_frame or final_right_landmarks_for_frame:
            self._update_cleansed_state(final_left_landmarks_for_frame, final_right_landmarks_for_frame, current_facing_left_is_palm, current_facing_right_is_palm)
        else: # 양손 다 랜드마크가 없으면 퍼센테이지 레이블만 업데이트 (0%로 유지되거나 이전 값)
            self._calculate_and_update_percentages()


        display_frame_bgr = frame.copy()
        display_frame_bgr = self._draw_landmarks_with_contact(display_frame_bgr, final_left_landmarks_for_frame, "Left", current_facing_left_is_palm)
        display_frame_bgr = self._draw_landmarks_with_contact(display_frame_bgr, final_right_landmarks_for_frame, "Right", current_facing_right_is_palm)

        if current_left_wrist_xy: self.last_left_wrist = current_left_wrist_xy
        if current_right_wrist_xy: self.last_right_wrist = current_right_wrist_xy

        try:
            img_rgb = cv2.cvtColor(display_frame_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            label_w, label_h = self.video_label.winfo_width(), self.video_label.winfo_height()
            if label_w < 10 or label_h < 10:
                size = (img_pil.width, img_pil.height) # 레이블 크기 아직 확정 안됨
            else:
                img_aspect = img_pil.width / img_pil.height
                if label_w / label_h > img_aspect:
                    target_h = label_h
                    target_w = int(label_h * img_aspect)
                else:
                    target_w = label_w
                    target_h = int(label_w / img_aspect)
                size=(target_w, target_h) if target_w > 0 and target_h > 0 else (img_pil.width, img_pil.height)


            img_tk = customtkinter.CTkImage(light_image=img_pil, dark_image=img_pil, size=size)
            self.video_label.configure(image=img_tk, text="")
            self.video_label.image = img_tk
        except Exception as e:
            print(f"Error updating video label in CleansedByPartScreen: {e}")

        if self.is_running:
            self.after(10, self.process_video_frame) # 프레임 처리 간격 (ms)