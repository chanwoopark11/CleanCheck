# ui/screens.py
import customtkinter
import cv2
from ultralytics import YOLO
from collections import deque, Counter
import time
import numpy as np
import mediapipe as mp
import tkinter # _tkinter.TclError를 명시적으로 다루기 위해 추가 (필수는 아님)

try:
    from PIL import Image, ImageTk # ImageTk도 필요합니다.
except ImportError:
    print("Pillow 라이브러리가 없습니다. 'pip install Pillow'로 설치해주세요.")
    exit()

class MenuScreen(customtkinter.CTkFrame): # 새로운 MenuScreen으로 교체
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.menu_options = [
            ("6단계 손씻기", lambda: controller.handle_action("6단계 손씻기 실행 요청")),
            ("손씻기 결과", lambda: controller.handle_action("손씻기 결과 실행 요청")), # 이전 '손씻기 확인' -> '손씻기 결과'로 명칭 변경
            ("설정", lambda: controller.handle_action("설정 화면으로 이동 요청")),
            ("종료", lambda: controller.handle_action("애플리케이션 종료 요청"))
        ]
        self.current_menu_index = 0

        self.grid_rowconfigure(0, weight=0) # 로고 행
        self.grid_rowconfigure(1, weight=1) # 메뉴 선택 행
        self.grid_rowconfigure(2, weight=0) # 하단 여백 또는 추가 정보 (옵션)

        self.grid_columnconfigure(0, weight=1) # 왼쪽 화살표
        self.grid_columnconfigure(1, weight=2) # 중앙 메뉴 텍스트
        self.grid_columnconfigure(2, weight=1) # 오른쪽 화살표

        self.logo_label = customtkinter.CTkLabel(
            self,
            text="Clean Check", # 로고 텍스트 변경
            font=customtkinter.CTkFont(size=30, weight="bold")
        )
        # 로고 위치를 왼쪽으로 변경 (sticky="w")
        self.logo_label.grid(row=0, column=0, columnspan=3, padx=20, pady=(40, 20), sticky="w") # [ 변경된 부분 ]

        # --- 간격 및 호버 효과 조정 ---
        arrow_hover_color = "gray16" if customtkinter.get_appearance_mode() == "Dark" else "gray85"
        text_hover_color = "gray18" if customtkinter.get_appearance_mode() == "Dark" else "gray90"


        self.left_button = customtkinter.CTkButton(
            self,
            text="◀",
            command=self._previous_menu,
            width=60,
            height=60,
            font=customtkinter.CTkFont(size=28),
            fg_color="transparent",
            hover_color=arrow_hover_color
        )
        self.left_button.grid(row=1, column=0, padx=(20, 5), pady=20, sticky="e")

        self.selected_menu_clickable_text = customtkinter.CTkButton(
            self,
            text=self.menu_options[self.current_menu_index][0],
            font=customtkinter.CTkFont(size=40, weight="bold"),
            command=self._execute_selected_menu,
            fg_color="transparent",
            hover_color=text_hover_color,
            text_color=("gray10", "gray90") # (light_mode_color, dark_mode_color)
        )
        self.selected_menu_clickable_text.grid(row=1, column=1, padx=10, pady=20, sticky="nsew")

        self.right_button = customtkinter.CTkButton(
            self,
            text="▶",
            command=self._next_menu,
            width=60,
            height=60,
            font=customtkinter.CTkFont(size=28),
            fg_color="transparent",
            hover_color=arrow_hover_color
        )
        self.right_button.grid(row=1, column=2, padx=(5, 20), pady=20, sticky="w")

        self._update_menu_display()

    def _update_menu_display(self):
        self.selected_menu_clickable_text.configure(text=self.menu_options[self.current_menu_index][0])

    def _previous_menu(self):
        self.current_menu_index = (self.current_menu_index - 1 + len(self.menu_options)) % len(self.menu_options)
        self._update_menu_display()

    def _next_menu(self):
        self.current_menu_index = (self.current_menu_index + 1) % len(self.menu_options)
        self._update_menu_display()

    def _execute_selected_menu(self):
        action_name, action_func = self.menu_options[self.current_menu_index]
        print(f"메뉴 실행: '{action_name}'")
        action_func()

class ExecutionScreen(customtkinter.CTkFrame):
    """
    손 씻기 과정을 실시간으로 모니터링하고 피드백을 제공하는 화면입니다.
    웹캠을 통해 영상을 받아 YOLO 모델과 MediaPipe를 사용하여 손 씻기 동작을 분석합니다.
    """
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.cap = None
        self.is_measuring = False

        self.HANDWASH_MODEL_PATH = r"models/best.pt" # YOLO 가중치 파일 경로
        try:
            self.handwash_model = YOLO(self.HANDWASH_MODEL_PATH) #
            print("YOLO 모델이 성공적으로 로드되었습니다.")
        except Exception as e:
            print(f"YOLO 모델 로드 오류: {e}")
            self.handwash_model = None

        self.mp_holistic = mp.solutions.holistic #
        self.holistic_detector = None

        self.CLASS_NAMES_HANDWASH = [
            "0.Palm to Palm", "1.Back of Hands", "2.Interlaced Fingers",
            "3.Backs of Fingers", "4.Thumbs", "5.Fingertips and Nails"
        ] #
        
        # --- 측정 관련 상수 정의 (기본값, SettingsScreen에서 수정 가능) ---
        self.MAX_HISTORY_LEN = 20 #
        self.MIN_FRAMES_FOR_STABLE_ACTION = 10 #
        self.MIN_ACTION_DURATION_FOR_COUNT = 1.5 #
        self.MIN_WRIST_MOVEMENT_THRESHOLD_MP = 3 #
        self.MIN_WRIST_VISIBILITY_MP = 0.5 #
        self.MAX_MEASUREMENT_DURATION_SEC = 120 #
        self.NO_MOVEMENT_THRESHOLD_SEC = 0.4 #
        self.RECOMMENDATION_TARGET_DURATION_SEC = 5.0 #
        self.RECOMMENDATION_INTERVAL_SEC = 3.0 #

        # 설정값 타입 정의 (SettingsScreen에서 사용)
        self.settings_types = {
            "MAX_HISTORY_LEN": int,
            "MIN_FRAMES_FOR_STABLE_ACTION": int,
            "MIN_ACTION_DURATION_FOR_COUNT": float,
            "MIN_WRIST_MOVEMENT_THRESHOLD_MP": int, # 원본 코드에서는 정수처럼 사용되었으나, float도 가능
            "MIN_WRIST_VISIBILITY_MP": float,
            "MAX_MEASUREMENT_DURATION_SEC": int,
            "NO_MOVEMENT_THRESHOLD_SEC": float,
            "RECOMMENDATION_TARGET_DURATION_SEC": float,
            "RECOMMENDATION_INTERVAL_SEC": float
        }
        # 설정값 한글 설명 (SettingsScreen에서 사용)
        self.settings_labels = {
            "MAX_HISTORY_LEN": "YOLO 예측 기록 길이 (프레임 수):",
            "MIN_FRAMES_FOR_STABLE_ACTION": "안정적 동작 간주 최소 프레임 수:",
            "MIN_ACTION_DURATION_FOR_COUNT": "동작 카운트 최소 지속 시간 (초):",
            "MIN_WRIST_MOVEMENT_THRESHOLD_MP": "MediaPipe 손목 움직임 감지 임계값 (픽셀):",
            "MIN_WRIST_VISIBILITY_MP": "MediaPipe 손목 랜드마크 최소 가시성 (0.0~1.0):",
            "MAX_MEASUREMENT_DURATION_SEC": "최대 손 씻기 측정 시간 (초):",
            "NO_MOVEMENT_THRESHOLD_SEC": "움직임 없음 간주 시간 (초):",
            "RECOMMENDATION_TARGET_DURATION_SEC": "각 동작별 권장 최소 시간 (초):",
            "RECOMMENDATION_INTERVAL_SEC": "권장 메시지 업데이트 간격 (초):"
        }


        self.grid_rowconfigure(0, weight=0) 
        self.grid_rowconfigure(1, weight=1) 
        self.grid_rowconfigure(2, weight=0) 
        self.grid_columnconfigure(0, weight=1)

        self.title_label = customtkinter.CTkLabel(self, text="손 씻기 진행 중...", 
                                                  font=customtkinter.CTkFont(size=24, weight="bold"))
        self.title_label.grid(row=0, column=0, pady=10)

        self.video_frame_label = None 

        self.info_frame = customtkinter.CTkFrame(self)
        self.info_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        self.info_frame.grid_columnconfigure((0,1,2,3), weight=1) 

        self.time_label = customtkinter.CTkLabel(self.info_frame, text="시간: 0s / 120s")
        self.time_label.grid(row=0, column=0, padx=5, pady=5)
        self.action_label = customtkinter.CTkLabel(self.info_frame, text="현재 동작: N/A")
        self.action_label.grid(row=0, column=1, padx=5, pady=5)
        self.recommendation_label = customtkinter.CTkLabel(self.info_frame, text="")
        self.recommendation_label.grid(row=0, column=2, padx=5, pady=5)
        self.movement_label = customtkinter.CTkLabel(self.info_frame, text="")
        self.movement_label.grid(row=0, column=3, padx=5, pady=5)

        self.stop_button = customtkinter.CTkButton(self.info_frame, text="손 씻기 중지", 
                                                   command=self.stop_measurement)
        self.stop_button.grid(row=1, column=0, columnspan=4, pady=10)

    def on_show(self):
        self.start_measurement()

    def start_measurement(self): #
        print("손 씻기 측정 시작 중...")
        if not self.handwash_model:
            if self.video_frame_label:
                self.video_frame_label.configure(image=None, text="YOLO 모델이 로드되지 않았습니다. 시작할 수 없습니다.")
            else:
                print("YOLO 모델이 로드되지 않았습니다. 측정을 시작할 수 없습니다. (video_frame_label 아직 없음)")
            return

        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0) 
            if not self.cap.isOpened():
                print("웹캠을 열 수 없습니다.")
                if self.video_frame_label:
                    self.video_frame_label.configure(image=None, text="웹캠을 찾을 수 없습니다!")
                else:
                    print("웹캠을 찾을 수 없습니다! (video_frame_label 아직 없음)")
                self.cap = None
                return

        if self.holistic_detector:
            try:
                self.holistic_detector.close()
                print("이전 MediaPipe Holistic 감지기가 닫혔습니다.")
            except Exception as e:
                print(f"참고: 이전 Holistic 감지기 닫기 오류: {e}")
        
        self.holistic_detector = self.mp_holistic.Holistic(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        print("MediaPipe Holistic 감지기가 초기화/재초기화되었습니다.")

        self.action_durations = {name: 0.0 for name in self.CLASS_NAMES_HANDWASH}
        self.action_counts = {name: 0 for name in self.CLASS_NAMES_HANDWASH}
        self.history_handwash_action = deque(maxlen=self.MAX_HISTORY_LEN) # 설정값 사용
        self.current_stable_handwash_action = None
        self.current_stable_handwash_action_start_time = 0.0
        self.prev_mp_left_wrist_pt = None
        self.prev_mp_right_wrist_pt = None
        self.measurement_start_time_total = time.time()
        self.last_frame_time = time.time()
        self.last_mp_movement_time = time.time()
        self.is_user_actively_washing = True
        self.show_wash_hands_prompt = False
        self.current_recommendation_text = ""
        self.last_recommendation_update_time = 0.0
        
        self.is_measuring = True

        if self.video_frame_label:
            self.video_frame_label.destroy()
            self.video_frame_label = None
        
        self.video_frame_label = customtkinter.CTkLabel(self, text="초기화 중...", fg_color="black")
        self.video_frame_label.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.video_frame_label.image = None 


        self.time_label.configure(text=f"시간: 0s / {self.MAX_MEASUREMENT_DURATION_SEC}s") # 설정값 사용
        self.action_label.configure(text="현재 동작: N/A")
        default_text_color = customtkinter.ThemeManager.theme["CTkLabel"]["text_color"]
        self.recommendation_label.configure(text="", text_color=default_text_color)
        self.movement_label.configure(text="")

        self.process_video_frame()

    def stop_measurement(self): #
        print("손 씻기 측정 중지 중.")
        self.is_measuring = False 

        if self.cap and self.cap.isOpened():
            self.cap.release()
            print("웹캠이 해제되었습니다.")
        self.cap = None

        if self.holistic_detector:
            self.holistic_detector.close()
            self.holistic_detector = None
            print("MediaPipe Holistic 감지기가 닫히고 None으로 설정되었습니다.")

        try:
            if hasattr(self.video_frame_label, 'image'):
                self.video_frame_label.image = None 
            if hasattr(self.video_frame_label, 'imgtk'):
                self.video_frame_label.imgtk = None
            self.video_frame_label.configure(image=None, text="측정 중지됨. 결과 보기 또는 새로 시작.")
        except tkinter.TclError as e:
            print(f"정지 시 video_frame_label 구성 오류 (TclError): {e}")
        except Exception as e:
            print(f"정지 시 video_frame_label 구성 오류 (일반 오류): {e}")
            import traceback
            traceback.print_exc()
        
        current_time_for_stop = time.time()
        if hasattr(self, 'current_stable_handwash_action') and self.current_stable_handwash_action is not None and \
           hasattr(self, 'current_stable_handwash_action_start_time') and self.current_stable_handwash_action_start_time > 0:
            stable_duration_for_count = current_time_for_stop - self.current_stable_handwash_action_start_time
            if stable_duration_for_count >= self.MIN_ACTION_DURATION_FOR_COUNT: # 설정값 사용
                if hasattr(self, 'action_counts') and self.current_stable_handwash_action in self.action_counts:
                    self.action_counts[self.current_stable_handwash_action] += 1
                    print(f"*** 손 씻기 카운트 (중지 시): {self.current_stable_handwash_action} ({stable_duration_for_count:.1f}초 동안) ***")
                else:
                    print(f"경고: action_counts 또는 특정 동작이 초기화되지 않음: {self.current_stable_handwash_action}")
        
        total_time_value = 0
        if hasattr(self, 'measurement_start_time_total'):
            total_time_value = current_time_for_stop - self.measurement_start_time_total
        
        results = {
            "total_time": total_time_value,
            "action_durations": self.action_durations.copy() if hasattr(self, 'action_durations') else {},
            "action_counts": self.action_counts.copy() if hasattr(self, 'action_counts') else {}
        }
        self.controller.set_handwash_results(results)
        self.controller.handle_action("손씻기 결과 실행 요청", data=results) # 변경: handle_action 사용

    def calculate_euclidean_distance(self, pt1, pt2): #
        if pt1 is None or pt2 is None: return float('inf')
        return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

    def determine_stable_action_class(self, current_history, class_list, min_frames_threshold): #
        if not current_history or len(current_history) < min_frames_threshold: return None
        most_common_list = Counter(current_history).most_common(1)
        if not most_common_list: return None
        action_candidate, count = most_common_list[0]
        if action_candidate is not None and count >= min_frames_threshold: 
            return action_candidate
        return None

    def update_recommendations(self): #
        if time.time() - self.last_recommendation_update_time < self.RECOMMENDATION_INTERVAL_SEC: # 설정값 사용
            return
        
        found_recommendation = False
        if not hasattr(self, 'action_durations'):
            self.current_recommendation_text = "초기화 중..."
            return

        for action_name in self.CLASS_NAMES_HANDWASH:
            if self.action_durations.get(action_name, 0.0) < self.RECOMMENDATION_TARGET_DURATION_SEC: # 설정값 사용
                self.current_recommendation_text = f"다음 동작: {action_name.split('.')[1].strip()} ({self.action_durations.get(action_name, 0.0):.1f}초)"
                found_recommendation = True
                break
        
        if not found_recommendation:
            self.current_recommendation_text = "모든 동작이 좋습니다!"
        self.last_recommendation_update_time = time.time()

    def process_video_frame(self): #
        try:
            if not self.is_measuring or not self.cap or not self.cap.isOpened() or not self.holistic_detector:
                if not self.is_measuring: 
                    print("측정 중지됨, 비디오 처리 루프 종료.")
                else: 
                    print("오류: 측정 활성화 상태이나 웹캠/holistic_detector 준비 안됨. 중지 중.")
                    if self.is_measuring: 
                        self.stop_measurement()
                return

            ret, frame = self.cap.read()
            if not ret:
                print("웹캠 프레임 읽기 오류. 측정 중지 중.")
                if self.is_measuring: self.stop_measurement()
                return

            current_time = time.time() 
            delta_time = current_time - self.last_frame_time 
            self.last_frame_time = current_time
            frame_height, frame_width, _ = frame.shape
            elapsed_time_total = current_time - self.measurement_start_time_total

            if elapsed_time_total >= self.MAX_MEASUREMENT_DURATION_SEC: # 설정값 사용
                print("측정 시간 제한에 도달했습니다.")
                if self.is_measuring: self.stop_measurement()
                return

            current_frame_handwash_prediction = None
            boxes_handwash = []

            if self.handwash_model:
                handwash_results = self.handwash_model.predict(source=frame, conf=0.5, verbose=False, device='cpu')
                if handwash_results and handwash_results[0].boxes:
                    boxes_handwash = handwash_results[0].boxes
                    if boxes_handwash: 
                        largest_box = max(boxes_handwash, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
                        class_id = int(largest_box.cls[0])
                        if 0 <= class_id < len(self.CLASS_NAMES_HANDWASH):
                            current_frame_handwash_prediction = self.CLASS_NAMES_HANDWASH[class_id]
            
            self.history_handwash_action.append(current_frame_handwash_prediction)
            new_stable_handwash_action = self.determine_stable_action_class(
                self.history_handwash_action, self.CLASS_NAMES_HANDWASH, self.MIN_FRAMES_FOR_STABLE_ACTION) # 설정값 사용

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False 
            mp_holistic_results = self.holistic_detector.process(frame_rgb)
            frame_rgb.flags.writeable = True 

            movement_this_frame_mp = False
            current_mp_left_wrist_pt, current_mp_right_wrist_pt = None, None

            if mp_holistic_results.pose_landmarks:
                pose_landmarks = mp_holistic_results.pose_landmarks.landmark
                left_wrist_landmark = pose_landmarks[self.mp_holistic.PoseLandmark.LEFT_WRIST.value]
                right_wrist_landmark = pose_landmarks[self.mp_holistic.PoseLandmark.RIGHT_WRIST.value]
                
                if left_wrist_landmark.visibility > self.MIN_WRIST_VISIBILITY_MP: # 설정값 사용
                    current_mp_left_wrist_pt = (int(left_wrist_landmark.x * frame_width), int(left_wrist_landmark.y * frame_height))
                    if self.prev_mp_left_wrist_pt:
                        dist_left = self.calculate_euclidean_distance(self.prev_mp_left_wrist_pt, current_mp_left_wrist_pt)
                        if self.MIN_WRIST_MOVEMENT_THRESHOLD_MP < dist_left < (frame_width / 2.5): # 설정값 사용
                            movement_this_frame_mp = True
                
                if not movement_this_frame_mp and right_wrist_landmark.visibility > self.MIN_WRIST_VISIBILITY_MP: # 설정값 사용
                    current_mp_right_wrist_pt = (int(right_wrist_landmark.x * frame_width), int(right_wrist_landmark.y * frame_height))
                    if self.prev_mp_right_wrist_pt:
                        dist_right = self.calculate_euclidean_distance(self.prev_mp_right_wrist_pt, current_mp_right_wrist_pt)
                        if self.MIN_WRIST_MOVEMENT_THRESHOLD_MP < dist_right < (frame_width / 2.5): # 설정값 사용
                            movement_this_frame_mp = True
            
            if movement_this_frame_mp:
                self.last_mp_movement_time = current_time
                self.is_user_actively_washing = True
                self.show_wash_hands_prompt = False
            else:
                if current_time - self.last_mp_movement_time > self.NO_MOVEMENT_THRESHOLD_SEC: # 설정값 사용
                    self.is_user_actively_washing = False
                    self.show_wash_hands_prompt = True 

            self.prev_mp_left_wrist_pt = current_mp_left_wrist_pt if current_mp_left_wrist_pt else self.prev_mp_left_wrist_pt
            self.prev_mp_right_wrist_pt = current_mp_right_wrist_pt if current_mp_right_wrist_pt else self.prev_mp_right_wrist_pt

            if new_stable_handwash_action != self.current_stable_handwash_action:
                if self.current_stable_handwash_action is not None and self.current_stable_handwash_action_start_time > 0:
                    stable_duration_for_count = current_time - self.current_stable_handwash_action_start_time
                    if stable_duration_for_count >= self.MIN_ACTION_DURATION_FOR_COUNT: # 설정값 사용
                        self.action_counts[self.current_stable_handwash_action] += 1
                        print(f"*** 손 씻기 카운트: {self.current_stable_handwash_action} ({stable_duration_for_count:.1f}초 동안) ***")
                
                self.current_stable_handwash_action = new_stable_handwash_action
                if self.current_stable_handwash_action is not None:
                    self.current_stable_handwash_action_start_time = current_time
                    print(f"[{int(elapsed_time_total)}초] 안정적 동작: {self.current_stable_handwash_action}")
                    self.show_wash_hands_prompt = False 
                else:
                    self.current_stable_handwash_action_start_time = 0
                    print(f"[{int(elapsed_time_total)}초] 안정적 동작: 없음 / 불안정")
                    if not self.is_user_actively_washing: 
                        self.show_wash_hands_prompt = True

            if self.current_stable_handwash_action is not None and self.is_user_actively_washing:
                self.action_durations[self.current_stable_handwash_action] += delta_time

            display_frame_bgr = frame.copy() 
            if self.handwash_model and boxes_handwash and handwash_results:
                 display_frame_bgr = handwash_results[0].plot() 

            label_width = self.video_frame_label.winfo_width()
            label_height = self.video_frame_label.winfo_height()
            if label_width < 10 or label_height < 10: 
                label_width, label_height = 640, 480 

            aspect_ratio = frame_width / frame_height 
            target_width = label_width
            target_height = int(target_width / aspect_ratio)
            
            if target_height > label_height:
                target_height = label_height
                target_width = int(target_height * aspect_ratio)
            
            if target_width <= 0: target_width = 1
            if target_height <= 0: target_height = 1

            img_resized_bgr = cv2.resize(display_frame_bgr, (target_width, target_height), interpolation=cv2.INTER_AREA)
            img_rgb_for_pil = cv2.cvtColor(img_resized_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb_for_pil)
            img_tk = customtkinter.CTkImage(light_image=img_pil, dark_image=img_pil, size=(target_width, target_height))
            
            self.video_frame_label.configure(image=img_tk, text="")
            self.video_frame_label.image = img_tk 

            self.time_label.configure(text=f"시간: {int(elapsed_time_total)}초 / {self.MAX_MEASUREMENT_DURATION_SEC}초") # 설정값 사용
            current_action_text = self.current_stable_handwash_action.split('.')[1].strip() if self.current_stable_handwash_action else 'N/A'
            self.action_label.configure(text=f"현재 동작: {current_action_text}")
            
            default_text_color = customtkinter.ThemeManager.theme["CTkLabel"]["text_color"]
            if self.show_wash_hands_prompt:
                self.recommendation_label.configure(text="!! 손을 씻어주세요 !!", text_color="red")
            else:
                self.update_recommendations() 
                self.recommendation_label.configure(text=self.current_recommendation_text, 
                                                     text_color="orange" if "다음 동작:" in self.current_recommendation_text else default_text_color)

            self.movement_label.configure(text="움직임 감지" if movement_this_frame_mp else "정지 상태",
                                          text_color="green" if movement_this_frame_mp else "red")

            if self.is_measuring:
                self.after(15, self.process_video_frame) 
        
        except tkinter.TclError as e: 
            print(f"process_video_frame에서 TclError 발생 (위젯 파괴 가능성): {e}")
            if self.is_measuring: self.stop_measurement() 
        except Exception as e:
            print(f"process_video_frame에서 일반 오류 발생: {e}")
            import traceback
            traceback.print_exc() 
            if self.is_measuring: self.stop_measurement()


class StatusScreen(customtkinter.CTkFrame):
    """
    이전 손 씻기 세션의 결과를 표시하는 화면입니다.
    총 시간, 각 동작의 지속 시간 및 카운트, 전반적인 손 씻기 품질 점수를 보여줍니다.
    """
    def __init__(self, parent, controller): #
        super().__init__(parent)
        self.controller = controller

        self.grid_rowconfigure(0, weight=0) 
        self.grid_rowconfigure(1, weight=1) 
        self.grid_rowconfigure(2, weight=0) 
        self.grid_columnconfigure(0, weight=1) 

        self.label = customtkinter.CTkLabel(self, text="손 씻기 상태 및 결과", 
                                            font=customtkinter.CTkFont(size=24, weight="bold"))
        self.label.grid(row=0, column=0, pady=20)

        self.results_text = customtkinter.CTkTextbox(self, wrap="word", width=600, height=300, 
                                                     corner_radius=10, font=customtkinter.CTkFont(size=14))
        self.results_text.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.results_text.configure(state="disabled") 

        self.back_button = customtkinter.CTkButton(self, text="메뉴로 돌아가기", 
                                                   command=lambda: controller.handle_action("메뉴 화면으로 이동 요청"), # 변경: handle_action 사용
                                                   font=customtkinter.CTkFont(size=18), height=40)
        self.back_button.grid(row=2, column=0, pady=20)

        # self.update_status(None) # on_show에서 호출하도록 변경

    def on_show(self, data=None): # 데이터 전달을 위해 on_show로 변경
        self.update_status(data)

    def update_status(self, data): #
        self.results_text.configure(state="normal") 
        self.results_text.delete("1.0", "end") 
        
        if not data or "action_durations" not in data or "action_counts" not in data :
            self.results_text.insert("end", "아직 손 씻기 데이터가 없습니다. 세션을 완료해주세요.")
            self.results_text.configure(state="disabled") 
            return

        total_time = data.get("total_time", 0)
        action_durations = data.get("action_durations", {})
        action_counts = data.get("action_counts", {})
        
        try:
            exec_screen = self.controller.frames.get("ExecutionScreen")
            if not exec_screen:
                raise KeyError("ExecutionScreen을 controller.frames에서 찾을 수 없습니다.")
            class_names_handwash = exec_screen.CLASS_NAMES_HANDWASH
            rec_target_duration = exec_screen.RECOMMENDATION_TARGET_DURATION_SEC # ExecutionScreen의 현재 설정값 참조
        except KeyError as e:
            print(f"ExecutionScreen 속성 접근 오류: {e}")
            self.results_text.insert("end", "오류: 표시를 위한 클래스 이름 또는 목표 지속 시간을 검색할 수 없습니다.\n")
            self.results_text.configure(state="disabled")
            return

        self.results_text.insert("end", f"--- 손 씻기 세션 요약 ---\n\n")
        self.results_text.insert("end", f"총 손 씻기 시간: {total_time:.2f} 초\n\n")
        
        self.results_text.insert("end", "[각 동작별 지속 시간 (움직임 감지 시)]\n")
        if action_durations and class_names_handwash:
            for name in class_names_handwash:
                duration = action_durations.get(name, 0.0)
                self.results_text.insert("end", f"- {name.split('.')[1].strip()}: {duration:.2f} 초\n")
        else:
            self.results_text.insert("end", "   특정 동작 지속 시간이 기록되지 않았습니다.\n")
        
        self.results_text.insert("end", "\n")
        self.results_text.insert("end", "[각 동작별 카운트 (1.5초 이상 안정적 유지 시)]\n") # MIN_ACTION_DURATION_FOR_COUNT 참조 필요 시 수정
        if action_counts and class_names_handwash:
            for name in class_names_handwash:
                count = action_counts.get(name, 0)
                self.results_text.insert("end", f"- {name.split('.')[1].strip()}: {count} 회\n")
        else:
            self.results_text.insert("end", "   특정 동작 카운트가 기록되지 않았습니다.\n")
        
        self.results_text.insert("end", "\n")
        
        overall_score = 0.0
        num_actions = len(class_names_handwash)

        if num_actions > 0 and rec_target_duration > 0 :
            achieved_score_sum = 0
            for action_name in class_names_handwash:
                duration = action_durations.get(action_name, 0.0)
                action_score = min(duration / rec_target_duration, 1.0)
                achieved_score_sum += action_score
            
            if num_actions > 0: 
                overall_score = (achieved_score_sum / num_actions) * 100
            else:
                overall_score = 0.0
        else: 
            overall_score = 0.0

        overall_score = max(0.0, min(overall_score, 100.0)) 

        if overall_score >= 80:
            quality = "매우 우수합니다!"
        elif overall_score >= 50:
            quality = "양호합니다!"
        else:
            quality = "더 연습이 필요합니다."
            
        self.results_text.insert("end", f"전반적인 손 씻기 품질: {quality} (점수: {overall_score:.1f}%)\n\n")
        self.results_text.insert("end", "손의 모든 부분을 깨끗하게 씻는 것을 잊지 마세요!")
        self.results_text.configure(state="disabled")


class SettingsScreen(customtkinter.CTkFrame):
    """
    애플리케이션의 측정 관련 설정을 변경하는 화면입니다.
    """
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.settings_vars = {} # CTkEntry와 연결될 StringVar들
        self.entry_widgets = {} # CTkEntry 위젯 저장

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=0) # 타이틀
        # 나머지 행들은 동적으로 생성되므로 weight는 content에 따라 조절

        title_label = customtkinter.CTkLabel(self, text="측정 설정 변경", font=customtkinter.CTkFont(size=24, weight="bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=20, sticky="n")

        # 스크롤 가능한 프레임 추가 (설정 항목이 많을 경우 대비)
        self.scrollable_frame = customtkinter.CTkScrollableFrame(self, label_text="설정 항목")
        self.scrollable_frame.grid(row=1, column=0, columnspan=2, padx=20, pady=(0,10), sticky="nsew")
        self.grid_rowconfigure(1, weight=1) # 스크롤 프레임이 확장되도록

        self.scrollable_frame.grid_columnconfigure(0, weight=1) # Label
        self.scrollable_frame.grid_columnconfigure(1, weight=1) # Entry

        # ExecutionScreen에서 설정 키와 레이블 가져오기
        # App 클래스가 완전히 초기화 된 후 controller를 통해 접근해야 함
        # 여기서는 일단 플레이스홀더로 두고, on_show에서 실제 생성
        
        # 버튼 프레임
        button_frame = customtkinter.CTkFrame(self)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky="s")
        button_frame.grid_columnconfigure((0,1), weight=1)


        save_button = customtkinter.CTkButton(button_frame, text="저장", command=self.save_settings)
        save_button.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        back_button = customtkinter.CTkButton(button_frame, text="메뉴로 돌아가기",
                                              command=lambda: controller.handle_action("메뉴 화면으로 이동 요청"))
        back_button.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        
        self.status_label = customtkinter.CTkLabel(self, text="", font=customtkinter.CTkFont(size=14))
        self.status_label.grid(row=3, column=0, columnspan=2, pady=(0,10), sticky="s")


    def on_show(self):
        """ 설정 화면이 표시될 때 현재 설정을 불러와 UI에 반영합니다. """
        current_settings = self.controller.get_execution_settings()
        exec_screen = self.controller.frames.get("ExecutionScreen") # 레이블, 타입 정보 가져오기 위함
        
        if not exec_screen:
            self.status_label.configure(text="오류: ExecutionScreen을 찾을 수 없습니다.", text_color="red")
            return

        # 기존 위젯들 제거 (on_show가 여러 번 호출될 경우 중복 생성 방지)
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.settings_vars.clear()
        self.entry_widgets.clear()

        if not current_settings:
            self.status_label.configure(text="설정값을 불러올 수 없습니다.", text_color="red")
            return

        self.status_label.configure(text="") # 이전 상태 메시지 초기화

        row_idx = 0
        for key, value in current_settings.items():
            label_text = exec_screen.settings_labels.get(key, key) # 한글 레이블 사용
            
            label = customtkinter.CTkLabel(self.scrollable_frame, text=label_text, anchor="w")
            label.grid(row=row_idx, column=0, padx=10, pady=5, sticky="w")

            var = customtkinter.StringVar(value=str(value))
            self.settings_vars[key] = var
            
            entry = customtkinter.CTkEntry(self.scrollable_frame, textvariable=var, width=150)
            entry.grid(row=row_idx, column=1, padx=10, pady=5, sticky="e")
            self.entry_widgets[key] = entry
            row_idx += 1
        
        print("SettingsScreen: 설정 로드 완료")

    def save_settings(self):
        """ 입력된 설정 값들을 ExecutionScreen에 저장합니다. """
        new_settings = {}
        has_errors = False
        exec_screen = self.controller.frames.get("ExecutionScreen")

        if not exec_screen:
            self.status_label.configure(text="오류: ExecutionScreen을 찾을 수 없습니다.", text_color="red")
            return

        for key, var in self.settings_vars.items():
            value_str = var.get()
            target_type = exec_screen.settings_types.get(key, str) # 기본 타입은 str
            try:
                if target_type == float:
                    new_settings[key] = float(value_str)
                elif target_type == int:
                    new_settings[key] = int(value_str)
                else:
                    new_settings[key] = value_str # 문자열 또는 기타 타입
                
                # 입력 필드 배경색 초기화 (오류 없으면)
                if key in self.entry_widgets:
                    self.entry_widgets[key].configure(border_color=customtkinter.ThemeManager.theme["CTkEntry"]["border_color"])

            except ValueError:
                print(f"오류: '{key}'에 대한 값이 잘못되었습니다. ({value_str}) 숫자를 입력해야 합니다.")
                self.status_label.configure(text=f"'{exec_screen.settings_labels.get(key,key)}'에 유효한 숫자를 입력하세요.", text_color="red")
                # 오류 발생한 입력 필드 강조 (예: 빨간색 테두리)
                if key in self.entry_widgets:
                     self.entry_widgets[key].configure(border_color="red") # 기본 테마 색상에 따라 조정 필요
                has_errors = True
                # return # 첫번째 오류에서 중단하거나, 모든 오류를 표시하려면 주석 처리
        
        if not has_errors:
            self.controller.update_execution_settings(new_settings)
            print("SettingsScreen: 설정 저장 완료")
            self.status_label.configure(text="설정이 성공적으로 저장되었습니다!", text_color="green")
            # 저장 후 일정 시간 뒤 메시지 초기화
            self.after(3000, lambda: self.status_label.configure(text=""))
        else:
            print("SettingsScreen: 유효하지 않은 입력값으로 인해 설정 저장 실패")