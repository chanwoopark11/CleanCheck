# ui/execution_screen.py
import customtkinter
import cv2 #
from ultralytics import YOLO #
from collections import deque, Counter #
import time #
import numpy as np #
import mediapipe as mp #
import tkinter # _tkinter.TclError를 명시적으로 다루기 위해 추가 (필수는 아님)

try:
    from PIL import Image, ImageTk # ImageTk도 필요합니다.
except ImportError:
    print("Pillow 라이브러리가 없습니다. 'pip install Pillow'로 설치해주세요.") #
    exit() #

class ExecutionScreen(customtkinter.CTkFrame): #
    """
    손 씻기 과정을 실시간으로 모니터링하고 피드백을 제공하는 화면입니다.
    웹캠을 통해 영상을 받아 YOLO 모델과 MediaPipe를 사용하여 손 씻기 동작을 분석합니다.
    """
    def __init__(self, parent, controller): #
        super().__init__(parent) #
        self.controller = controller #

        self.cap = None #
        self.is_measuring = False #

        self.HANDWASH_MODEL_PATH = r"models/best.pt" # YOLO 가중치 파일 경로
        try:
            self.handwash_model = YOLO(self.HANDWASH_MODEL_PATH) #
            print("YOLO 모델이 성공적으로 로드되었습니다.") #
        except Exception as e:
            print(f"YOLO 모델 로드 오류: {e}") #
            self.handwash_model = None #

        self.mp_holistic = mp.solutions.holistic #
        self.holistic_detector = None #

        self.CLASS_NAMES_HANDWASH = [ #
            "0.Palm to Palm", "1.Back of Hands", "2.Interlaced Fingers", #
            "3.Backs of Fingers", "4.Thumbs", "5.Fingertips and Nails" #
        ]
        
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
        self.settings_types = { #
            "MAX_HISTORY_LEN": int, #
            "MIN_FRAMES_FOR_STABLE_ACTION": int, #
            "MIN_ACTION_DURATION_FOR_COUNT": float, #
            "MIN_WRIST_MOVEMENT_THRESHOLD_MP": int, # 원본 코드에서는 정수처럼 사용되었으나, float도 가능
            "MIN_WRIST_VISIBILITY_MP": float, #
            "MAX_MEASUREMENT_DURATION_SEC": int, #
            "NO_MOVEMENT_THRESHOLD_SEC": float, #
            "RECOMMENDATION_TARGET_DURATION_SEC": float, #
            "RECOMMENDATION_INTERVAL_SEC": float #
        }
        # 설정값 한글 설명 (SettingsScreen에서 사용)
        self.settings_labels = { #
            "MAX_HISTORY_LEN": "YOLO 예측 기록 길이 (프레임 수):", #
            "MIN_FRAMES_FOR_STABLE_ACTION": "안정적 동작 간주 최소 프레임 수:", #
            "MIN_ACTION_DURATION_FOR_COUNT": "동작 카운트 최소 지속 시간 (초):", #
            "MIN_WRIST_MOVEMENT_THRESHOLD_MP": "MediaPipe 손목 움직임 감지 임계값 (픽셀):", #
            "MIN_WRIST_VISIBILITY_MP": "MediaPipe 손목 랜드마크 최소 가시성 (0.0~1.0):", #
            "MAX_MEASUREMENT_DURATION_SEC": "최대 손 씻기 측정 시간 (초):", #
            "NO_MOVEMENT_THRESHOLD_SEC": "움직임 없음 간주 시간 (초):", #
            "RECOMMENDATION_TARGET_DURATION_SEC": "각 동작별 권장 최소 시간 (초):", #
            "RECOMMENDATION_INTERVAL_SEC": "권장 메시지 업데이트 간격 (초):" #
        }


        self.grid_rowconfigure(0, weight=0) #
        self.grid_rowconfigure(1, weight=1) #
        self.grid_rowconfigure(2, weight=0) #
        self.grid_columnconfigure(0, weight=1) #

        self.title_label = customtkinter.CTkLabel(self, text="손 씻기 진행 중...", #
                                                  font=customtkinter.CTkFont(size=24, weight="bold")) #
        self.title_label.grid(row=0, column=0, pady=10) #

        self.video_frame_label = None #

        self.info_frame = customtkinter.CTkFrame(self) #
        self.info_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew") #
        self.info_frame.grid_columnconfigure((0,1,2,3), weight=1) #

        self.time_label = customtkinter.CTkLabel(self.info_frame, text="시간: 0s / 120s") #
        self.time_label.grid(row=0, column=0, padx=5, pady=5) #
        self.action_label = customtkinter.CTkLabel(self.info_frame, text="현재 동작: N/A") #
        self.action_label.grid(row=0, column=1, padx=5, pady=5) #
        self.recommendation_label = customtkinter.CTkLabel(self.info_frame, text="") #
        self.recommendation_label.grid(row=0, column=2, padx=5, pady=5) #
        self.movement_label = customtkinter.CTkLabel(self.info_frame, text="") #
        self.movement_label.grid(row=0, column=3, padx=5, pady=5) #

        self.stop_button = customtkinter.CTkButton(self.info_frame, text="손 씻기 중지", #
                                                   command=self.stop_measurement) #
        self.stop_button.grid(row=1, column=0, columnspan=4, pady=10) #

    def on_show(self): #
        self.start_measurement() #

    def start_measurement(self): #
        print("손 씻기 측정 시작 중...") #
        if not self.handwash_model: #
            if self.video_frame_label: #
                self.video_frame_label.configure(image=None, text="YOLO 모델이 로드되지 않았습니다. 시작할 수 없습니다.") #
            else:
                print("YOLO 모델이 로드되지 않았습니다. 측정을 시작할 수 없습니다. (video_frame_label 아직 없음)") #
            return

        if self.cap is None or not self.cap.isOpened(): #
            self.cap = cv2.VideoCapture(0) #
            if not self.cap.isOpened(): #
                print("웹캠을 열 수 없습니다.") #
                if self.video_frame_label: #
                    self.video_frame_label.configure(image=None, text="웹캠을 찾을 수 없습니다!") #
                else:
                    print("웹캠을 찾을 수 없습니다! (video_frame_label 아직 없음)") #
                self.cap = None #
                return

        if self.holistic_detector: #
            try:
                self.holistic_detector.close() #
                print("이전 MediaPipe Holistic 감지기가 닫혔습니다.") #
            except Exception as e:
                print(f"참고: 이전 Holistic 감지기 닫기 오류: {e}") #
        
        self.holistic_detector = self.mp_holistic.Holistic( #
            min_detection_confidence=0.5, #
            min_tracking_confidence=0.5 #
        )
        print("MediaPipe Holistic 감지기가 초기화/재초기화되었습니다.") #

        self.action_durations = {name: 0.0 for name in self.CLASS_NAMES_HANDWASH} #
        self.action_counts = {name: 0 for name in self.CLASS_NAMES_HANDWASH} #
        self.history_handwash_action = deque(maxlen=self.MAX_HISTORY_LEN) # 설정값 사용
        self.current_stable_handwash_action = None #
        self.current_stable_handwash_action_start_time = 0.0 #
        self.prev_mp_left_wrist_pt = None #
        self.prev_mp_right_wrist_pt = None #
        self.measurement_start_time_total = time.time() #
        self.last_frame_time = time.time() #
        self.last_mp_movement_time = time.time() #
        self.is_user_actively_washing = True #
        self.show_wash_hands_prompt = False #
        self.current_recommendation_text = "" #
        self.last_recommendation_update_time = 0.0 #
        
        self.is_measuring = True #

        if self.video_frame_label: #
            self.video_frame_label.destroy() #
            self.video_frame_label = None #
        
        self.video_frame_label = customtkinter.CTkLabel(self, text="초기화 중...", fg_color="black") #
        self.video_frame_label.grid(row=1, column=0, padx=20, pady=10, sticky="nsew") #
        self.video_frame_label.image = None #


        self.time_label.configure(text=f"시간: 0s / {self.MAX_MEASUREMENT_DURATION_SEC}s") # 설정값 사용
        self.action_label.configure(text="현재 동작: N/A") #
        default_text_color = customtkinter.ThemeManager.theme["CTkLabel"]["text_color"] #
        self.recommendation_label.configure(text="", text_color=default_text_color) #
        self.movement_label.configure(text="") #

        self.process_video_frame() #

    def stop_measurement(self): #
        print("손 씻기 측정 중지 중.") #
        self.is_measuring = False #

        if self.cap and self.cap.isOpened(): #
            self.cap.release() #
            print("웹캠이 해제되었습니다.") #
        self.cap = None #

        if self.holistic_detector: #
            self.holistic_detector.close() #
            self.holistic_detector = None #
            print("MediaPipe Holistic 감지기가 닫히고 None으로 설정되었습니다.") #

        try:
            if hasattr(self.video_frame_label, 'image'): #
                self.video_frame_label.image = None #
            if hasattr(self.video_frame_label, 'imgtk'): #
                self.video_frame_label.imgtk = None #
            self.video_frame_label.configure(image=None, text="측정 중지됨. 결과 보기 또는 새로 시작.") #
        except tkinter.TclError as e:
            print(f"정지 시 video_frame_label 구성 오류 (TclError): {e}") #
        except Exception as e:
            print(f"정지 시 video_frame_label 구성 오류 (일반 오류): {e}") #
            import traceback #
            traceback.print_exc() #
        
        current_time_for_stop = time.time() #
        if hasattr(self, 'current_stable_handwash_action') and self.current_stable_handwash_action is not None and \
           hasattr(self, 'current_stable_handwash_action_start_time') and self.current_stable_handwash_action_start_time > 0: #
            stable_duration_for_count = current_time_for_stop - self.current_stable_handwash_action_start_time #
            if stable_duration_for_count >= self.MIN_ACTION_DURATION_FOR_COUNT: # 설정값 사용
                if hasattr(self, 'action_counts') and self.current_stable_handwash_action in self.action_counts: #
                    self.action_counts[self.current_stable_handwash_action] += 1 #
                    print(f"*** 손 씻기 카운트 (중지 시): {self.current_stable_handwash_action} ({stable_duration_for_count:.1f}초 동안) ***") #
                else:
                    print(f"경고: action_counts 또는 특정 동작이 초기화되지 않음: {self.current_stable_handwash_action}") #
        
        total_time_value = 0 #
        if hasattr(self, 'measurement_start_time_total'): #
            total_time_value = current_time_for_stop - self.measurement_start_time_total #
        
        results = { #
            "total_time": total_time_value, #
            "action_durations": self.action_durations.copy() if hasattr(self, 'action_durations') else {}, #
            "action_counts": self.action_counts.copy() if hasattr(self, 'action_counts') else {} #
        }
        self.controller.set_handwash_results(results) #
        self.controller.handle_action("손씻기 결과 실행 요청", data=results) # 변경: handle_action 사용

    def calculate_euclidean_distance(self, pt1, pt2): #
        if pt1 is None or pt2 is None: return float('inf') #
        return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2) #

    def determine_stable_action_class(self, current_history, class_list, min_frames_threshold): #
        if not current_history or len(current_history) < min_frames_threshold: return None #
        most_common_list = Counter(current_history).most_common(1) #
        if not most_common_list: return None #
        action_candidate, count = most_common_list[0] #
        if action_candidate is not None and count >= min_frames_threshold: #
            return action_candidate #
        return None #

    def update_recommendations(self): #
        if time.time() - self.last_recommendation_update_time < self.RECOMMENDATION_INTERVAL_SEC: # 설정값 사용
            return
        
        found_recommendation = False #
        if not hasattr(self, 'action_durations'): #
            self.current_recommendation_text = "초기화 중..." #
            return

        for action_name in self.CLASS_NAMES_HANDWASH: #
            if self.action_durations.get(action_name, 0.0) < self.RECOMMENDATION_TARGET_DURATION_SEC: # 설정값 사용
                self.current_recommendation_text = f"다음 동작: {action_name.split('.')[1].strip()} ({self.action_durations.get(action_name, 0.0):.1f}초)" #
                found_recommendation = True #
                break
        
        if not found_recommendation: #
            self.current_recommendation_text = "모든 동작이 좋습니다!" #
        self.last_recommendation_update_time = time.time() #

    def process_video_frame(self): #
        try:
            if not self.is_measuring or not self.cap or not self.cap.isOpened() or not self.holistic_detector: #
                if not self.is_measuring: #
                    print("측정 중지됨, 비디오 처리 루프 종료.") #
                else:
                    print("오류: 측정 활성화 상태이나 웹캠/holistic_detector 준비 안됨. 중지 중.") #
                    if self.is_measuring: #
                        self.stop_measurement() #
                return

            ret, frame = self.cap.read() #
            if not ret: #
                print("웹캠 프레임 읽기 오류. 측정 중지 중.") #
                if self.is_measuring: self.stop_measurement() #
                return

            current_time = time.time() #
            delta_time = current_time - self.last_frame_time #
            self.last_frame_time = current_time #
            frame_height, frame_width, _ = frame.shape #
            elapsed_time_total = current_time - self.measurement_start_time_total #

            if elapsed_time_total >= self.MAX_MEASUREMENT_DURATION_SEC: # 설정값 사용
                print("측정 시간 제한에 도달했습니다.") #
                if self.is_measuring: self.stop_measurement() #
                return

            current_frame_handwash_prediction = None #
            boxes_handwash = [] #

            if self.handwash_model: #
                handwash_results = self.handwash_model.predict(source=frame, conf=0.5, verbose=False, device='cpu') #
                if handwash_results and handwash_results[0].boxes: #
                    boxes_handwash = handwash_results[0].boxes #
                    if boxes_handwash: #
                        largest_box = max(boxes_handwash, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1])) #
                        class_id = int(largest_box.cls[0]) #
                        if 0 <= class_id < len(self.CLASS_NAMES_HANDWASH): #
                            current_frame_handwash_prediction = self.CLASS_NAMES_HANDWASH[class_id] #
            
            self.history_handwash_action.append(current_frame_handwash_prediction) #
            new_stable_handwash_action = self.determine_stable_action_class( #
                self.history_handwash_action, self.CLASS_NAMES_HANDWASH, self.MIN_FRAMES_FOR_STABLE_ACTION) # 설정값 사용 #

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #
            frame_rgb.flags.writeable = False #
            mp_holistic_results = self.holistic_detector.process(frame_rgb) #
            frame_rgb.flags.writeable = True #

            movement_this_frame_mp = False #
            current_mp_left_wrist_pt, current_mp_right_wrist_pt = None, None #

            if mp_holistic_results.pose_landmarks: #
                pose_landmarks = mp_holistic_results.pose_landmarks.landmark #
                left_wrist_landmark = pose_landmarks[self.mp_holistic.PoseLandmark.LEFT_WRIST.value] #
                right_wrist_landmark = pose_landmarks[self.mp_holistic.PoseLandmark.RIGHT_WRIST.value] #
                
                if left_wrist_landmark.visibility > self.MIN_WRIST_VISIBILITY_MP: # 설정값 사용
                    current_mp_left_wrist_pt = (int(left_wrist_landmark.x * frame_width), int(left_wrist_landmark.y * frame_height)) #
                    if self.prev_mp_left_wrist_pt: #
                        dist_left = self.calculate_euclidean_distance(self.prev_mp_left_wrist_pt, current_mp_left_wrist_pt) #
                        if self.MIN_WRIST_MOVEMENT_THRESHOLD_MP < dist_left < (frame_width / 2.5): # 설정값 사용
                            movement_this_frame_mp = True #
                
                if not movement_this_frame_mp and right_wrist_landmark.visibility > self.MIN_WRIST_VISIBILITY_MP: # 설정값 사용
                    current_mp_right_wrist_pt = (int(right_wrist_landmark.x * frame_width), int(right_wrist_landmark.y * frame_height)) #
                    if self.prev_mp_right_wrist_pt: #
                        dist_right = self.calculate_euclidean_distance(self.prev_mp_right_wrist_pt, current_mp_right_wrist_pt) #
                        if self.MIN_WRIST_MOVEMENT_THRESHOLD_MP < dist_right < (frame_width / 2.5): # 설정값 사용
                            movement_this_frame_mp = True #
            
            if movement_this_frame_mp: #
                self.last_mp_movement_time = current_time #
                self.is_user_actively_washing = True #
                self.show_wash_hands_prompt = False #
            else:
                if current_time - self.last_mp_movement_time > self.NO_MOVEMENT_THRESHOLD_SEC: # 설정값 사용
                    self.is_user_actively_washing = False #
                    self.show_wash_hands_prompt = True #

            self.prev_mp_left_wrist_pt = current_mp_left_wrist_pt if current_mp_left_wrist_pt else self.prev_mp_left_wrist_pt #
            self.prev_mp_right_wrist_pt = current_mp_right_wrist_pt if current_mp_right_wrist_pt else self.prev_mp_right_wrist_pt #

            if new_stable_handwash_action != self.current_stable_handwash_action: #
                if self.current_stable_handwash_action is not None and self.current_stable_handwash_action_start_time > 0: #
                    stable_duration_for_count = current_time - self.current_stable_handwash_action_start_time #
                    if stable_duration_for_count >= self.MIN_ACTION_DURATION_FOR_COUNT: # 설정값 사용
                        self.action_counts[self.current_stable_handwash_action] += 1 #
                        print(f"*** 손 씻기 카운트: {self.current_stable_handwash_action} ({stable_duration_for_count:.1f}초 동안) ***") #
                
                self.current_stable_handwash_action = new_stable_handwash_action #
                if self.current_stable_handwash_action is not None: #
                    self.current_stable_handwash_action_start_time = current_time #
                    print(f"[{int(elapsed_time_total)}초] 안정적 동작: {self.current_stable_handwash_action}") #
                    self.show_wash_hands_prompt = False #
                else:
                    self.current_stable_handwash_action_start_time = 0 #
                    print(f"[{int(elapsed_time_total)}초] 안정적 동작: 없음 / 불안정") #
                    if not self.is_user_actively_washing: #
                        self.show_wash_hands_prompt = True #

            if self.current_stable_handwash_action is not None and self.is_user_actively_washing: #
                self.action_durations[self.current_stable_handwash_action] += delta_time #

            display_frame_bgr = frame.copy() #
            if self.handwash_model and boxes_handwash and handwash_results: #
                 display_frame_bgr = handwash_results[0].plot() #

            label_width = self.video_frame_label.winfo_width() #
            label_height = self.video_frame_label.winfo_height() #
            if label_width < 10 or label_height < 10: #
                label_width, label_height = 640, 480 #

            aspect_ratio = frame_width / frame_height #
            target_width = label_width #
            target_height = int(target_width / aspect_ratio) #
            
            if target_height > label_height: #
                target_height = label_height #
                target_width = int(target_height * aspect_ratio) #
            
            if target_width <= 0: target_width = 1 #
            if target_height <= 0: target_height = 1 #

            img_resized_bgr = cv2.resize(display_frame_bgr, (target_width, target_height), interpolation=cv2.INTER_AREA) #
            img_rgb_for_pil = cv2.cvtColor(img_resized_bgr, cv2.COLOR_BGR2RGB) #
            img_pil = Image.fromarray(img_rgb_for_pil) #
            img_tk = customtkinter.CTkImage(light_image=img_pil, dark_image=img_pil, size=(target_width, target_height)) #
            
            self.video_frame_label.configure(image=img_tk, text="") #
            self.video_frame_label.image = img_tk #

            self.time_label.configure(text=f"시간: {int(elapsed_time_total)}초 / {self.MAX_MEASUREMENT_DURATION_SEC}초") # 설정값 사용
            current_action_text = self.current_stable_handwash_action.split('.')[1].strip() if self.current_stable_handwash_action else 'N/A' #
            self.action_label.configure(text=f"현재 동작: {current_action_text}") #
            
            default_text_color = customtkinter.ThemeManager.theme["CTkLabel"]["text_color"] #
            if self.show_wash_hands_prompt: #
                self.recommendation_label.configure(text="!! 손을 씻어주세요 !!", text_color="red") #
            else:
                self.update_recommendations() #
                self.recommendation_label.configure(text=self.current_recommendation_text, #
                                                     text_color="orange" if "다음 동작:" in self.current_recommendation_text else default_text_color) #

            self.movement_label.configure(text="움직임 감지" if movement_this_frame_mp else "정지 상태", #
                                          text_color="green" if movement_this_frame_mp else "red") #

            if self.is_measuring: #
                self.after(15, self.process_video_frame) #
        
        except tkinter.TclError as e: #
            print(f"process_video_frame에서 TclError 발생 (위젯 파괴 가능성): {e}") #
            if self.is_measuring: self.stop_measurement() #
        except Exception as e:
            print(f"process_video_frame에서 일반 오류 발생: {e}") #
            import traceback #
            traceback.print_exc() #
            if self.is_measuring: self.stop_measurement() #