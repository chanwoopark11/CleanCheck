# core/handwash_mp.py
import numpy as np
import cv2
import mediapipe as mp

from core.handwash_base import HandwashDetector

class HandwashMediaPipe(HandwashDetector):
    """
    MediaPipe Holistic(또는 Pose)를 사용하여 손 씻기 동작 및 움직임을 감지하는 클래스입니다.
    HandwashDetector 추상 베이스 클래스를 구현합니다.
    """
    def __init__(self, class_names: list):
        """
        HandwashMediaPipe 클래스의 생성자입니다.
        
        Args:
            class_names (list): 감지할 손 씻기 동작 클래스 이름 목록 (여기서는 MediaPipe가 직접 분류하진 않지만,
                                 HandwashDetector의 인터페이스를 맞추기 위해 받습니다).
        """
        super().__init__(class_names)
        self.mp_holistic = mp.solutions.holistic
        self.holistic_detector = None
        self.pose_drawing = mp.solutions.drawing_utils # 랜드마크 그리기 유틸리티

        # MediaPipe 관련 상수 (ExecutionScreen에서 가져옴)
        self.MIN_WRIST_MOVEMENT_THRESHOLD_MP = 10 # MediaPipe 손목 움직임 감지 최소 임계값 (픽셀)
        self.MIN_WRIST_VISIBILITY_MP = 0.5 # MediaPipe 손목 랜드마크 최소 가시성
        
        self.prev_left_wrist_pt = None # 이전 프레임의 왼쪽 손목 좌표
        self.prev_right_wrist_pt = None # 이전 프레임의 오른쪽 손목 좌표

        # 모델 로드 (MediaPipe는 모델 파일을 로드하는 개념이 아니므로, detector 인스턴스를 생성)
        self.load_model(None) # model_path는 사용하지 않음

    def load_model(self, model_path: str = None):
        """
        MediaPipe Holistic detector 인스턴스를 초기화합니다.
        model_path는 MediaPipe에서 사용되지 않으므로 None으로 처리합니다.
        """
        try:
            self.holistic_detector = self.mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("MediaPipe Holistic detector가 성공적으로 초기화되었습니다.")
        except Exception as e:
            print(f"MediaPipe Holistic detector 초기화 오류: {e}")
            self.holistic_detector = None


    def process_frame(self, frame: np.ndarray) -> dict:
        """
        단일 비디오 프레임을 처리하여 손목 움직임을 감지하고 랜드마크를 그립니다.
        
        Args:
            frame (np.ndarray): 처리할 비디오 프레임 (OpenCV 이미지).
            
        Returns:
            dict: 감지 결과를 담은 딕셔너리.
                  'prediction': 'Handwashing' 또는 'No Movement' (str)
                  'annotated_frame': 랜드마크가 그려진 프레임 (np.ndarray)
                  'landmarks': 감지된 랜드마크 데이터 (리스트)
                  'is_moving': 손목 움직임 감지 여부 (bool)
        """
        if self.holistic_detector is None:
            print("MediaPipe detector가 초기화되지 않았습니다.")
            return {'prediction': 'No Movement', 'annotated_frame': frame, 'landmarks': [], 'is_moving': False}

        # 프레임을 RGB로 변환 (MediaPipe는 RGB 입력을 선호)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 쓰기 불가능하게 설정하여 성능 최적화
        frame_rgb.flags.writeable = False
        
        results = self.holistic_detector.process(frame_rgb)
        
        # 쓰기 가능하게 다시 설정
        frame_rgb.flags.writeable = True
        annotated_frame = frame.copy() # 원본 BGR 프레임에 그리기

        prediction = 'No Movement'
        landmarks = []
        is_moving = False

        frame_height, frame_width, _ = frame.shape

        # 포즈 랜드마크 그리기 및 손목 움직임 감지
        if results.pose_landmarks:
            self.pose_drawing.draw_landmarks(
                annotated_frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
            
            # 왼쪽 손목 랜드마크 (LEFT_WRIST)
            left_wrist_landmark = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_WRIST]
            current_left_wrist_pt = None
            if left_wrist_landmark.visibility > self.MIN_WRIST_VISIBILITY_MP:
                current_left_wrist_pt = (int(left_wrist_landmark.x * frame_width), int(left_wrist_landmark.y * frame_height))
                if self.prev_left_wrist_pt:
                    dist_left = np.linalg.norm(np.array(current_left_wrist_pt) - np.array(self.prev_left_wrist_pt))
                    if dist_left > self.MIN_WRIST_MOVEMENT_THRESHOLD_MP:
                        is_moving = True
            
            # 오른쪽 손목 랜드마크 (RIGHT_WRIST)
            right_wrist_landmark = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.RIGHT_WRIST]
            current_right_wrist_pt = None
            if right_wrist_landmark.visibility > self.MIN_WRIST_VISIBILITY_MP:
                current_right_wrist_pt = (int(right_wrist_landmark.x * frame_width), int(right_wrist_landmark.y * frame_height))
                if self.prev_right_wrist_pt:
                    dist_right = np.linalg.norm(np.array(current_right_wrist_pt) - np.array(self.prev_right_wrist_pt))
                    if dist_right > self.MIN_WRIST_MOVEMENT_THRESHOLD_MP:
                        is_moving = True
            
            # 현재 손목 좌표 저장 (다음 프레임 비교용)
            self.prev_left_wrist_pt = current_left_wrist_pt if current_left_wrist_pt else self.prev_left_wrist_pt
            self.prev_right_wrist_pt = current_right_wrist_pt if current_right_wrist_pt else self.prev_right_wrist_pt
            
            # 모든 포즈 랜드마크를 리스트로 저장 (필요하다면)
            landmarks = [(lm.x, lm.y, lm.z, lm.visibility) for lm in results.pose_landmarks.landmark]

        # 여기서는 MediaPipe로 '어떤 손 씻기 동작인지'를 직접 분류하는 것이 아니라
        # 손목 움직임 여부 등을 판단하는 데 사용.
        # 실제 동작 분류는 별도의 MediaPipe 기반 동작 분류 모델이 필요함.
        # 현재 코드에서는 움직임 여부만 반환.
        # 만약 MediaPipe로 손 씻기 동작을 분류하는 모델을 추가한다면 prediction 로직을 수정해야 합니다.
        prediction = 'Handwashing' if is_moving else 'No Movement' # 기본값

        return {
            'prediction': prediction, 
            'landmarks': landmarks, 
            'is_moving': is_moving,
            'annotated_frame': annotated_frame
        }