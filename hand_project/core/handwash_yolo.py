# core/handwash_yolo.py
import numpy as np
from ultralytics import YOLO
import cv2 # 시각화를 위해 필요할 수 있습니다.

from core.handwash_base import HandwashDetector

class HandwashYOLO(HandwashDetector):
    """
    YOLO(You Only Look Once) 모델을 사용하여 손 씻기 동작을 감지하는 클래스입니다.
    HandwashDetector 추상 베이스 클래스를 구현합니다.
    """
    def __init__(self, class_names: list, model_path: str):
        """
        HandwashYOLO 클래스의 생성자입니다.
        
        Args:
            class_names (list): YOLO 모델의 클래스 이름 목록.
            model_path (str): YOLO 모델 가중치 파일 경로.
        """
        super().__init__(class_names)
        self.model = None
        self.load_model(model_path) # 생성 시 모델 로드

    def load_model(self, model_path: str):
        """
        지정된 경로에서 YOLO 모델을 로드합니다.
        
        Args:
            model_path (str): YOLO 모델 파일의 경로 (예: 'models/best.pt').
        """
        try:
            self.model = YOLO(model_path)
            print(f"YOLO 모델이 '{model_path}'에서 성공적으로 로드되었습니다.")
        except Exception as e:
            print(f"YOLO 모델 로드 오류: {e}")
            self.model = None # 로드 실패 시 모델을 None으로 설정

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        단일 비디오 프레임을 처리하여 손 씻기 동작을 감지합니다.
        
        Args:
            frame (np.ndarray): 처리할 비디오 프레임 (OpenCV 이미지).
            
        Returns:
            dict: 감지 결과를 담은 딕셔너리.
                  'prediction': 감지된 동작 이름 (str)
                  'annotated_frame': 바운딩 박스/레이블이 그려진 프레임 (np.ndarray)
                  'boxes': 감지된 모든 바운딩 박스 정보 (YOLO Results 객체)
        """
        prediction = 'No action' # 기본값
        annotated_frame = frame.copy() # 원본 프레임 복사하여 주석 추가
        results_obj = None # YOLO Results 객체

        if self.model is None:
            print("YOLO 모델이 로드되지 않았습니다.")
            return {'prediction': prediction, 'annotated_frame': annotated_frame, 'boxes': None}
            
        # YOLO 모델 실행
        # verbose=False로 설정하여 콘솔 출력을 줄임
        results = self.model.predict(source=frame, show=False, conf=0.5, iou=0.7, verbose=False) 
        results_obj = results[0] # 첫 번째 결과 (단일 이미지 처리)

        if results_obj and results_obj.boxes:
            boxes = results_obj.boxes
            if boxes:
                # 감지된 모든 바운딩 박스 추출
                # boxes_xyxy = boxes.xyxy.cpu().numpy().tolist() # 사용하지 않는다면 제거

                # 가장 큰 바운딩 박스를 찾아 주요 동작으로 간주
                # (YOLO 모델이 여러 클래스를 동시에 감지할 수 있으므로 가장 큰 객체를 선택하는 방식)
                largest_box = None
                max_area = 0
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        largest_box = box

                if largest_box is not None:
                    class_id = int(largest_box.cls[0])
                    if 0 <= class_id < len(self.class_names):
                        prediction = self.class_names[class_id]
            
            # YOLO가 제공하는 plot 함수를 사용하여 바운딩 박스와 레이블을 프레임에 그립니다.
            # 이 함수는 새로운 이미지 배열을 반환합니다.
            annotated_frame = results_obj.plot()

        return {
            'prediction': prediction, 
            'annotated_frame': annotated_frame, 
            'boxes': results_obj # YOLO Results 객체 자체를 반환하여 필요시 추가 정보 활용
        }