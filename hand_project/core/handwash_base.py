# core/handwash_base.py
from abc import ABC, abstractmethod
import numpy as np

class HandwashDetector(ABC):
    """
    손 씻기 감지 시스템을 위한 추상 베이스 클래스입니다.
    모든 구체적인 감지 방법(YOLO, MediaPipe 등)은 이 클래스를 상속받아
    필수 메서드를 구현해야 합니다.
    """
    def __init__(self, class_names: list):
        """
        HandwashDetector의 기본 생성자입니다.
        
        Args:
            class_names (list): 감지할 손 씻기 동작 클래스 이름 목록.
        """
        self.class_names = class_names

    @abstractmethod
    def load_model(self, model_path: str):
        """
        감지 모델을 로드하는 추상 메서드입니다.
        구체적인 구현 클래스는 이 메서드를 오버라이드하여 모델 로직을 작성해야 합니다.
        
        Args:
            model_path (str): 모델 파일의 경로.
        """
        pass

    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        단일 비디오 프레임을 처리하여 손 씻기 동작을 감지하는 추상 메서드입니다.
        
        Args:
            frame (np.ndarray): 처리할 비디오 프레임 (OpenCV 이미지).
            
        Returns:
            dict: 감지 결과를 담은 딕셔너리.
                  'prediction': 감지된 동작 이름 (str)
                  'annotated_frame': 바운딩 박스/랜드마크가 그려진 프레임 (np.ndarray)
                  (추가적인 정보는 각 구현 클래스에서 추가될 수 있음)
        """
        pass