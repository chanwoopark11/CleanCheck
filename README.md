# Deep Clean 
**핸드 제스처 인식 기반 손씻은 영역 시각화 시스템**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue?logo=python)](https://www.python.org/)  
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-orange)](https://github.com/ultralytics/ultralytics)  
[![MediaPipe 0.10.22](https://img.shields.io/badge/MediaPipe-0.10.22-brightgreen)](https://developers.google.com/mediapipe)

> **Deep Clean**은 WHO 6단계 손씻기 동작을 실시간으로 인식하고, 자유 손씻기에서도 세척 영역을 직관적으로 시각화하여  
> 의료진과 일반 사용자 모두에게 *“지금 내가 제대로 씻고 있는지”* 를 한눈에 보여주는 AI 손위생 어시스턴트입니다. :contentReference[oaicite:0]{index=0}

---

## 목차
1. [프로젝트 개요](#프로젝트-개요)  
2. [주요 기능](#주요-기능)  
3. [데모](#데모)  
4. [시스템 아키텍처](#시스템-아키텍처)  
5. [설치 & 실행](#설치--실행)  
6. [디렉터리 구조](#디렉터리-구조)  
7. [로드맵](#로드맵)  
8. [기여](#기여)  
9. [라이선스](#라이선스)  

---

## 프로젝트 개요
| 항목 | 내용 |
|------|------|
| **팀명** | CleanCheck |
| **참가 부문** | 2025 CNU Project Fair – 개발과제 |
| **목표** | ① WHO 6단계 **손씻기 동작** 검출<br>② **자유 손씻기** 세척 영역 실시간 시각화<br>③ 두 방식 **정확도·사용성·만족도** 비교 |
| **핵심 기술** | Ultralytics YOLOv8 + Kalman Filter, MediaPipe Holistic, OpenCV, CustomTkinter GUI |
| **실행 환경** | Windows 11 / WSL2 Ubuntu 22.04, Intel Ultra 7 228V, 32 GB RAM, Logitech C920 Webcam |

---

## 주요 기능
### 1️⃣ WHO 6-단계 손씻기 인식
* YOLOv8n 학습 (mAP50 0.992)으로 6단계 바운딩박스를 분류  
* 20 프레임 슬라이딩 윈도우 → 10 프레임 이상 지속 시 *확정*  
* 단계별 누적 시간 계산 & 5 초 미만 단계에 **재시도** 알림 :contentReference[oaicite:1]{index=1}

### 2️⃣ 자유 손씻기 세척 영역 시각화
* MediaPipe Holistic로 84 개 손-키포인트 추적  
* Kalman Filter + Wrist Δ 보간으로 오클루전 복원  
* 접촉 횟수·시간으로 세척 여부 판단 → 손 모델에 **색상 오버레이** :contentReference[oaicite:2]{index=2}

### 3️⃣ 결과 리포트 & 경고
* 60 초 경과 시 결과 화면 전환  
* 부족 부위, 생략 단계, 정지 경고(0.5 초 이상) 등을 종합 리포트

---

## 데모

## 시스템 아키텍처
