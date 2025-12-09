# 🧼 CleanCheck: 온디바이스 AI 기반 손 위생 커버리지 시각화 시스템(PC 버전)

## Acknowledgement
> 이 프로젝트는 2025년도 정부(과학기술정보통신부)의 재원으로 정보통신기획평가원의 지원을 받아 수행된 연구임
> (No.RS-2022-00155857, 인공지능융합혁신인재양성(충남대학교))
>
> This project was supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korean government (MSIT) (No.RS-2022-00155857, AI Convergence Innovation Talent Training Program at Chungnam National University).

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-orange)](https://github.com/ultralytics/ultralytics)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-brightgreen)](https://mediapipe.dev/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **CleanCheck**는 컴퓨터 비전 기술을 활용하여 WHO 표준 6단계 손 씻기 동작을 인식하고, 손의 세정 영역을 실시간으로 시각화(Visualization)하여 즉각적인 피드백을 제공하는 **온디바이스 AI 기반 손 위생 코칭 시스템**입니다.

정형화된 동작은 **YOLOv8**로 분류하고, 비정형적인 문지름 동작은 **MediaPipe**와 **기하학적 분석**을 결합한 하이브리드 로직을 통해 "지금 내 손이 꼼꼼히 씻기고 있는지"를 직관적인 히트맵으로 보여줍니다.

---

## 🌐 CleanCheck

[cleancheck.org](https://cleancheck.org) (HomePage)
[![Clean Check DEMO](https://img.youtube.com/vi/GN2RJRM0xCs/0.jpg)](https://www.youtube.com/watch?v=GN2RJRM0xCs)

> **깨끗한 손, 안전한 일상 – CleanCheck가 함께합니다.**

---

## 📑 목차

1. [주요 기능](#-주요-기능-key-features)
2. [핵심 기술 및 알고리즘](#-핵심-기술-및-알고리즘-core-technology)
3. [시작하기](#-시작하기-getting-started)
4. [프로젝트 구조](#-프로젝트-구조-project-structure)
5. [향후 계획](#-향후-계획-roadmap)
6. [기여하기](#-기여하기-contributing)
7. [팀 정보](#-팀-정보-team)

---

## ✨ 주요 기능 (Key Features)

- **WHO 6단계 표준 동작 코칭**
  YOLOv8n 모델을 사용하여 WHO가 권장하는 6가지 손 씻기 동작(손바닥, 손등, 손깍지 등)을 실시간으로 분류합니다. 절차적 준수도를 평가하여 누락된 단계를 안내합니다.

- **실시간 세정 커버리지 시각화**
  MediaPipe Hand Landmarker로 추출한 21개 랜드마크를 기반으로 손의 표면적을 계산하고, 반대편 손과의 접촉 여부를 판정하여 세정된 부위를 색상 오버레이로 즉시 시각화합니다.

- **AI 기반 정밀 분석 및 피드백**
  단순한 동작 인식을 넘어, 칼만 필터(Kalman Filter)를 적용하여 손 떨림을 보정하고 정확한 접촉 좌표를 추적합니다. 사용자가 손을 씻는 동안 놓치기 쉬운 부위를 실시간으로 피드백합니다.

- **개인정보 보호 (On-device AI)**
  모든 영상 데이터는 외부 서버 전송 없이 로컬 디바이스에서 처리되므로, 병원 등 민감한 환경에서도 개인정보 유출 걱정 없이 안전하게 사용할 수 있습니다.

---

## 🛠️ 핵심 기술 및 알고리즘 (Core Technology)

본 시스템은 정밀한 손 위생 평가를 위해 다음과 같은 하이브리드 분석 로직을 사용합니다.

| 구분 | 기술 스택 / 알고리즘 | 설명 |
| :--- | :--- | :--- |
| **동작 인식** | **YOLOv8 (Nano)** | 커스텀 데이터셋(약 5,800장)으로 학습된 모델을 통해 6단계 표준 제스처 분류 |
| **랜드마크 추적** | **MediaPipe Hands** | 양손의 21개 키포인트를 3D 좌표로 실시간 추적 |
| **좌표 보정** | **Kalman Filter** | 등속 모델 기반 필터링으로 랜드마크의 Jitter(떨림) 현상 제거 및 궤적 평활화 |
| **영역 분석** | **Convex Hull** & **Point-in-Polygon** | 손의 외곽선(Convex Hull)을 생성하고, 교차 검사를 통해 물리적 접촉 및 세정 여부 판정 |
| **GUI** | **customtkinter** | 직관적인 사용자 경험을 위한 모던 Python GUI 프레임워크 |

---

## 🚀 시작하기 (Getting Started)

### 1. 사전 요구 사항 (Prerequisites)
- Python 3.10 이상
- CUDA 지원 GPU (권장, YOLOv8 가속용)
- 웹캠 (720p 이상 권장)

### 2. 설치 (Installation)

```bash
# 1) 리포지토리 클론
$git clone [https://github.com/chanwoopark11/CleanCheck.git$](https://github.com/chanwoopark11/CleanCheck.git$) cd CleanCheck

# 2) 가상환경 생성 (선택 사항)
$python -m venv venv$ source venv/bin/activate  # Windows: venv\Scripts\activate

# 3) 의존성 패키지 설치
$ pip install -r requirements.txt
````

### 3\. 실행 (Usage)

```bash
# 애플리케이션 실행
$ python src/desktop-windows/main.py
```

프로그램이 실행되면 웹캠 권한을 허용해주세요. 메인 화면에서 "시작" 버튼을 누르면 AI 분석이 시작됩니다.

-----

## 📂 프로젝트 구조 (Project Structure)

```text
CleanCheck/
├─ docs/                  # 문서, 발표 자료 및 시연 이미지
│  ├─ mediapipe/
│  └─ yolo/
├─ experiments/           # 알고리즘 검증 및 실험용 스크립트
│  ├─ mediapipe/          # 랜드마크 추적 및 필터링 실험
│  └─ yolo/               # YOLO 모델 학습 및 추론 테스트
├─ models/                # 학습된 모델 파일 (.pt, .tflite)
├─ src/                   # 소스 코드
│  ├─ desktop-windows/    # 윈도우 데스크톱 클라이언트
│  │  ├─ core/            # AI 로직 (Detector, Analyzer, KalmanFilter)
│  │  ├─ models/          # 데이터 구조 및 스키마
│  │  └─ ui/              # GUI (customtkinter) 구성
│  └─ mobile/             # (Planned) Android 클라이언트 소스
└─ requirements.txt       # 의존성 목록
```

-----

## 🗺️ 향후 계획 (Roadmap)

  - [x] **핵심 기능 구현** (YOLOv8 동작 인식, MediaPipe 시각화)
  - [ ] **모델 경량화 & 최적화**
      - [ ] 모델 양자화(Quantization)를 통한 추론 속도 개선
      - [ ] 저사양 기기 지원을 위한 알고리즘 최적화
  - [ ] **지능형 피드백 고도화**
      - [ ] 사용자가 자주 놓치는 부위 통계 분석 및 예측 알림
  - [ ] **플랫폼 확장**
      - [ ] Android / iOS 모바일 앱 정식 출시 (Kotlin/Swift)
      - [ ] 웹 브라우저 기반 구동 (WebAssembly)

-----

## 🤝 기여하기 (Contributing)

CleanCheck 프로젝트에 기여하고 싶으신가요? Pull Request와 Issue는 언제나 환영입니다\!

1.  `dev` 브랜치에서 새로운 기능을 구현해 주세요.
2.  커밋 메시지는 **Conventional Commits** 규칙을 따르는 것을 권장합니다.
3.  PR 전 `flake8` 등을 통해 코드 스타일을 점검해 주세요.

-----

## 🧑‍💻 팀 정보 (Team)

| 역할 | 이름 | 연락처 (Email) | 소속 |
| :--- | :--- | :--- | :--- |
| **프로젝트 매니저** | **주다빈** | [programbins@gmail.com](mailto:programbins@gmail.com) | 충남대학교 컴퓨터융합학부 |
| **AI CV 모델링** | **최민서** | [msc503@naver.com](mailto:msc503@naver.com) | 충남대학교 컴퓨터융합학부 |
| **데이터 수집·검증** | **박찬우** | [pcw22600@gmail.com](mailto:pcw22600@gmail.com) | 충남대학교 컴퓨터융합학부 |
| **실험 및 검증** | **민영순** | [zhzhxh@cnuh.co.kr](mailto:zhzhxh@cnuh.co.kr) | 충남대학교병원 |
| **프로젝트 지도** | **김재정 교수** | [jjkim@cnu.ac.kr](mailto:jjkim@cnu.ac.kr) | 충남대학교 / 바이오AI융합연구센터 |

-----

Copyright © 2025 CleanCheck Team. All Rights Reserved.
