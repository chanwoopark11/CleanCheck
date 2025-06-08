# 🧼 CleanCheck: 핸드 제스쳐 인식 기반 손씻기 영역 시각화 시스템

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-orange)](https://github.com/ultralytics/ultralytics)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-brightgreen)](https://mediapipe.dev/)

> **CleanCheck**는 WHO 표준 6단계 손씻기 동작과 자유 손씻기 모두를 실시간으로 인식하여, 손의 세정 영역을 직관적인 색상으로 시각화해 주는 **AI 기반 손 위생 코칭 데스크톱 애플리케이션**입니다.

핸드 트래킹 · 객체 탐지를 결합해 \*\*“지금 내 손이 제대로 씻기고 있는지”\*\*를 한눈에 보여 주며, 부족한 부위와 생략한 단계를 즉시 피드백합니다.

---

## 📑 목차

1. [주요 기능](#-주요-기능-key-features)
2. [사용 기술](#-사용-기술-tech-stack)
3. [시작하기](#-시작하기-getting-started)
4. [프로젝트 구조](#-프로젝트-구조-project-structure)
5. [향후 계획](#-향후-계획-roadmap)
6. [기여하기](#-기여하기-contributing)
7. [라이선스](#-라이선스-license)
8. [팀 정보](#-팀-정보-team)

---

## ✨ 주요 기능 (Key Features)

* **WHO 6단계 손씻기 코칭**
  YOLOv8 모델로 WHO 6단계 손씻기 동작을 실시간 분류하고, *20‑프레임 슬라이딩 윈도우*에서 10프레임 이상 지속 시 해당 단계를 확정합니다. 단계별 누적 시간을 측정해 **5초 미만** 단계에 재실행 알림을 제공합니다.

* **세정 영역 실시간 시각화**
  MediaPipe Holistic로 손 랜드마크 84개를 트래킹하고, Kalman 필터와 보간 알고리즘으로 오클루전을 보정합니다. 손 표면 접촉 여부를 계산해 **세척된 영역을 색상 오버레이**로 표시합니다.

* **결과 분석 & 피드백**
  손씻기 종료(기본 60초) 후 각 단계별 소요 시간·누락 부위·정지 경고(0.5초↑)를 요약 리포트로 제공합니다.

* **안정적인 손 추적**
  Frameless Kalman 예측 + 손목 Δ 기반 좌/우 판별로 저조도·빠른 움직임 환경에서도 안정적인 인식률을 달성합니다.

---

## 🛠️ 사용 기술 (Tech Stack)

| 구분          | 내용                                                         |
| ----------- | ---------------------------------------------------------- |
| **언어**      | Python 3.11                                                |
| **AI / CV** | Ultralytics YOLOv8, OpenCV 4.10, MediaPipe 0.10 (Holistic) |
| **GUI**     | customtkinter v5                                           |
| **패키지 관리**  | pip + `requirements.txt`                                   |
| **개발 환경**   | Windows 11 / WSL2 Ubuntu 22.04 (GPU 가속 지원)                 |

---

## 🚀 시작하기 (Getting Started)

### 1. 사전 요구 사항 (Prerequisites)

* Python ≥ 3.10
* pip (최신 버전 권장)
* 웹캠 (720p 이상 권장)

### 2. 설치 (Installation)

```bash
# 1) 리포지토리 클론
$ git clone https://github.com/<your‑username>/CleanCheck.git
$ cd CleanCheck

# 2) 의존성 설치
$ pip install -r requirements.txt

# 3) 학습된 YOLOv8 가중치 다운로드 (예: best.pt) 후 models/ 에 배치
#    👉 모델 다운로드 링크나 경로를 여기에 기재하세요.
```

### 3. 실행 (Usage)

```bash
# 데스크톱 앱 실행
$ python src/desktop-windows/main.py
```

실행 후 **웹캠 허용** 팝업이 뜨면 승인하세요. 인트로 화면 → “손씻기 시작” 버튼을 클릭하면 실시간 분석이 시작됩니다.

---

## 📂 프로젝트 구조 (Project Structure)

```text
CleanCheck/
├─ docs/                  # 문서·발표 자료 (YOLO 결과, 손 모델 이미지 등)
│  ├─ mediapipe/
│  └─ yolo/
├─ experiments/           # 모델·알고리즘 검증 스크립트
│  ├─ mediapipe/
│  └─ yolo/
├─ legacy/                # 사용되지 않는 이전 버전 코드 보관
├─ models/                # 학습된 모델 파일 (.pt 등)
└─ src/                   # 애플리케이션 소스
   ├─ desktop-windows/    # 윈도우 데스크톱 클라이언트
   │  ├─ core/            # 비전·AI 로직
   │  ├─ models/          # 데이터 모델 클래스
   │  └─ ui/              # GUI 화면 구성
   └─ mobile/             # (예정) 모바일 앱 클라이언트
```

> **TIP**: 스크린샷(GIF)·데모 영상을 `docs/` 하위에 두고, README 내에서 `![demo](docs/.../demo.gif)` 식으로 바로 임베드하면 좋습니다.

---

## 🗺️ 향후 계획 (Roadmap)

* [ ] **모델 개선** – YOLO 모델 경량화(TensorRT) 및 모바일 GPU 최적화
* [ ] **기능 강화** – 미씻은 부위 실시간 하이라이트, 세부 결과 대시보드
* [ ] **플랫폼 확장** – Android / iOS / WebAssembly 지원
* [ ] **UI·UX 개선** – 사용자 맞춤 레이아웃(어린이·병원 모드) 및 다국어 지원
* [ ] **신뢰성 향상** – UV‑잉크 검증 데이터셋 추가 수집, 지속적 재학습 파이프라인 구축

---

## 🤝 기여하기 (Contributing)

Pull Request와 Issue는 언제나 환영입니다! 

1. `dev` 브랜치에서 기능을 구현한 뒤 PR을 보내 주세요.
2. 커밋 메시지는 **Conventional Commits** 스타일을 권장합니다.
3. **pre‑commit** 훅(`flake8`, `isort`, `black`)이 자동으로 코드 스타일을 검사합니다.

---

## 📄 라이선스 (License)

이 프로젝트는 **MIT License**를 따릅니다. 자세한 내용은 `LICENSE` 파일을 확인하세요.

---

## 🧑‍💻 팀 정보 (Team)

| 역할                | 이름  | 연락처                                                       |
| ----------------- | --- | --------------------------------------------------------- |
| 프로젝트 매니저 / CV 모델링 | 홍길동 | [example1@cleancheck.dev](mailto:example1@cleancheck.dev) |
| GUI/UX 개발         | 김다빈 | [example2@cleancheck.dev](mailto:example2@cleancheck.dev) |
| 데이터 수집·검증         | 이지현 | [example3@cleancheck.dev](mailto:example3@cleancheck.dev) |

> 궁금한 점이 있다면 언제든 메일 주세요. 프로젝트에 기여해 주셔서 감사합니다!
