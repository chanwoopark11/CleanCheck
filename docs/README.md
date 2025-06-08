
| 구분 | 주요 내용 | 예시 파일 |
|------|-----------|-----------|
| **mediapipe/** | 손 랜드마크 추적·보간 실험 캡처, Kalman 필터 비교 GIF | `CleanCheck/docs/mediapipe/img/holistic_보간.gif` |
| **yolo/** | WHO 6단계 동작 분류 모델 학습 로그, confusion matrix | `Train_v1/result/val_confusion_matrix.png` |
| **img/** | 문서에 직접 삽입할 고해상도 이미지 자산 | `yolo/img/stage3_example.jpg` |

> 이미지·영상은 Git LFS 또는 외부 스토리지(예: Nextcloud)로 관리 가능하며, README/보고서에서는 **절대 경로 대신 URL** 또는 상대 경로를 사용해 주세요.

---

## 📝 작성·추가 규칙

1. **폴더 구분**  
   - `mediapipe/`, `yolo/`처럼 **기술·모델 단위**로 나누고, 하위에 `img/`, `result/` 등을 둡니다.  
2. **파일 명명**  
   - 스크린샷: `<기능>_<YYYYMMDD>.png`  
   - 그래프/로그: `<실험이름>_<지표>.pdf`  
3. **대용량 파일**  
   - 100 MB 초과 파일은 `docs`에 직접 커밋하지 말고, `docs/README_external.md`에 링크만 남깁니다.  
4. **Markdown 가이드**  
   - 모든 문서는 **H2(`##`)까지만 목차에 포함**될 수 있게 작성하고, 코드 블록 언어 지정을 합니다.  
5. **PR 체크리스트**  
   - 파일 경로 / 링크 확인 ↔ 빌드(예: MkDocs) 오류 없는지 → 리뷰어 확인 후 병합

---

## 🔍 빠른 참조

| 참고 문서 | 설명 |
|-----------|------|
| [`/docs/mediapipe/README.md`](mediapipe/README.md) | 손 랜드마크 추적 실험 설정, 파라미터 표 |
| [`/docs/yolo/README.md`](yolo/README.md) | 6단계 동작 분류 모델 학습·평가 절차 |
| [`/README.md`](../README.md) (루트) | 프로젝트 개요, 설치 및 실행 방법 |

---

## 🤝 기여 방법

1. 이슈 또는 Pull Request로 문서 추가·수정 의사를 알립니다.  
2. 양식에 맞춰 파일을 `/docs` 하위에 배치 → `git add` → 커밋 메시지에 **[docs]** 접두어를 붙여 주세요.  
3. 리뷰 승인 후 `main`에 병합되면, CI가 문서 사이트(Build → `/site`)를 자동 업데이트합니다.  

---

문서를 통해 **CleanCheck**의 모든 실험·성과를 한눈에 파악할 수 있도록 지속적으로 업데이트해 주세요!
