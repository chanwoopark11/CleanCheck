# ui/status_screen.py
import customtkinter

class StatusScreen(customtkinter.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.grid_rowconfigure(0, weight=0) # 타이틀
        self.grid_rowconfigure(1, weight=1) # 결과 텍스트박스
        self.grid_rowconfigure(2, weight=0) # 버튼
        self.grid_columnconfigure(0, weight=1)

        self.label = customtkinter.CTkLabel(self, text="손 씻기 결과",
                                            font=customtkinter.CTkFont(size=24, weight="bold"))
        self.label.grid(row=0, column=0, pady=20)

        self.results_text = customtkinter.CTkTextbox(self, wrap="word", width=600, height=350,
                                                     corner_radius=10, font=customtkinter.CTkFont(size=14))
        self.results_text.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.results_text.configure(state="disabled")

        self.back_button = customtkinter.CTkButton(self, text="메뉴로 돌아가기",
                                                   command=lambda: controller.handle_action("메뉴 화면으로 이동 요청"),
                                                   font=customtkinter.CTkFont(size=18), height=40)
        self.back_button.grid(row=2, column=0, pady=20)

    def on_show(self, data=None):
        self.update_status(data)

    def update_status(self, data):
        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", "end")

        if not data:
            self.results_text.insert("end", "표시할 손 씻기 데이터가 없습니다.")
            self.results_text.configure(state="disabled")
            return

        result_type = data.get("type")

        # --- 수정된 부분 시작 ---
        # type 키가 없고, 기존 ExecutionScreen 결과로 추정되는 키 (action_durations)가 있다면
        # result_type을 "6_step_handwash"로 간주합니다.
        if result_type is None and "action_durations" in data and "action_counts" in data:
            assumed_type = "6_step_handwash"
            print(f"StatusScreen: 'type' key not found in data, assuming '{assumed_type}' based on content.")
        elif result_type is None and "left_percentage" in data and "right_percentage" in data: # CleansedByPartScreen의 레거시 데이터 (type 없는 경우)
             assumed_type = "cleansed_by_part"
             print(f"StatusScreen: 'type' key not found in data, assuming '{assumed_type}' based on content.")
        else:
            assumed_type = result_type
        # --- 수정된 부분 끝 ---


        if assumed_type == "cleansed_by_part":
            self.label.configure(text="부위별 손씻기 분석 결과")
            total_time = data.get("total_time", 0)
            left_percentage = data.get("left_percentage", 0)
            right_percentage = data.get("right_percentage", 0)

            self.results_text.insert("end", f"--- 부위별 손씻기 분석 요약 ---\n\n")
            self.results_text.insert("end", f"총 분석 시간: {total_time:.2f} 초\n\n")
            self.results_text.insert("end", f"왼손 세척률: {left_percentage:.1f}%\n")
            self.results_text.insert("end", f"오른손 세척률: {right_percentage:.1f}%\n\n")

            overall_avg_percentage = (left_percentage + right_percentage) / 2
            quality = ""
            if overall_avg_percentage >= 90:
                quality = "매우 꼼꼼하게 잘 씻으셨습니다! 👍"
            elif overall_avg_percentage >= 70:
                quality = "잘 하셨어요! 조금만 더 신경쓰면 완벽해요. 😊"
            elif overall_avg_percentage >= 50:
                quality = "괜찮아요. 다음엔 놓친 부분 없이 씻어봐요. 😉"
            else:
                quality = "더 꼼꼼한 손씻기가 필요해 보여요. 💪"
            
            self.results_text.insert("end", f"종합 평가: {quality} (평균 세척률: {overall_avg_percentage:.1f}%)\n\n")
            self.results_text.insert("end", "팁: 손가락 사이사이와 손톱 밑까지 신경 써주세요!")

        elif assumed_type == "6_step_handwash":
            self.label.configure(text="6단계 손씻기 결과")
            total_time = data.get("total_time", 0)
            action_durations = data.get("action_durations", {})
            action_counts = data.get("action_counts", {})
            
            try:
                exec_screen = self.controller.frames.get("ExecutionScreen")
                if not exec_screen: raise KeyError("ExecutionScreen not found")
                class_names_handwash = exec_screen.CLASS_NAMES_HANDWASH
                rec_target_duration = exec_screen.RECOMMENDATION_TARGET_DURATION_SEC
            except KeyError:
                self.results_text.insert("end", "오류: 6단계 손씻기 세부 정보 참조 중 문제가 발생했습니다.\n")
                class_names_handwash = [ # ExecutionScreen.CLASS_NAMES_HANDWASH 기본값과 유사하게
                    "0.Palm to Palm", "1.Back of Hands", "2.Interlaced Fingers",
                    "3.Backs of Fingers", "4.Thumbs", "5.Fingertips and Nails"
                ]
                rec_target_duration = 5.0 # ExecutionScreen.RECOMMENDATION_TARGET_DURATION_SEC 기본값과 유사하게

            self.results_text.insert("end", f"--- 6단계 손 씻기 세션 요약 ---\n\n")
            self.results_text.insert("end", f"총 손 씻기 시간: {total_time:.2f} 초\n\n")
            
            self.results_text.insert("end", "[각 동작별 지속 시간 (움직임 감지 시)]\n")
            if action_durations and class_names_handwash:
                for name in class_names_handwash:
                    duration = action_durations.get(name, 0.0)
                    # 클래스 이름에서 "X." 부분 제거
                    display_name = name.split('.', 1)[-1].strip() if '.' in name else name.strip()
                    self.results_text.insert("end", f"- {display_name}: {duration:.2f} 초\n")
            else:
                self.results_text.insert("end", "  동작별 지속 시간이 기록되지 않았습니다.\n")
            
            self.results_text.insert("end", "\n[각 동작별 카운트 (안정적 유지 시)]\n")
            if action_counts and class_names_handwash:
                for name in class_names_handwash:
                    count = action_counts.get(name, 0)
                    display_name = name.split('.', 1)[-1].strip() if '.' in name else name.strip()
                    self.results_text.insert("end", f"- {display_name}: {count} 회\n")
            else:
                self.results_text.insert("end", "  동작별 카운트가 기록되지 않았습니다.\n")
            
            self.results_text.insert("end", "\n")
            
            overall_score = 0.0
            if class_names_handwash and rec_target_duration > 0 and action_durations:
                achieved_score_sum = 0
                num_actions = len(class_names_handwash)
                for action_name in class_names_handwash:
                    duration = action_durations.get(action_name, 0.0)
                    action_score = min(duration / rec_target_duration, 1.0)
                    achieved_score_sum += action_score
                if num_actions > 0: overall_score = (achieved_score_sum / num_actions) * 100
            
            quality = "더 연습이 필요합니다."
            if overall_score >= 80: quality = "매우 우수합니다! 👍"
            elif overall_score >= 50: quality = "양호합니다! 😊"
                
            self.results_text.insert("end", f"전반적인 손 씻기 품질: {quality} (점수: {overall_score:.1f}%)\n\n")
            self.results_text.insert("end", "손의 모든 부분을 깨끗하게 씻는 것을 잊지 마세요!")
        
        else: # 알 수 없는 타입 또는 모든 조건 불일치
            self.label.configure(text="손 씻기 상태 및 결과")
            self.results_text.insert("end", "결과 데이터 형식을 알 수 없거나, 세부 정보가 부족합니다.\n")
            if data:
                # 데이터 내용을 조금 더 안전하게 문자열로 변환하여 표시
                try:
                    data_str = str(data)
                except Exception as e:
                    data_str = f"(데이터를 문자열로 변환 중 오류: {e})"
                self.results_text.insert("end", f"받은 데이터 (일부): {data_str[:300]}...\n")

        self.results_text.configure(state="disabled")