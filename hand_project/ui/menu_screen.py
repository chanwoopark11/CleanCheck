# ui/menu_screen.py
import customtkinter

class MenuScreen(customtkinter.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # [수정] 새로운 메뉴 옵션 추가
        self.menu_options = [
            ("6단계 손씻기", lambda: controller.handle_action("6단계 손씻기 실행 요청")),
            ("부위별 손씻기 분석", lambda: controller.handle_action("부위별 손씻기 실행 요청")), # 새로운 메뉴 항목
            ("손씻기 결과", lambda: controller.handle_action("손씻기 결과 실행 요청")),
            ("설정", lambda: controller.handle_action("설정 화면으로 이동 요청")),
            ("종료", lambda: controller.handle_action("애플리케이션 종료 요청"))
        ]
        self.current_menu_index = 0

        # --- 나머지 UI 구성 코드는 이전과 거의 동일 ---
        self.grid_rowconfigure(0, weight=0) # 로고 행
        self.grid_rowconfigure(1, weight=1) # 메뉴 선택 행 (중앙 정렬을 위해 weight 부여)
        self.grid_rowconfigure(2, weight=0) # 하단 여백 (필요시)

        self.grid_columnconfigure(0, weight=1) # 왼쪽 화살표
        self.grid_columnconfigure(1, weight=2) # 중앙 메뉴 텍스트 (더 많은 공간 할애)
        self.grid_columnconfigure(2, weight=1) # 오른쪽 화살표

        self.logo_label = customtkinter.CTkLabel(
            self,
            text="Clean Check",
            font=customtkinter.CTkFont(size=30, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, columnspan=3, padx=20, pady=(40, 20), sticky="w")


        # 버튼 및 텍스트 색상 (테마에 따라 자동 조정될 수 있음)
        arrow_hover_color = customtkinter.ThemeManager.theme["CTkButton"]["hover_color"]
        text_hover_color = customtkinter.ThemeManager.theme["CTkButton"]["hover_color"] # 동일하게 사용하거나 미세 조정


        self.left_button = customtkinter.CTkButton(
            self, text="◀", command=self._previous_menu, width=60, height=60,
            font=customtkinter.CTkFont(size=28), fg_color="transparent", hover_color=arrow_hover_color
        )
        self.left_button.grid(row=1, column=0, padx=(20, 5), pady=20, sticky="e")

        self.selected_menu_clickable_text = customtkinter.CTkButton(
            self, text=self.menu_options[self.current_menu_index][0],
            font=customtkinter.CTkFont(size=40, weight="bold"),
            command=self._execute_selected_menu, fg_color="transparent",
            hover_color=text_hover_color #, text_color=("gray10", "gray90") # 자동 테마 적용
        )
        self.selected_menu_clickable_text.grid(row=1, column=1, padx=10, pady=20, sticky="nsew")

        self.right_button = customtkinter.CTkButton(
            self, text="▶", command=self._next_menu, width=60, height=60,
            font=customtkinter.CTkFont(size=28), fg_color="transparent", hover_color=arrow_hover_color
        )
        self.right_button.grid(row=1, column=2, padx=(5, 20), pady=20, sticky="w")
        # --- 여기까지 UI 구성 ---

        self._update_menu_display() # 초기 메뉴 텍스트 설정

    def _update_menu_display(self):
        """현재 선택된 메뉴를 버튼 텍스트에 반영합니다."""
        self.selected_menu_clickable_text.configure(text=self.menu_options[self.current_menu_index][0])

    def _previous_menu(self):
        """이전 메뉴 항목으로 이동합니다."""
        self.current_menu_index = (self.current_menu_index - 1 + len(self.menu_options)) % len(self.menu_options)
        self._update_menu_display()

    def _next_menu(self):
        """다음 메뉴 항목으로 이동합니다."""
        self.current_menu_index = (self.current_menu_index + 1) % len(self.menu_options)
        self._update_menu_display()

    def _execute_selected_menu(self):
        """현재 선택된 메뉴의 액션을 실행합니다."""
        action_name, action_func = self.menu_options[self.current_menu_index]
        print(f"메뉴 실행: '{action_name}'")
        action_func()