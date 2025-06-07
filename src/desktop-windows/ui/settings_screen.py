# ui/settings_screen.py
import customtkinter

class SettingsScreen(customtkinter.CTkFrame): #
    """
    애플리케이션의 측정 관련 설정을 변경하는 화면입니다.
    """
    def __init__(self, parent, controller): #
        super().__init__(parent) #
        self.controller = controller #
        self.settings_vars = {} # CTkEntry와 연결될 StringVar들
        self.entry_widgets = {} # CTkEntry 위젯 저장

        self.grid_columnconfigure(0, weight=1) #
        self.grid_columnconfigure(1, weight=2) #
        self.grid_rowconfigure(0, weight=0) # 타이틀
        # 나머지 행들은 동적으로 생성되므로 weight는 content에 따라 조절

        title_label = customtkinter.CTkLabel(self, text="측정 설정 변경", font=customtkinter.CTkFont(size=24, weight="bold")) #
        title_label.grid(row=0, column=0, columnspan=2, pady=20, sticky="n") #

        # 스크롤 가능한 프레임 추가 (설정 항목이 많을 경우 대비)
        self.scrollable_frame = customtkinter.CTkScrollableFrame(self, label_text="설정 항목") #
        self.scrollable_frame.grid(row=1, column=0, columnspan=2, padx=20, pady=(0,10), sticky="nsew") #
        self.grid_rowconfigure(1, weight=1) # 스크롤 프레임이 확장되도록

        self.scrollable_frame.grid_columnconfigure(0, weight=1) # Label
        self.scrollable_frame.grid_columnconfigure(1, weight=1) # Entry

        # ExecutionScreen에서 설정 키와 레이블 가져오기
        # App 클래스가 완전히 초기화 된 후 controller를 통해 접근해야 함
        # 여기서는 일단 플레이스홀더로 두고, on_show에서 실제 생성
        
        # 버튼 프레임
        button_frame = customtkinter.CTkFrame(self) #
        button_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky="s") #
        button_frame.grid_columnconfigure((0,1), weight=1) #


        save_button = customtkinter.CTkButton(button_frame, text="저장", command=self.save_settings) #
        save_button.grid(row=0, column=0, padx=10, pady=10, sticky="ew") #

        back_button = customtkinter.CTkButton(button_frame, text="메뉴로 돌아가기", #
                                              command=lambda: controller.handle_action("메뉴 화면으로 이동 요청")) #
        back_button.grid(row=0, column=1, padx=10, pady=10, sticky="ew") #
        
        self.status_label = customtkinter.CTkLabel(self, text="", font=customtkinter.CTkFont(size=14)) #
        self.status_label.grid(row=3, column=0, columnspan=2, pady=(0,10), sticky="s") #


    def on_show(self): #
        """ 설정 화면이 표시될 때 현재 설정을 불러와 UI에 반영합니다. """
        current_settings = self.controller.get_execution_settings() #
        exec_screen = self.controller.frames.get("ExecutionScreen") # 레이블, 타입 정보 가져오기 위함
        
        if not exec_screen: #
            self.status_label.configure(text="오류: ExecutionScreen을 찾을 수 없습니다.", text_color="red") #
            return

        # 기존 위젯들 제거 (on_show가 여러 번 호출될 경우 중복 생성 방지)
        for widget in self.scrollable_frame.winfo_children(): #
            widget.destroy() #
        self.settings_vars.clear() #
        self.entry_widgets.clear() #

        if not current_settings: #
            self.status_label.configure(text="설정값을 불러올 수 없습니다.", text_color="red") #
            return

        self.status_label.configure(text="") # 이전 상태 메시지 초기화

        row_idx = 0 #
        for key, value in current_settings.items(): #
            label_text = exec_screen.settings_labels.get(key, key) # 한글 레이블 사용
            
            label = customtkinter.CTkLabel(self.scrollable_frame, text=label_text, anchor="w") #
            label.grid(row=row_idx, column=0, padx=10, pady=5, sticky="w") #

            var = customtkinter.StringVar(value=str(value)) #
            self.settings_vars[key] = var #
            
            entry = customtkinter.CTkEntry(self.scrollable_frame, textvariable=var, width=150) #
            entry.grid(row=row_idx, column=1, padx=10, pady=5, sticky="e") #
            self.entry_widgets[key] = entry #
            row_idx += 1 #
        
        print("SettingsScreen: 설정 로드 완료") #

    def save_settings(self): #
        """ 입력된 설정 값들을 ExecutionScreen에 저장합니다. """
        new_settings = {} #
        has_errors = False #
        exec_screen = self.controller.frames.get("ExecutionScreen") #

        if not exec_screen: #
            self.status_label.configure(text="오류: ExecutionScreen을 찾을 수 없습니다.", text_color="red") #
            return

        for key, var in self.settings_vars.items(): #
            value_str = var.get() #
            target_type = exec_screen.settings_types.get(key, str) # 기본 타입은 str
            try:
                if target_type == float: #
                    new_settings[key] = float(value_str) #
                elif target_type == int: #
                    new_settings[key] = int(value_str) #
                else:
                    new_settings[key] = value_str # 문자열 또는 기타 타입
                
                # 입력 필드 배경색 초기화 (오류 없으면)
                if key in self.entry_widgets: #
                    self.entry_widgets[key].configure(border_color=customtkinter.ThemeManager.theme["CTkEntry"]["border_color"]) #

            except ValueError:
                print(f"오류: '{key}'에 대한 값이 잘못되었습니다. ({value_str}) 숫자를 입력해야 합니다.") #
                self.status_label.configure(text=f"'{exec_screen.settings_labels.get(key,key)}'에 유효한 숫자를 입력하세요.", text_color="red") #
                # 오류 발생한 입력 필드 강조 (예: 빨간색 테두리)
                if key in self.entry_widgets: #
                     self.entry_widgets[key].configure(border_color="red") # 기본 테마 색상에 따라 조정 필요
                has_errors = True #
                # return # 첫번째 오류에서 중단하거나, 모든 오류를 표시하려면 주석 처리
        
        if not has_errors: #
            self.controller.update_execution_settings(new_settings) #
            print("SettingsScreen: 설정 저장 완료") #
            self.status_label.configure(text="설정이 성공적으로 저장되었습니다!", text_color="green") #
            # 저장 후 일정 시간 뒤 메시지 초기화
            self.after(3000, lambda: self.status_label.configure(text="")) #
        else:
            print("SettingsScreen: 유효하지 않은 입력값으로 인해 설정 저장 실패") #