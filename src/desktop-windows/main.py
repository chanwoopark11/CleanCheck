# main.py
import customtkinter
import tkinter # _tkinter.TclError를 명시적으로 다루기 위해 추가 (필수는 아님)

from ui.menu_screen import MenuScreen
from ui.execution_screen import ExecutionScreen
from ui.status_screen import StatusScreen
from ui.settings_screen import SettingsScreen
from ui.cleansed_by_part_screen import CleansedByPartScreen # [추가] 새로운 화면 임포트

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("CleanCheck")
        # 지오메트리 설정은 사용자의 화면 해상도 및 선호에 따라 조절
        self.geometry(f"{self.winfo_screenwidth()//2 + 200}x{self.winfo_screenheight()//2 + 200}") # 예시
        self.minsize(800, 600) # 최소 크기 설정
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.container = customtkinter.CTkFrame(self)
        self.container.grid(row=0, column=0, sticky="nsew")
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        self.handwash_results_data = {} # 이전과 동일

        self.create_frames()
        self.handle_action("메뉴 화면으로 이동 요청") # 시작 화면

    def create_frames(self):
        """애플리케이션에서 사용될 모든 프레임(화면)을 생성합니다."""
        self.frames["MenuScreen"] = MenuScreen(parent=self.container, controller=self)
        self.frames["ExecutionScreen"] = ExecutionScreen(parent=self.container, controller=self)
        self.frames["StatusScreen"] = StatusScreen(parent=self.container, controller=self)
        self.frames["SettingsScreen"] = SettingsScreen(parent=self.container, controller=self)
        # [추가] 새로운 CleansedByPartScreen 인스턴스 생성
        self.frames["CleansedByPartScreen"] = CleansedByPartScreen(parent=self.container, controller=self)

        for name, frame in self.frames.items():
            frame.grid(row=0, column=0, sticky="nsew") # 모든 프레임을 동일한 위치에 배치

    def show_frame(self, page_name, data=None):
        """지정된 이름의 프레임을 화면에 표시합니다."""
        # 현재 활성화된 (is_running=True) 다른 측정 화면이 있다면 중지
        for f_name, f_instance in self.frames.items():
            if f_name != page_name and hasattr(f_instance, 'is_running') and f_instance.is_running:
                if hasattr(f_instance, 'stop_measurement'):
                    print(f"Switching away from {f_name}, stopping measurement.")
                    f_instance.stop_measurement()

        frame = self.frames[page_name]
        if hasattr(frame, "on_show"): # 화면 표시 전 필요한 초기화 작업 수행
            if page_name == "StatusScreen" and data is not None:
                frame.on_show(data)
            else:
                frame.on_show() # ExecutionScreen, SettingsScreen, CleansedByPartScreen
        frame.tkraise() # 해당 프레임을 가장 위로 올림

    def handle_action(self, action_message: str, data=None):
        """UI 요소로부터의 액션 요청을 중앙에서 처리합니다."""
        print(f"Controller received action: {action_message}")
        if action_message == "6단계 손씻기 실행 요청":
            self.show_frame("ExecutionScreen")
        elif action_message == "손씻기 결과 실행 요청":
            results_to_show = data if data is not None else self.get_handwash_results()
            self.show_frame("StatusScreen", data=results_to_show)
        elif action_message == "설정 화면으로 이동 요청":
            self.show_frame("SettingsScreen")
        elif action_message == "부위별 손씻기 실행 요청": # [추가] 새로운 화면 전환 액션
            self.show_frame("CleansedByPartScreen")
        elif action_message == "메뉴 화면으로 이동 요청":
            self.show_frame("MenuScreen")
        elif action_message == "애플리케이션 종료 요청":
            self.quit_app()
        else:
            print(f"알 수 없는 액션 요청: {action_message}")

    def set_handwash_results(self, results): # 이전과 동일
        self.handwash_results_data = results

    def get_handwash_results(self): # 이전과 동일
        return self.handwash_results_data

    def get_execution_settings(self): # 이전과 동일
        exec_screen = self.frames.get("ExecutionScreen")
        if exec_screen:
            settings = {}
            for key in exec_screen.settings_types.keys():
                if hasattr(exec_screen, key):
                    settings[key] = getattr(exec_screen, key)
            return settings
        return {}

    def update_execution_settings(self, new_settings: dict): # 이전과 동일
        exec_screen = self.frames.get("ExecutionScreen")
        if exec_screen:
            for key, value in new_settings.items():
                if hasattr(exec_screen, key):
                    target_type = exec_screen.settings_types.get(key, str)
                    try:
                        converted_value = target_type(value)
                        setattr(exec_screen, key, converted_value)
                        print(f"설정 업데이트: {key} = {converted_value} (타입: {target_type.__name__})")
                    except ValueError:
                        print(f"오류: {key}에 대한 값 ({value})을(를) {target_type.__name__}(으)로 변환할 수 없습니다.")
            # 설정 변경 후 UI 업데이트 (필요시)
            if "MAX_MEASUREMENT_DURATION_SEC" in new_settings and exec_screen.is_measuring:
                 exec_screen.time_label.configure(text=f"시간: ... / {exec_screen.MAX_MEASUREMENT_DURATION_SEC}s")


    def quit_app(self):
        """애플리케이션을 종료합니다. 실행 중인 측정이 있다면 중지합니다."""
        print("애플리케이션 종료 중...")
        for frame_name, frame_instance in self.frames.items():
            if hasattr(frame_instance, 'is_running') and frame_instance.is_running:
                if hasattr(frame_instance, 'stop_measurement'):
                    print(f"종료 전 {frame_name}의 측정/분석 중지 중...")
                    frame_instance.stop_measurement()
        self.destroy() # CTk 윈도우 파괴
        # self.quit() # CTk의 mainloop를 중단 (destroy()가 내부적으로 처리할 수 있음)

if __name__ == "__main__":
    customtkinter.set_appearance_mode("System") # System, Dark, Light
    customtkinter.set_default_color_theme("blue") # blue, green, dark-blue

    app = App()
    # 창 닫기 버튼(X) 클릭 시 quit_app 메서드 호출 설정
    app.protocol("WM_DELETE_WINDOW", app.quit_app)
    app.mainloop()