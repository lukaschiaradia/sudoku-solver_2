from pathlib import Path

import pyautogui


def capture_screen(output_path: str = 'outputs/screenshot.png') -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    screenshot = pyautogui.screenshot()
    screenshot.save(output_path)
    return output_path
