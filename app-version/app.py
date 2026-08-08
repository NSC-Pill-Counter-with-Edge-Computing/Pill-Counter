import sys
import os
import cv2
import math
import time
import threading
import numpy as np
from collections import deque
from ultralytics import YOLO

# PyQt6 UI imports
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QStackedWidget,
    QLineEdit, QMessageBox, QGraphicsDropShadowEffect, QSizePolicy
)

# GPIO 4 Vibration Support
try:
    from gpiozero import OutputDevice
    from gpiozero.pins.lgpio import LGPIOFactory
    factory = LGPIOFactory(chip=0)
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False


# ============================================================================
# DESIGN TOKENS - ธีม "เครื่องมือแพทย์" (Clinical Teal & White)
# ============================================================================
class Palette:
    BG = "#EAF2F4"
    CARD = "#FFFFFF"
    BORDER = "#D7E3E7"
    PRIMARY = "#0E7C86"
    PRIMARY_DARK = "#0A626B"
    PRIMARY_SOFT = "rgba(14,124,134,0.10)"
    TEXT_DARK = "#1E2E36"
    TEXT_MUTED = "#647982"
    SUCCESS = "#1E9E6B"
    SUCCESS_BG = "rgba(30,158,107,0.12)"
    DANGER = "#D6455A"
    DANGER_BG = "rgba(214,69,90,0.12)"
    WARNING = "#E0912B"
    WARNING_BG = "rgba(224,145,43,0.14)"
    NEUTRAL_BG = "#F1F6F7"


FONT_FAMILY = "Noto Sans Thai, Segoe UI, Arial"


def apply_shadow(widget, blur=24, dx=0, dy=6, alpha=35):
    """ใส่เงานุ่มๆ ให้การ์ด ให้ความรู้สึกสะอาด ยกตัวขึ้นจากพื้นหลัง (คลินิก/พรีเมียม)"""
    effect = QGraphicsDropShadowEffect()
    effect.setBlurRadius(blur)
    effect.setOffset(dx, dy)
    effect.setColor(QColor(15, 46, 54, alpha))
    widget.setGraphicsEffect(effect)


# ============================================================================
# AI WORKER - กล้อง + โมเดล YOLO + ตรวจจับการซ้อนทับของเม็ดยา
# ============================================================================
class MultiThreadedAIWorker(QThread):
    OVERLAP_CONFIRM_FRAMES = 3   # ต้องเจอ overlap ติดกันกี่เฟรมถึงจะยืนยันว่าเจอจริง
    OVERLAP_CLEAR_FRAMES = 5     # ต้องหาย overlap ติดกันกี่เฟรมถึงจะเคลียร์สถานะ
    CONCAVE_DEFECT_RATIO = 0.20   # รอยบุ๋มลึกกว่ากี่ % ของรัศมีถึงถือว่าเป็นรอยต่อ 2 เม็ด
    BLOB_AREA_MULTIPLIER = 1.6    # พื้นที่ blob ใหญ่กว่าเม็ดยาทั่วไปกี่เท่าถึงถือว่ามี 2 เม็ดรวมกัน

    def __init__(self, model_path="../openvino-model/best_openvino_model"):
        super().__init__()

        # ตัวแปรสำหรับควบคุมการเปิด/ปิดระบบแจ้งเตือน Overlap (ใช้ได้ทั้ง 2 โหมด)
        self.enable_overlap = True

        # ------------------------------------------------------------------
        # 1) เช็คก่อนว่ามีเว็บเซิร์ฟเวอร์ (เว็บไซต์นับเม็ดยา) รันอยู่ที่ localhost:5000
        #    หรือไม่ ถ้ามี -> เข้าสู่ "โหมดเชื่อมต่อเว็บไซต์" (Co-existence Mode):
        #    รับภาพจาก stream ของเว็บ และดึงผลนับ/overlap จาก API แทนการรันโมเดลเอง
        #    ถ้าไม่มี -> ทำงานแบบเดิมทุกอย่าง (โหลดโมเดล + เปิดกล้องเครื่องนี้เอง)
        # ------------------------------------------------------------------
        self.coexistence_mode = False
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:5000/api/status", timeout=1.0)
            self.coexistence_mode = True
            print("พบเว็บเซิร์ฟเวอร์ที่ localhost:5000 -> ทำงานในโหมดเชื่อมต่อเว็บไซต์ (Co-existence)")
        except Exception:
            print("ไม่พบเว็บเซิร์ฟเวอร์ที่ localhost:5000 -> ทำงานแบบเดี่ยว (Standalone)")

        # 2) โหลดโมเดล AI เฉพาะตอนทำงานแบบเดี่ยวเท่านั้น (โหมดเว็บไซต์ให้เซิร์ฟเวอร์
        #    เป็นคนรันโมเดลแทน จึงไม่ต้อง/ไม่ควรโหลดโมเดลซ้ำในเครื่องนี้)
        self.model = None
        self.model_ok = False
        if not self.coexistence_mode:
            if not os.path.exists(model_path):
                alt_path = "../pytouch-model/best.pt"
                if os.path.exists(alt_path):
                    model_path = alt_path
                else:
                    model_path = "best.pt"
            self.model_path = model_path
            try:
                print(f"Loading AI Model: {self.model_path}")
                self.model = YOLO(self.model_path)
                self.model_ok = True
            except Exception as e:
                print(f"[AI Worker] โหลดโมเดลไม่สำเร็จ: {e}")
        else:
            # โหมดเว็บไซต์: ไม่มีโมเดลในเครื่องนี้ แต่ถือว่า "พร้อม" เพราะเซิร์ฟเวอร์
            # เป็นคนตรวจจับให้แทน ไม่ควรบล็อกการเข้าหน้านับด้วยเหตุผลนี้
            self.model_ok = True

        # 3) เลือกแหล่งรับภาพตามโหมด
        if self.coexistence_mode:
            print("กำลังเชื่อมต่อ video stream จากเว็บเซิร์ฟเวอร์: http://localhost:5000/video_feed")
            self.cap = cv2.VideoCapture("http://localhost:5000/video_feed")
        else:
            print("กำลังเปิดกล้องของเครื่องนี้โดยตรง...")
            self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            self.cap.set(cv2.CAP_PROP_FOCUS, 85)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)

        self.camera_ok = self.cap.isOpened()

        self.running = True
        self.raw_frame = None
        self.boxes = []
        self.overlap_flags = []
        self.is_overlapping = False
        self.detect_count = 0
        self.stable_count = 0
        self.count_history = deque(maxlen=10)
        self.lock = threading.Lock()

        self._overlap_true_streak = 0
        self._overlap_false_streak = 0

        # Thread 1: Camera Frame Grabber
        self.grab_thread = threading.Thread(target=self._grab_frames, daemon=True)
        self.grab_thread.start()

        wait_start = time.time()
        while self.raw_frame is None and time.time() - wait_start < 3.0:
            time.sleep(0.01)

    def _grab_frames(self):
        """ดึงเฟรมภาพต่อเนื่อง รองรับทั้งกล้องเครื่องนี้และ HTTP stream จากเว็บไซต์
        พร้อม auto-reconnect ถ้าหลุดการเชื่อมต่อ (กันแอปค้างเมื่อกล้อง/เน็ตเวิร์กสะดุด)"""
        consecutive_failures = 0
        while self.running:
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.raw_frame = frame
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
            else:
                consecutive_failures += 1

            if consecutive_failures >= 15:
                warning_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(warning_frame, "กล้องขาดการเชื่อมต่อ", (110, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                cv2.putText(warning_frame, "กำลังเชื่อมต่อใหม่...", (150, 280),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                self.raw_frame = warning_frame

                if self.cap is not None:
                    try:
                        self.cap.release()
                    except Exception:
                        pass

                if self.coexistence_mode:
                    time.sleep(1.0)
                    self.cap.open("http://localhost:5000/video_feed")
                    if self.cap.isOpened():
                        consecutive_failures = 0
                else:
                    for index in [0, 2, 4, 1, 3]:
                        try:
                            cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
                            if cap.isOpened():
                                ret_test, test_frame = cap.read()
                                if ret_test and test_frame is not None:
                                    self.cap = cap
                                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                                    self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                                    self.cap.set(cv2.CAP_PROP_FOCUS, 85)
                                    consecutive_failures = 0
                                    print(f"เชื่อมต่อกล้อง index {index} สำเร็จอีกครั้ง")
                                    break
                                else:
                                    cap.release()
                        except Exception:
                            pass
                    time.sleep(1.0)
            else:
                time.sleep(0.005)

    # ------------------------------------------------------------------
    # Overlap detection helpers (global connected-components + convexity defects)
    # ------------------------------------------------------------------
    def _detect_overlaps(self, gray_frame, boxes):
        n = len(boxes)
        flags = [False] * n
        has_overlap = False
        if n == 0:
            return flags, False

        h, w = gray_frame.shape[:2]
        try:
            _, binary = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        except cv2.error:
            return flags, False

        num_labels, labels = cv2.connectedComponents(binary)

        # หา label ของ blob ที่จุดศูนย์กลางแต่ละกล่องตกอยู่
        box_labels = []
        for (x1, y1, x2, y2) in boxes:
            cx = int(min(max((x1 + x2) / 2, 0), w - 1))
            cy = int(min(max((y1 + y2) / 2, 0), h - 1))
            box_labels.append(int(labels[cy, cx]))

        label_to_indices = {}
        for i, lbl in enumerate(box_labels):
            if lbl == 0:  # label 0 = พื้นหลัง
                continue
            label_to_indices.setdefault(lbl, []).append(i)

        # อ้างอิงพื้นที่ "เม็ดยา 1 เม็ดทั่วไป" จาก blob ที่ map กับกล่องเดียวชัดเจน
        single_areas = [
            self._blob_area(labels, lbl)
            for lbl, idxs in label_to_indices.items() if len(idxs) == 1
        ]
        single_areas = [a for a in single_areas if a > 0]
        reference_area = float(np.median(single_areas)) if single_areas else None

        for lbl, idxs in label_to_indices.items():
            if len(idxs) != 1:
                continue  # มี 0 หรือ >=2 กล่องคลุม blob นี้แล้ว ไม่ใช่ความเสี่ยงที่ต้องเตือน
            if self._blob_has_concave_merge(labels, lbl, reference_area):
                flags[idxs[0]] = True
                has_overlap = True

        return flags, has_overlap

    @staticmethod
    def _blob_area(labels, label_id):
        return float(np.count_nonzero(labels == label_id))

    def _blob_has_concave_merge(self, labels, label_id, reference_area=None):
        mask = np.zeros(labels.shape, dtype=np.uint8)
        mask[labels == label_id] = 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if len(c) < 5 or area < 20:
            return False

        rect = cv2.minAreaRect(c)
        (_, _), (w_rect, h_rect), _ = rect
        if w_rect <= 0 or h_rect <= 0:
            return False
        long_side, short_side = max(w_rect, h_rect), min(w_rect, h_rect)
        aspect_ratio = long_side / short_side if short_side > 0 else 1.0

        if reference_area and reference_area > 0 and area > reference_area * self.BLOB_AREA_MULTIPLIER:
            return True

        hull_idx = cv2.convexHull(c, returnPoints=False)
        if hull_idx is None or len(hull_idx) < 3:
            return False
        try:
            hull_idx = np.sort(hull_idx, axis=0)
            defects = cv2.convexityDefects(c, hull_idx)
        except cv2.error:
            return False
        if defects is None:
            return False

        (_, _), r = cv2.minEnclosingCircle(c)
        if r <= 0:
            return False

        defects = np.asarray(defects).reshape(-1, 4)
        depths_px = defects[:, 3].astype(np.float64) / 256.0
        max_depth = depths_px.max() if len(depths_px) else 0.0

        ratio_threshold = self.CONCAVE_DEFECT_RATIO
        if aspect_ratio <= 1.5:
            ratio_threshold *= 1.15

        return bool(max_depth > r * ratio_threshold)

    def run(self):
        while self.running:
            if self.coexistence_mode:
                # -----------------------------------------------------------
                # โหมดเชื่อมต่อเว็บไซต์: ไม่รันโมเดลเอง แค่ดึงผลนับ/overlap ที่
                # เว็บเซิร์ฟเวอร์คำนวณไว้แล้วมาแสดงผล
                # -----------------------------------------------------------
                try:
                    import urllib.request
                    import json
                    response = urllib.request.urlopen(
                        "http://localhost:5000/api/counter/data", timeout=0.5)
                    data = json.loads(response.read().decode())
                    raw_count = int(data.get("raw_count", 0))
                    smoothed_count = int(data.get("stable_count", raw_count))

                    # ------------------------------------------------------
                    # sync สถานะ enable_overlap จากเซิร์ฟเวอร์ (แหล่งความจริง
                    # ตอนอยู่ใน coexistence mode) เข้ามาที่ client เงียบๆ โดย
                    # ไม่ push กลับ (กัน loop) - ครอบคลุมกรณีมีคนสั่งปิด/เปิด
                    # จากเครื่องอื่น หรือจาก /api/counter/toggle_overlap โดยตรง
                    # ------------------------------------------------------
                    server_overlap_enabled = bool(data.get("overlap_enabled", True))
                    if server_overlap_enabled != self.enable_overlap:
                        self.enable_overlap = server_overlap_enabled

                    # เคารพสวิตช์ "เปิด/ปิดระบบตรวจจับการซ้อนทับ" ฝั่งไคลเอนต์เสมอ
                    has_overlap_raw = bool(data.get("is_overlapping", False)) and self.enable_overlap
                except Exception:
                    raw_count = self.detect_count
                    smoothed_count = self.stable_count
                    has_overlap_raw = False

                confirmed_overlap = self._apply_overlap_hysteresis(has_overlap_raw)

                with self.lock:
                    self.detect_count = raw_count
                    self.stable_count = smoothed_count
                    self.is_overlapping = confirmed_overlap
                    self.overlap_flags = []
                    self.boxes = []

                time.sleep(0.1)

            elif self.raw_frame is not None and self.model_ok:
                try:
                    img_copy = self.raw_frame.copy()

                    gray_frame = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
                    img_input = cv2.merge([gray_frame, gray_frame, gray_frame])

                    results = self.model(img_input, conf=0.45, iou=0.5, device="cpu", verbose=False)
                    result = results[0]

                    filtered_boxes = [
                        (float(x1), float(y1), float(x2), float(y2))
                        for (x1, y1, x2, y2) in
                        (box.xyxy[0].cpu().numpy() for box in result.boxes)
                    ]

                    # เช็คก่อนว่าฟีเจอร์เปิดอยู่หรือไม่
                    if self.enable_overlap:
                        overlap_flags, has_overlap_raw = self._detect_overlaps(gray_frame, filtered_boxes)
                    else:
                        overlap_flags = [False] * len(filtered_boxes)
                        has_overlap_raw = False

                    confirmed_overlap = self._apply_overlap_hysteresis(has_overlap_raw)

                    raw_count = len(filtered_boxes)
                    self.count_history.append(raw_count)
                    smoothed_count = max(set(self.count_history), key=self.count_history.count) if self.count_history else 0

                    with self.lock:
                        self.detect_count = raw_count
                        self.stable_count = smoothed_count
                        self.is_overlapping = confirmed_overlap
                        self.overlap_flags = overlap_flags
                        self.boxes = filtered_boxes
                except Exception as e:
                    print(f"[AI Worker] ข้ามเฟรมนี้เนื่องจากเกิดข้อผิดพลาด: {e}")

                time.sleep(0.01)
            else:
                time.sleep(0.01)

    def _apply_overlap_hysteresis(self, has_overlap_raw):
        """Hysteresis เดียวกัน ใช้ร่วมกันได้ทั้งโหมด standalone และโหมดเว็บไซต์
        กันสถานะ overlap กระพริบเฟรมต่อเฟรม/รอบ poll ต่อรอบ poll"""
        if has_overlap_raw:
            self._overlap_true_streak += 1
            self._overlap_false_streak = 0
        else:
            self._overlap_false_streak += 1
            self._overlap_true_streak = 0

        if self._overlap_true_streak >= self.OVERLAP_CONFIRM_FRAMES:
            return True
        elif self._overlap_false_streak >= self.OVERLAP_CLEAR_FRAMES:
            return False
        return self.is_overlapping

    def set_overlap_enabled(self, enable: bool):
        """สั่งเปิด/ปิดระบบตรวจจับ overlap

        - โหมด standalone: แก้ค่า self.enable_overlap ตรงๆ พอ เพราะ ui.py เป็น
          คนรันโมเดล+ตรวจ overlap เองอยู่แล้ว
        - โหมด coexistence: ต้องยิง POST ไปบอกเซิร์ฟเวอร์ (app.py) ด้วย ไม่งั้น
          แค่ตัวแปรฝั่ง client เปลี่ยน แต่เซิร์ฟเวอร์ยังรัน _detect_overlaps()
          (connectedComponents + convexHull + convexityDefects) ทุกเฟรมเหมือน
          เดิม ซึ่งเป็นส่วนที่กิน CPU มากสุดใน pipeline - เสียเปล่าบน Raspberry
          Pi 5 ถ้าผู้ใช้ปิดฟีเจอร์นี้ไว้แล้วเซิร์ฟเวอร์ยังคำนวณอยู่
        """
        self.enable_overlap = enable

        if self.coexistence_mode:
            def _push():
                try:
                    import urllib.request
                    import json
                    payload = json.dumps({"enable": enable}).encode("utf-8")
                    req = urllib.request.Request(
                        "http://localhost:5000/api/counter/toggle_overlap",
                        data=payload,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    urllib.request.urlopen(req, timeout=1.0)
                except Exception as e:
                    print(f"[AI Worker] แจ้งเซิร์ฟเวอร์เรื่องปิด/เปิด overlap ไม่สำเร็จ: {e}")
            threading.Thread(target=_push, daemon=True).start()

    def reset_counting(self):
        with self.lock:
            self.count_history.clear()
            self.stable_count = 0
            self.detect_count = 0
            self.is_overlapping = False
            self.overlap_flags = []
            self._overlap_true_streak = 0
            self._overlap_false_streak = 0

    def stop(self):
        self.running = False
        if self.grab_thread.is_alive():
            self.grab_thread.join()
        self.wait()
        self.cap.release()


class DesktopAppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MediCount - Automatic Pill Counter")
        self.setGeometry(50, 50, 1200, 760)
        

        self.vibrator = OutputDevice(6,pin_factory=factory) if GPIO_AVAILABLE else None
        self.motor_connected = GPIO_AVAILABLE

        self.target_count = 0
        self.keypad_input_buffer = ""

        self._build_global_style()

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.page1_setup = self.build_page1_setup()
        self.page2_counter = self.build_page2_counter()

        self.stacked_widget.addWidget(self.page1_setup)
        self.stacked_widget.addWidget(self.page2_counter)

        self.ai_worker = MultiThreadedAIWorker()
        self.ai_worker.start()

        self.ui_timer = QTimer()
        self.ui_timer.setInterval(30)
        self.ui_timer.timeout.connect(self.render_ui_loop)
        self.ui_timer.start()

        self.update_page1_status()

    # ========================================================================
    # GLOBAL STYLE
    # ========================================================================
    def _build_global_style(self):
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {Palette.BG}; }}
            QWidget {{ font-family: {FONT_FAMILY}; }}
            QLabel {{ color: {Palette.TEXT_DARK}; }}

            QFrame#card {{
                background-color: {Palette.CARD};
                border: 1px solid {Palette.BORDER};
                border-radius: 16px;
            }}
            QFrame#headerBar {{
                background-color: {Palette.CARD};
                border: none;
                border-bottom: 1px solid {Palette.BORDER};
                border-radius: 0px;
            }}

            QPushButton#btnPrimary {{
                background-color: {Palette.PRIMARY}; color: white;
                border-radius: 14px; padding: 16px; font-weight: 700; font-size: 16px; border: none;
            }}
            QPushButton#btnPrimary:hover {{ background-color: {Palette.PRIMARY_DARK}; }}
            QPushButton#btnPrimary:disabled {{ background-color: #A9C4C7; }}

            QPushButton#btnKey {{
                background-color: {Palette.NEUTRAL_BG}; color: {Palette.TEXT_DARK};
                border: 1px solid {Palette.BORDER}; border-radius: 14px;
                font-size: 20px; font-weight: 600; padding: 14px;
            }}
            QPushButton#btnKey:hover {{ background-color: {Palette.PRIMARY_SOFT}; border-color: {Palette.PRIMARY}; }}
            QPushButton#btnKeyClear {{
                background-color: {Palette.DANGER_BG}; color: {Palette.DANGER};
                border: 1px solid {Palette.DANGER}; border-radius: 14px;
                font-size: 16px; font-weight: 700; padding: 14px;
            }}
            QPushButton#btnKeyClear:hover {{ background-color: #F7D8DC; }}

            QPushButton#btnVibe {{
                background-color: {Palette.PRIMARY}; color: white; border: none;
                border-radius: 14px; font-size: 15px; font-weight: 700; padding: 18px;
            }}
            QPushButton#btnVibe:hover {{ background-color: {Palette.PRIMARY_DARK}; }}
            QPushButton#btnVibeAlert {{
                background-color: {Palette.DANGER}; color: white; border: none;
                border-radius: 14px; font-size: 15px; font-weight: 700; padding: 18px;
            }}
            QPushButton#btnVibeAlert:hover {{ background-color: #B93A4C; }}
            QPushButton#btnVibe:disabled, QPushButton#btnVibeAlert:disabled {{
                background-color: #C9D3D6; color: #7C8A90;
            }}

            QPushButton#btnReset {{
                background-color: {Palette.CARD}; color: {Palette.PRIMARY};
                border: 2px solid {Palette.PRIMARY}; border-radius: 14px;
                font-size: 15px; font-weight: 700; padding: 16px;
            }}
            QPushButton#btnReset:hover {{ background-color: {Palette.PRIMARY_SOFT}; }}

            QPushButton#btnToggle {{
                background-color: {Palette.PRIMARY_SOFT}; color: {Palette.PRIMARY_DARK};
                border: 1px solid {Palette.PRIMARY}; border-radius: 14px;
                font-size: 12px; font-weight: 700; padding: 4px 12px;
            }}
            QPushButton#btnToggle:hover {{ border-color: {Palette.PRIMARY_DARK}; }}
            QPushButton#btnToggle:checked {{
                background-color: {Palette.NEUTRAL_BG}; color: {Palette.TEXT_MUTED};
                border: 1px solid {Palette.BORDER};
            }}
            QPushButton#btnToggle:checked:hover {{ border-color: {Palette.TEXT_MUTED}; }}

            QLineEdit#targetDisplay {{
                background-color: {Palette.NEUTRAL_BG}; color: {Palette.PRIMARY_DARK};
                border: 2px solid {Palette.PRIMARY}; border-radius: 16px;
                font-size: 40px; font-weight: 800; padding: 12px;
            }}

            QLabel#eyebrow {{
                color: {Palette.TEXT_MUTED}; font-size: 12px; font-weight: 700;
                letter-spacing: 1px;
            }}
            QLabel#appTitle {{ color: {Palette.PRIMARY_DARK}; font-size: 21px; font-weight: 800; }}
            QLabel#appSubtitle {{ color: {Palette.TEXT_MUTED}; font-size: 12px; }}
        """)

    # ========================================================================
    # SHARED HEADER BAR
    # ========================================================================
    def _make_header(self, title_text, subtitle_text, right_widget=None):
        bar = QFrame()
        bar.setObjectName("headerBar")
        bar.setFixedHeight(78)
        row = QHBoxLayout(bar)
        row.setContentsMargins(28, 12, 28, 12)

        logo = QLabel("\u271A")  # ✚
        logo.setFixedSize(44, 44)
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setStyleSheet(f"""
            background-color: {Palette.PRIMARY}; color: white;
            border-radius: 22px; font-size: 20px; font-weight: 700;
        """)
        row.addWidget(logo)

        text_col = QVBoxLayout()
        text_col.setSpacing(0)
        t1 = QLabel(title_text)
        t1.setObjectName("appTitle")
        t2 = QLabel(subtitle_text)
        t2.setObjectName("appSubtitle")
        text_col.addWidget(t1)
        text_col.addWidget(t2)
        row.addLayout(text_col)

        row.addStretch()
        if right_widget is not None:
            row.addWidget(right_widget)

        return bar

    def _make_card(self, min_height=None):
        card = QFrame()
        card.setObjectName("card")
        if min_height:
            card.setMinimumHeight(min_height)
        apply_shadow(card)
        return card

    def _make_eyebrow(self, text):
        lbl = QLabel(text.upper())
        lbl.setObjectName("eyebrow")
        return lbl

    # ========================================================================
    # PAGE 1 : สถานะอุปกรณ์ + คีย์แพดกรอกจำนวนยา + ปุ่มยืนยัน
    # ========================================================================
    def build_page1_setup(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.lbl_mode_p1 = QLabel("กำลังตรวจสอบโหมด...")
        self.lbl_mode_p1.setStyleSheet(f"""
            color: {Palette.PRIMARY_DARK}; background-color: {Palette.PRIMARY_SOFT};
            border-radius: 14px; padding: 8px 16px; font-weight: 700; font-size: 13px;
        """)
        outer.addWidget(self._make_header("MediCount", "ระบบนับเม็ดยาอัตโนมัติ · ตั้งค่าก่อนเริ่มนับ",
                                            right_widget=self.lbl_mode_p1))

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(36, 28, 36, 32)
        body_layout.setSpacing(22)
        outer.addWidget(body, stretch=1)

        body_layout.addWidget(self._make_eyebrow("สถานะอุปกรณ์ (Device Status)"))

        status_row = QHBoxLayout()
        status_row.setSpacing(18)

        self.card_cam, self.cam_dot, self.cam_badge, self.cam_sub = self._build_status_card("กล้อง", "Camera")
        self.card_model, self.model_dot, self.model_badge, self.model_sub = self._build_status_card("โมเดล AI", "Model")
        self.card_motor, self.motor_dot, self.motor_badge, self.motor_sub = self._build_status_card(
            "มอเตอร์สั่น", "Vibration Motor")

        status_row.addWidget(self.card_cam)
        status_row.addWidget(self.card_model)
        status_row.addWidget(self.card_motor)
        body_layout.addLayout(status_row)

        body_layout.addWidget(self._make_eyebrow("จำนวนยาที่ต้องการ (Target Quantity)"))

        target_card = self._make_card()
        target_row = QHBoxLayout(target_card)
        target_row.setContentsMargins(28, 24, 28, 24)
        target_row.setSpacing(36)

        display_col = QVBoxLayout()
        display_col.setSpacing(10)
        display_col.addStretch()

        self.target_display = QLineEdit("0")
        self.target_display.setObjectName("targetDisplay")
        self.target_display.setReadOnly(True)
        self.target_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.target_display.setFixedHeight(96)
        display_col.addWidget(self.target_display)

        unit_lbl = QLabel("เม็ด (Tablets)")
        unit_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        unit_lbl.setStyleSheet(f"color:{Palette.TEXT_MUTED}; font-size: 13px;")
        display_col.addWidget(unit_lbl)

        display_col.addStretch()
        target_row.addLayout(display_col, stretch=2)

        grid = QGridLayout()
        grid.setSpacing(10)
        keys = [
            ("7", 0, 0), ("8", 0, 1), ("9", 0, 2),
            ("4", 1, 0), ("5", 1, 1), ("6", 1, 2),
            ("1", 2, 0), ("2", 2, 1), ("3", 2, 2),
            ("0", 3, 1), ("X", 3, 2),
        ]
        for label, r, c in keys:
            btn = QPushButton(label)
            btn.setFixedSize(76, 58)
            if label == "X":
                btn.setObjectName("btnKeyClear")
                btn.clicked.connect(self.keypad_clear)
            else:
                btn.setObjectName("btnKey")
                btn.clicked.connect(lambda checked, d=label: self.keypad_press(d))
            grid.addWidget(btn, r, c)
        target_row.addLayout(grid, stretch=3)

        body_layout.addWidget(target_card, stretch=1)

        self.btn_confirm = QPushButton("ยืนยัน และเริ่มนับ  \u2192")
        self.btn_confirm.setObjectName("btnPrimary")
        self.btn_confirm.setFixedHeight(58)
        self.btn_confirm.clicked.connect(self.confirm_and_go_to_page2)
        body_layout.addWidget(self.btn_confirm)

        return page

    def _build_status_card(self, label_th, label_en):
        card = self._make_card(min_height=118)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(6)

        top_row = QHBoxLayout()
        dot = QLabel("\u25CF")
        dot.setStyleSheet(f"font-size: 15px; color: {Palette.TEXT_MUTED};")
        top_row.addWidget(dot)
        name_lbl = QLabel(f"{label_th}")
        name_lbl.setStyleSheet("font-weight: 700; font-size: 14px;")
        top_row.addWidget(name_lbl)
        top_row.addStretch()
        layout.addLayout(top_row)

        en_lbl = QLabel(label_en)
        en_lbl.setStyleSheet(f"color:{Palette.TEXT_MUTED}; font-size: 11px;")
        layout.addWidget(en_lbl)

        badge = QLabel("...")
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setFixedHeight(30)
        layout.addWidget(badge)
        layout.addStretch()

        return card, dot, badge, en_lbl

    def keypad_press(self, digit):
        if len(self.keypad_input_buffer) >= 4:
            return
        self.keypad_input_buffer += digit
        cleaned = str(int(self.keypad_input_buffer)) if self.keypad_input_buffer else "0"
        self.target_display.setText(cleaned)

    def keypad_clear(self):
        self.keypad_input_buffer = ""
        self.target_display.setText("0")

    def update_page1_status(self):
        coexistence = getattr(self.ai_worker, "coexistence_mode", False) if hasattr(self, "ai_worker") else False

        if hasattr(self, "lbl_mode_p1"):
            self.lbl_mode_p1.setText(
                "\U0001F310  โหมดเชื่อมต่อเว็บไซต์" if coexistence else "\U0001F4BB  โหมดทำงานเดี่ยว"
            )

        cam_ok = self._camera_ready()
        model_ok = getattr(self.ai_worker, "model_ok", False) if hasattr(self, "ai_worker") else False

        if coexistence:
            self.cam_sub.setText("Video Stream (Web Server)")
            self.model_sub.setText("ประมวลผลบน Web Server")
            cam_text = "เชื่อมต่อแล้ว (ผ่านเว็บไซต์)" if cam_ok else "เชื่อมต่อเว็บไซต์ไม่สำเร็จ"
            model_text = "พร้อมใช้งาน (เว็บเซิร์ฟเวอร์ประมวลผลให้)"
        else:
            self.cam_sub.setText("Camera")
            self.model_sub.setText("Model")
            cam_text = "เชื่อมต่อแล้ว" if cam_ok else "ไม่พบอุปกรณ์"
            model_text = "เชื่อมต่อแล้ว" if model_ok else "โหลดไม่สำเร็จ"

        self._set_status_badge(self.cam_dot, self.cam_badge, cam_ok, cam_text)
        self._set_status_badge(self.model_dot, self.model_badge, model_ok, model_text)
        self._set_status_badge(self.motor_dot, self.motor_badge, self.motor_connected,
                                "เชื่อมต่อแล้ว" if self.motor_connected else "ไม่พบอุปกรณ์")

    def _set_status_badge(self, dot_label, badge_label, ok, text):
        color = Palette.SUCCESS if ok else Palette.DANGER
        bg = Palette.SUCCESS_BG if ok else Palette.DANGER_BG
        dot_label.setStyleSheet(f"font-size: 15px; color: {color};")
        badge_label.setText(text)
        badge_label.setStyleSheet(f"""
            background-color: {bg}; color: {color};
            border-radius: 15px; font-weight: 700; font-size: 12px;
        """)

    def _camera_ready(self):
        if not hasattr(self, "ai_worker"):
            return False
        try:
            return bool(self.ai_worker.camera_ok and self.ai_worker.cap.isOpened())
        except Exception:
            return False

    def _check_hardware_ready(self):
        missing = []
        if not self._camera_ready():
            missing.append("กล้อง (Camera)")
        if not getattr(self.ai_worker, "model_ok", False):
            missing.append("โมเดล AI (Model)")
        return missing

    def confirm_and_go_to_page2(self):
        try:
            value = int(self.target_display.text())
        except ValueError:
            value = 0

        if value <= 0:
            QMessageBox.warning(self, "กรอกจำนวนยา", "กรุณากรอกจำนวนยาที่ต้องการนับ (มากกว่า 0) ก่อนกดยืนยัน")
            return

        self.update_page1_status()
        missing = self._check_hardware_ready()
        if missing:
            QMessageBox.critical(
                self, "ไม่พร้อมใช้งาน",
                "ไม่สามารถเริ่มนับได้ เนื่องจากอุปกรณ์ต่อไปนี้ไม่พร้อม:\n\n"
                + "\n".join(f"•  {m}" for m in missing)
                + "\n\nกรุณาตรวจสอบการเชื่อมต่อแล้วลองใหม่อีกครั้ง"
            )
            return

        self.target_count = value
        self.lbl_target_goal_p2.setText(f"{self.target_count} เม็ด")
        self.ai_worker.reset_counting()
        self.stacked_widget.setCurrentIndex(1)

    # ========================================================================
    # PAGE 2 : กล้อง realtime + overlap warning + สถานะยา + ปุ่มสั่น + ปุ่มรีเซ็ต
    # ========================================================================
    def build_page2_counter(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.lbl_target_goal_p2 = QLabel("0 เม็ด")
        self.lbl_target_goal_p2.setStyleSheet(f"""
            color: {Palette.PRIMARY_DARK}; background-color: {Palette.PRIMARY_SOFT};
            border-radius: 14px; padding: 8px 16px; font-weight: 700; font-size: 13px;
        """)
        outer.addWidget(self._make_header("MediCount", "กำลังนับเม็ดยาแบบเรียลไทม์",
                                            right_widget=self.lbl_target_goal_p2))

        body = QWidget()
        content = QHBoxLayout(body)
        content.setContentsMargins(28, 24, 28, 28)
        content.setSpacing(22)
        outer.addWidget(body, stretch=1)

        video_card = self._make_card()
        video_layout = QVBoxLayout(video_card)
        video_layout.setContentsMargins(12, 12, 12, 12)
        self.video_view = QLabel("กำลังเปิดกล้อง...")
        self.video_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_view.setStyleSheet("""
            background-color: #0B1E24; border-radius: 12px; color: #9FC4C9;
        """)
        self.video_view.setMinimumSize(640, 480)
        video_layout.addWidget(self.video_view)
        content.addWidget(video_card, stretch=3)

        right_col = QVBoxLayout()
        right_col.setSpacing(16)

        self.overlap_box = self._make_card(min_height=150)
        ov_layout = QVBoxLayout(self.overlap_box)
        ov_layout.setContentsMargins(22, 16, 22, 18)
        ov_layout.setSpacing(8)

        ov_header_row = QHBoxLayout()
        ov_header_row.addWidget(self._make_eyebrow("การตรวจจับการซ้อนทับ"))
        ov_header_row.addStretch()
        self.btn_toggle_ov = QPushButton("\u25CF เปิดใช้งาน")
        self.btn_toggle_ov.setObjectName("btnToggle")
        self.btn_toggle_ov.setCheckable(True)
        self.btn_toggle_ov.setFixedHeight(28)
        self.btn_toggle_ov.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_toggle_ov.clicked.connect(self.toggle_overlap)
        ov_header_row.addWidget(self.btn_toggle_ov)
        ov_layout.addLayout(ov_header_row)

        self.overlap_title = QLabel("\u2713  ไม่มีการซ้อนทับ")
        self.overlap_title.setStyleSheet(f"color:{Palette.SUCCESS}; font-size: 17px; font-weight: 800;")
        self.overlap_title.setWordWrap(True)
        ov_layout.addWidget(self.overlap_title)
        self.overlap_desc = QLabel("เม็ดยาแยกจากกันชัดเจน สามารถนับต่อได้ตามปกติ")
        self.overlap_desc.setWordWrap(True)
        self.overlap_desc.setStyleSheet(f"color:{Palette.TEXT_MUTED}; font-size: 12px;")
        ov_layout.addWidget(self.overlap_desc)
        ov_layout.addStretch()
        right_col.addWidget(self.overlap_box)

        self.btn_vibe = QPushButton("\U0001F514  สั่นถาด (VIBRATION)")
        self.btn_vibe.setObjectName("btnVibe")
        self.btn_vibe.setFixedHeight(64)
        self.btn_vibe.clicked.connect(self.trigger_vibe)
        right_col.addWidget(self.btn_vibe)
        self._apply_motor_availability()

        self.status_card = self._make_card(min_height=150)
        st_layout = QVBoxLayout(self.status_card)
        st_layout.setContentsMargins(22, 16, 22, 16)
        st_layout.addWidget(self._make_eyebrow("สถานะจำนวนยา"))
        self.lbl_status = QLabel("ขาด")
        self.lbl_status.setFont(QFont(FONT_FAMILY, 30, QFont.Weight.Bold))
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        st_layout.addWidget(self.lbl_status)
        self.lbl_status_detail = QLabel("ตรวจพบ 0 / เป้าหมาย 0 เม็ด")
        self.lbl_status_detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status_detail.setStyleSheet(f"color:{Palette.TEXT_MUTED}; font-size: 12px;")
        st_layout.addWidget(self.lbl_status_detail)
        right_col.addWidget(self.status_card)

        right_col.addStretch()

        self.btn_reset = QPushButton("\u21BB  นับใหม่ / กลับไปตั้งค่า")
        self.btn_reset.setObjectName("btnReset")
        self.btn_reset.setFixedHeight(54)
        self.btn_reset.clicked.connect(self.reset_and_go_to_page1)
        right_col.addWidget(self.btn_reset)

        right_panel = QWidget()
        right_panel.setLayout(right_col)
        right_panel.setFixedWidth(360)
        content.addWidget(right_panel, stretch=0)

        return page

    def toggle_overlap(self, checked):
        # เรียกผ่าน set_overlap_enabled() แทนการเซ็ต attribute ตรงๆ เพื่อให้ยิง
        # แจ้งเซิร์ฟเวอร์ด้วยเมื่ออยู่ใน coexistence mode (ประหยัด CPU ฝั่ง Pi
        # เพราะเซิร์ฟเวอร์จะหยุดคำนวณ overlap จริงๆ ไม่ใช่แค่ซ่อนผลฝั่ง client)
        self.ai_worker.set_overlap_enabled(not checked)
        if checked:
            self.btn_toggle_ov.setText("\u25CB ปิดใช้งาน")
        else:
            self.btn_toggle_ov.setText("\u25CF เปิดใช้งาน")

    def reset_and_go_to_page1(self):
        self.ai_worker.reset_counting()
        self.stacked_widget.setCurrentIndex(0)
        self.update_page1_status()

    def _apply_motor_availability(self):
        self.btn_vibe.setEnabled(self.motor_connected)
        if self.motor_connected:
            self.btn_vibe.setToolTip("")
        else:
            self.btn_vibe.setToolTip("ไม่พบมอเตอร์สั่น (Vibration Motor) - ปุ่มนี้ถูกปิดใช้งาน")

    def trigger_vibe(self):
        if not self.motor_connected:
            return 
        if self.vibrator:
            def vibe():
                try:
                    for _ in range(3):
                        self.vibrator.on()
                        time.sleep(0.2)
                        self.vibrator.off()
                        time.sleep(0.2)
                except Exception:
                    pass
            threading.Thread(target=vibe, daemon=True).start()
        else:
            print("[Vibration] (จำลอง) ผู้ใช้กดปุ่มสั่นถาด")

    # ========================================================================
    # RENDER LOOP
    # ========================================================================
    def render_ui_loop(self):
        if self.ai_worker.raw_frame is None:
            return

        frame = self.ai_worker.raw_frame.copy()

        with self.ai_worker.lock:
            coexistence = self.ai_worker.coexistence_mode
            boxes = list(self.ai_worker.boxes)
            overlap_flags = list(self.ai_worker.overlap_flags)
            is_overlapping = self.ai_worker.is_overlapping
            stable_count = self.ai_worker.stable_count

        # ------------------------------------------------------------------
        # sync ปุ่ม toggle ให้ตรงกับสถานะจริงของ ai_worker.enable_overlap เสมอ
        # (สำคัญในโหมด coexistence เพราะค่าอาจถูกเปลี่ยนจากฝั่งเซิร์ฟเวอร์ /
        # เครื่องอื่นได้ ไม่ใช่แค่จากปุ่มนี้เพียงอย่างเดียว) เช็คแบบเบาๆ ทุกเฟรม
        # UI (30ms) ไม่กระทบ performance
        # ------------------------------------------------------------------
        worker_enabled = self.ai_worker.enable_overlap
        btn_checked = self.btn_toggle_ov.isChecked()  # checked = ปิดอยู่ (ตาม logic เดิม)
        if worker_enabled == btn_checked:
            self.btn_toggle_ov.blockSignals(True)
            self.btn_toggle_ov.setChecked(not worker_enabled)
            self.btn_toggle_ov.setText("\u25CF เปิดใช้งาน" if worker_enabled else "\u25CB ปิดใช้งาน")
            self.btn_toggle_ov.blockSignals(False)

        # โหมดเดี่ยว: วาด dot/label overlay เอง | โหมดเว็บไซต์: เว็บเซิร์ฟเวอร์วาด
        # ทับในสตรีมให้แล้ว ไม่ต้องวาดซ้ำฝั่งนี้อีก
        if not coexistence:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                is_ov = (len(overlap_flags) > i and overlap_flags[i])
                color = (74, 69, 214) if is_ov else (134, 158, 30)  # BGR

                cv2.circle(frame, (cx, cy), 5, color, -1)
                cv2.circle(frame, (cx, cy), 7, (255, 255, 255), 1)

                if is_ov:
                    cv2.putText(frame, "Overlap!", (cx + 10, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.putText(frame, f"Pills Detected: {stable_count}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                        (74, 69, 214) if is_overlapping else (134, 124, 14), 3)

        if self.stacked_widget.currentIndex() == 1:
            self._update_overlap_box(is_overlapping)
            self._update_status_label(stable_count)

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(q_img)
            scaled_p = pix.scaled(self.video_view.width(), self.video_view.height(),
                                   Qt.AspectRatioMode.KeepAspectRatio)
            self.video_view.setPixmap(scaled_p)

    def _update_overlap_box(self, is_overlapping):
        if not self.ai_worker.enable_overlap:
            self.overlap_box.setStyleSheet(f"""
                QFrame#card {{ background-color: {Palette.NEUTRAL_BG};
                border: 1px dashed {Palette.BORDER}; border-radius: 16px; }}
            """)
            self.overlap_title.setText("\u26D4 ปิดระบบตรวจจับแล้ว")
            self.overlap_title.setStyleSheet(f"color:{Palette.TEXT_MUTED}; font-size: 16px; font-weight: 800;")
            self.overlap_desc.setText("ระบบจะไม่แจ้งเตือนเมื่อเม็ดยาซ้อนทับกัน")
            self.btn_vibe.setObjectName("btnVibe")
            self.btn_vibe.setText("\U0001F514  สั่นถาด (VIBRATION)"
                                   if self.motor_connected else "\U0001F514  ไม่พบมอเตอร์สั่น")
        elif is_overlapping:
            self.overlap_box.setStyleSheet(f"""
                QFrame#card {{ background-color: {Palette.DANGER_BG};
                border: 1.5px solid {Palette.DANGER}; border-radius: 16px; }}
            """)
            self.overlap_title.setText("\u26A0  แจ้งเตือน: เม็ดยาซ้อนทับ")
            self.overlap_title.setStyleSheet(f"color:{Palette.DANGER}; font-size: 17px; font-weight: 800;")
            if self.motor_connected:
                self.overlap_desc.setText("ตรวจพบเม็ดยาซ้อนทับกัน กรุณากดปุ่มสั่นถาดด้านล่าง เพื่อให้เม็ดยาแยกออกจากกัน")
            else:
                self.overlap_desc.setText("ตรวจพบเม็ดยาซ้อนทับกัน แต่ไม่พบมอเตอร์สั่น กรุณาแยกเม็ดยาด้วยตนเอง")
            self.btn_vibe.setObjectName("btnVibeAlert" if self.motor_connected else "btnVibe")
            self.btn_vibe.setText("\u26A0  กดสั่นถาดตอนนี้" if self.motor_connected
                                   else "\U0001F514  ไม่พบมอเตอร์สั่น")
        else:
            self.overlap_box.setStyleSheet(f"""
                QFrame#card {{ background-color: {Palette.CARD};
                border: 1px solid {Palette.BORDER}; border-radius: 16px; }}
            """)
            self.overlap_title.setText("\u2713  ไม่มีการซ้อนทับ")
            self.overlap_title.setStyleSheet(f"color:{Palette.SUCCESS}; font-size: 17px; font-weight: 800;")
            self.overlap_desc.setText("เม็ดยาแยกจากกันชัดเจน สามารถนับต่อได้ตามปกติ")
            self.btn_vibe.setObjectName("btnVibe")
            self.btn_vibe.setText("\U0001F514  สั่นถาด (VIBRATION)"
                                   if self.motor_connected else "\U0001F514  ไม่พบมอเตอร์สั่น")
            
        self.btn_vibe.setEnabled(self.motor_connected)
        self.btn_vibe.style().unpolish(self.btn_vibe)
        self.btn_vibe.style().polish(self.btn_vibe)

    def _update_status_label(self, stable_count):
        target = self.target_count
        self.lbl_status_detail.setText(f"ตรวจพบ {stable_count} / เป้าหมาย {target} เม็ด")

        if target <= 0:
            self.lbl_status.setText("-")
            self.lbl_status.setStyleSheet(f"color:{Palette.TEXT_MUTED};")
            return

        diff = stable_count - target
        
        font_style = QFont(FONT_FAMILY, 24, QFont.Weight.Bold)
        self.lbl_status.setFont(font_style)

        if diff == 0:
            self.lbl_status.setText("ครบพอดี")
            self.lbl_status.setStyleSheet(f"color:{Palette.SUCCESS};")
        elif diff > 0:
            self.lbl_status.setText(f"เกิน {diff} เม็ด")
            self.lbl_status.setStyleSheet(f"color:{Palette.DANGER};")
        else:
            self.lbl_status.setText(f"ขาด {abs(diff)} เม็ด")
            self.lbl_status.setStyleSheet(f"color:{Palette.WARNING};")
          

    def closeEvent(self, event):
        self.ui_timer.stop()
        self.ai_worker.stop()
        if self.vibrator:
            self.vibrator.off()
        event.accept()
   




if __name__ == "__main__":
    app = QApplication(sys.argv)    
    window = DesktopAppWindow()
    window.showFullScreen()
    sys.exit(app.exec())