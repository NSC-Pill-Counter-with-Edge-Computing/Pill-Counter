import cv2
import os
import sys
import time
import socket
import threading
import subprocess
import numpy as np
from collections import deque
from flask import Flask, render_template, Response, jsonify, request, redirect
from ultralytics import YOLO

# GPIO 4 Vibration Support
try:
    from gpiozero import OutputDevice
    from gpiozero.pins.lgpio import LGPIOFactory
    factory = LGPIOFactory(chip=0)
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False


app = Flask(__name__, static_folder='static')


class ThreadedPillDetectorEngine:
    """
    Multi-threaded Camera & AI Engine matching main.py/ui.py logic:
    Uses Connected Components + Convexity Defects to accurately detect overlapping pills.

    หมายเหตุ (sync กับ ui.py):
    - ระบบสั่น (vibration) เป็นแบบ manual เท่านั้น สั่งผ่าน trigger_vibe_pulse()
      (เรียกจาก /api/trigger_shake) เหมือนปุ่ม "สั่นถาด" ใน ui.py ไม่มีการสั่ง
      สั่นอัตโนมัติจาก inference loop อีกต่อไป (ของเดิมสั่นอัตโนมัติทุกครั้งที่
      นับเพิ่ม/เจอ overlap ซึ่งไม่ตรงกับพฤติกรรมฝั่ง desktop app)
    - enable_overlap ปรับได้ผ่าน /api/counter/toggle_overlap ให้ตรงกับปุ่ม toggle
      ใน ui.py
    - reset_counting() ให้ผลเหมือนปุ่ม "นับใหม่" ใน ui.py
    """
    OVERLAP_CONFIRM_FRAMES = 3   # ต้องเจอ overlap ติดกันกี่เฟรมถึงจะยืนยันว่าเจอจริง
    OVERLAP_CLEAR_FRAMES = 5     # ต้องหาย overlap ติดกันกี่เฟรมถึงจะเคลียร์สถานะ
    CONCAVE_DEFECT_RATIO = 0.20  # รอยบุ๋มลึกกว่ากี่ % ของรัศมีถึงถือว่าเป็นรอยต่อ 2 เม็ด
    BLOB_AREA_MULTIPLIER = 1.6   # พื้นที่ blob ใหญ่กว่าเม็ดยาทั่วไปกี่เท่าถึงถือว่ามี 2 เม็ดรวมกัน

    def __init__(self):
        # Explicitly load from user-requested path
        self.model_path = "/home/pill/Downloads/Full Application/web-version/best.pt"
        if not os.path.exists(self.model_path):
            # Fallback dynamic checks if missing
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.model_path = os.path.join(script_dir, "../openvino-model/best_openvino_model")
            if not os.path.exists(self.model_path):
                self.model_path = os.path.join(script_dir, "../pytouch-model/best.pt")
                if not os.path.exists(self.model_path):
                    self.model_path = os.path.join(script_dir, "best.pt")

        print(f"Loading AI Model: {self.model_path}")
        self.model = YOLO(self.model_path)

        # Robust camera discovery loop (handles slow USB power up on boot)
        self.cap = None
        for attempt in range(15):
            for index in [0, 2, 4, 1, 3]:
                try:
                    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
                    if cap.isOpened():
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            self.cap = cap
                            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                            self.cap.set(cv2.CAP_PROP_FOCUS, 85)
                            print(f"Successfully opened camera on index {index} via V4L2")
                            break
                        else:
                            cap.release()
                except Exception:
                    pass
            if self.cap is not None:
                break
            print("Webcam not ready yet, retrying in 1 second...")
            time.sleep(1)

        if self.cap is None or not self.cap.isOpened():
            print("Fallback: Attempting default VideoCapture(0)")
            self.cap = cv2.VideoCapture(0)

        self.vibrator = OutputDevice(6,pin_factory=factory) if GPIO_AVAILABLE else None
        self.motor_connected = GPIO_AVAILABLE

        self.running = True
        self.raw_frame = None

        self.detect_count = 0
        self.stable_count = 0
        self.target_count = 20
        self.is_overlapping = False
        self.overlap_flags = []
        self.boxes = []
        self.count_history = deque(maxlen=10)
        self.lock = threading.Lock()

        # สำหรับควบคุมระบบ Hysteresis (เหมือน ui.py เป๊ะๆ)
        self.enable_overlap = True
        self._overlap_true_streak = 0
        self._overlap_false_streak = 0

        # Thread 1: Camera Frame Grabber (อัปเดต raw_frame เร็วสุดเท่าที่กล้องให้ได้
        # ไม่ขึ้นกับความเร็วโมเดลเลย - นี่คือส่วนที่ทำให้ video_feed ลื่นไม่ผูกกับโมเดล)
        self.grab_thread = threading.Thread(target=self._grab_frames, daemon=True)
        self.grab_thread.start()

        # Wait up to 5 seconds for the first frame to arrive
        start_wait = time.time()
        while self.raw_frame is None:
            if time.time() - start_wait > 5.0:
                print("Warning: Timed out waiting for camera frame. Using black placeholder.")
                self.raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                break
            time.sleep(0.01)

        # Thread 2: Background AI Inference Loop (รันแยกอิสระ ช้าแค่ไหนก็ไม่กระทบ
        # ความลื่นของภาพที่ /video_feed ส่งออกไป เพราะ video_feed อ่านจาก raw_frame
        # ตรงๆ ไม่รอผลจาก thread นี้)
        self.inference_thread = threading.Thread(target=self._run_inference, daemon=True)
        self.inference_thread.start()

    def _grab_frames(self):
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

            # If the webcam is unplugged (consecutive failed frames)
            if consecutive_failures >= 15:
                # 1. Show a warning frame on screen
                warning_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(warning_frame, "WEBCAM DISCONNECTED", (80, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                cv2.putText(warning_frame, "Reconnecting...", (200, 280),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                self.raw_frame = warning_frame

                # 2. Release and attempt to reopen the camera
                if self.cap is not None:
                    try:
                        self.cap.release()
                    except Exception:
                        pass

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
                                print(f"Successfully reconnected to webcam on index {index}")
                                break
                            else:
                                cap.release()
                    except Exception:
                        pass
                time.sleep(1.0)
            else:
                time.sleep(0.005)

    # ------------------------------------------------------------------
    # Overlap detection logic (ตรงกับ ui.py: connected-components +
    # convexity defects + area-ratio, gate ด้วยจำนวนกล่องต่อ blob)
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

        # อ้างอิงพื้นที่ "เม็ดยา 1 เม็ดทั่วไป" จาก blob ที่ map กับกล่องเดียวชัดเจนในเฟรมนี้
        single_areas = [
            self._blob_area(labels, lbl)
            for lbl, idxs in label_to_indices.items() if len(idxs) == 1
        ]
        single_areas = [a for a in single_areas if a > 0]
        reference_area = float(np.median(single_areas)) if single_areas else None

        # เตือนเฉพาะ blob ที่มีแค่ 1 กล่องคลุมอยู่ แต่รูปทรง/พื้นที่บ่งชี้ว่าน่าจะมี
        # 2 เม็ดซ่อนอยู่จริง (เสี่ยงนับขาด) - blob ที่มี >=2 กล่องแยกอยู่แล้วไม่ต้องเตือน
        for lbl, idxs in label_to_indices.items():
            if len(idxs) != 1:
                continue
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

    def _apply_overlap_hysteresis(self, has_overlap_raw):
        """Hysteresis เดียวกับ ui.py กันสถานะ overlap กระพริบเฟรมต่อเฟรม"""
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

    def _run_inference(self):
        # หมายเหตุ: เอาการสั่นอัตโนมัติ (เดิมสั่นทุกครั้งที่นับเพิ่ม/เจอ overlap) ออก
        # แล้ว เพื่อให้ตรงกับ ui.py ซึ่งสั่นเฉพาะตอนผู้ใช้กดปุ่มเองเท่านั้น
        while self.running:
            if self.raw_frame is not None:
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

    def toggle_overlap(self, enable: bool):
        """เปิด/ปิดระบบตรวจจับการซ้อนทับ - ตรงกับปุ่ม toggle ใน ui.py"""
        with self.lock:
            self.enable_overlap = bool(enable)
            if not self.enable_overlap:
                self.is_overlapping = False
                self.overlap_flags = []
                self._overlap_true_streak = 0
                self._overlap_false_streak = 0
        return self.enable_overlap

    def reset_counting(self):
        """รีเซ็ตค่าการนับทั้งหมด - ตรงกับปุ่ม 'นับใหม่' ใน ui.py"""
        with self.lock:
            self.count_history.clear()
            self.stable_count = 0
            self.detect_count = 0
            self.is_overlapping = False
            self.overlap_flags = []
            self._overlap_true_streak = 0
            self._overlap_false_streak = 0

    def trigger_vibe_pulse(self):
        """สั่นแบบ manual เท่านั้น (3 จังหวะ) - ตรงกับปุ่ม 'สั่นถาด' ใน ui.py"""
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
            return True
        return False

    def stop(self):
        self.running = False
        if self.grab_thread.is_alive(): self.grab_thread.join()
        if self.inference_thread.is_alive(): self.inference_thread.join()
        self.cap.release()


def get_ip_address():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return '10.42.0.1'

def get_hostname():
    return socket.gethostname()


engine = ThreadedPillDetectorEngine()

@app.context_processor
def inject_globals():
    return {
        'server_ip': get_ip_address(),
        'hostname': get_hostname(),
        'gpio_active': GPIO_AVAILABLE,
        'domain_name': 'pillcounter.com'
    }

# ==========================================
# Flask Web Routes (UI Pages)
# ==========================================

@app.route('/')
def page_menu():
    return render_template('menu.html', active_page='menu')

@app.route('/network/hotspot')
def page_network_hotspot():
    return render_template('hotspot.html', active_page='hotspot')

@app.route('/network/wifi')
def page_network_wifi():
    return render_template('wifi.html', active_page='wifi')

@app.route('/counter/normal')
def page_counter_normal():
    return render_template('counter_normal.html', active_page='normal')

@app.route('/counter/target')
def page_counter_target():
    return render_template('counter_target.html', active_page='target')

# Legacy route redirect
@app.route('/wifi')
def page_wifi_redirect():
    return redirect('/network/hotspot')

# ==========================================
# Video Streaming & API Endpoints
# ==========================================

def generate_mjpeg_stream():
    """
    Stream ภาพจาก raw_frame โดยตรง - ไม่รอผลจากโมเดลเลย ดังนั้นความลื่นของ
    ภาพขึ้นอยู่กับความเร็วกล้อง + การ encode JPEG เท่านั้น ไม่ผูกกับ FPS ของ
    โมเดล AI (โมเดลรันอยู่คนละ thread แล้วแค่ "แปะ" ผลล่าสุดที่มีทับภาพ)

    ปรับจากเดิม: เอา time.sleep(0.03) คงที่ (ล็อกเพดานไว้ ~33fps เสมอ) ออก
    เปลี่ยนเป็นเช็คว่าเฟรมใหม่มาหรือยัง (เทียบ id ของ object) ถ้ายังไม่มีเฟรม
    ใหม่จาก _grab_frames ก็รอสั้นๆ แล้ว skip ไม่ encode ซ้ำเฟรมเดิม (ประหยัด CPU)
    แต่ถ้ามีเฟรมใหม่มาก็ส่งทันที ทำให้ throughput จริงตามความเร็วกล้อง ไม่ใช่
    เพดานคงที่ที่ตั้งไว้เอง
    """
    last_frame_id = None
    while True:
        frame_src = engine.raw_frame
        if frame_src is None:
            time.sleep(0.01)
            continue

        if id(frame_src) == last_frame_id:
            # ยังไม่มีเฟรมใหม่จากกล้อง - รอสั้นๆ กัน busy loop กิน CPU เปล่าๆ
            time.sleep(0.004)
            continue
        last_frame_id = id(frame_src)

        frame = frame_src.copy()

        with engine.lock:
            boxes = list(engine.boxes)
            overlap_flags = list(engine.overlap_flags)
            is_overlapping = engine.is_overlapping
            stable_count = engine.stable_count

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            is_ov = (len(overlap_flags) > i and overlap_flags[i])
            color = (0, 0, 255) if is_ov else (0, 255, 0)

            cv2.circle(frame, (cx, cy), 5, color, -1)
            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), 1)

            if is_ov:
                cv2.putText(frame, "Overlap!", (cx + 10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.putText(frame, f"Pills Detected: {stable_count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255) if is_overlapping else (0, 255, 0), 3)

        if is_overlapping:
            cv2.putText(frame, "WARNING: PILLS OVERLAPPING!", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    return jsonify({
        'camera_connected': engine.cap.isOpened() if engine.cap else False,
        'model_loaded': True,
        'model_name': os.path.basename(engine.model_path),
        'motor_connected': engine.motor_connected,
        'server_ip': get_ip_address(),
        'hostname': get_hostname(),
        'domain_name': 'pillcounter.com'
    })

@app.route('/api/counter/data')
def api_counter_data():
    with engine.lock:
        return jsonify({
            'stable_count': engine.stable_count,
            'raw_count': engine.detect_count,
            'is_overlapping': engine.is_overlapping,
            'overlap_enabled': engine.enable_overlap,
            'target_count': engine.target_count
        })

@app.route('/api/counter/set_target', methods=['POST'])
def api_set_target():
    data = request.get_json() or {}
    val = int(data.get('target', 20))
    with engine.lock:
        engine.target_count = val
    return jsonify({'status': 'ok', 'target': val})

@app.route('/api/counter/toggle_overlap', methods=['POST'])
def api_toggle_overlap():
    """เปิด/ปิดระบบตรวจจับการซ้อนทับ - เทียบเท่าปุ่ม toggle ใน ui.py"""
    data = request.get_json() or {}
    enable = bool(data.get('enable', True))
    new_state = engine.toggle_overlap(enable)
    return jsonify({'status': 'ok', 'overlap_enabled': new_state})

@app.route('/api/counter/reset', methods=['POST'])
def api_reset_counter():
    """รีเซ็ตค่าการนับ - เทียบเท่าปุ่ม 'นับใหม่' ใน ui.py"""
    engine.reset_counting()
    return jsonify({'status': 'ok'})

@app.route('/api/trigger_shake', methods=['POST'])
def api_trigger_shake():
    triggered = engine.trigger_vibe_pulse()
    return jsonify({'status': 'triggered' if triggered else 'no_motor'})

@app.route('/api/wifi/scan')
def api_wifi_scan():

    try:
        subprocess.run("sudo nmcli device wifi rescan", shell=True, timeout=5)
        res = subprocess.run("sudo nmcli --terse --fields SSID,SIGNAL,SECURITY device wifi list", shell=True, capture_output=True, text=True)
        networks = []
        seen_ssids = set()
        for line in res.stdout.splitlines():
            parts = line.split(':')
            if len(parts) >= 3:
                ssid = parts[0].strip()
                signal = parts[1].strip()
                security = parts[2].strip()
                if ssid and ssid not in seen_ssids and ssid != 'PillCounter-Hotspot':
                    seen_ssids.add(ssid)
                    networks.append({
                        'ssid': ssid,
                        'signal': signal,
                        'security': security if security else 'Open'
                    })
        return jsonify({'status': 'ok', 'networks': networks})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/wifi/configure', methods=['POST'])
def api_wifi_configure():
    data = request.get_json() or {}
    mode = data.get('mode', 'hotspot')

    try:
        if mode == 'hotspot':
            subprocess.run("sudo nmcli device disconnect wlan0 >/dev/null 2>&1", shell=True)
            time.sleep(1)
            cmd = "sudo nmcli device wifi hotspot ssid PillCounter-Hotspot password 'pillcounter123' ifname wlan0"
            subprocess.Popen(cmd, shell=True)
            msg = "Hotspot Mode Activated! Connect phone to 'PillCounter-Hotspot' (Password: pillcounter123). URL: http://pillcounter.com:5000"
        else:
            ssid = data.get('ssid', '').strip()
            password = data.get('password', '').strip()

            # Stop Hotspot first
            subprocess.run("sudo nmcli connection down Hotspot >/dev/null 2>&1", shell=True)
            subprocess.run("sudo nmcli device disconnect wlan0 >/dev/null 2>&1", shell=True)
            time.sleep(1)

            # WPA2 requires password length >= 8 characters. If shorter/empty, try open network
            if len(password) >= 8:
                cmd = f'sudo nmcli device wifi connect "{ssid}" password "{password}"'
            else:
                cmd = f'sudo nmcli device wifi connect "{ssid}"'

            res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            success = (res.returncode == 0)

            if not success:
                # Retry without password for Open network
                retry_cmd = f'sudo nmcli device wifi connect "{ssid}"'
                res_retry = subprocess.run(retry_cmd, shell=True, capture_output=True, text=True)
                success = (res_retry.returncode == 0)

            if success:
                time.sleep(2) # Wait for DHCP IP assignment
                new_ip = get_ip_address()
                msg = f"Successfully connected to Wi-Fi '{ssid}'! Your New Access URL: http://{new_ip}:5000 (or http://pillcounter.com:5000)"
            else:
                msg = f"Failed to connect to '{ssid}'. Please check password."
        return jsonify({'status': 'ok', 'message': msg})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Captive Portal Redirects
@app.route('/hotspot-detect.html')             # iOS / Apple
@app.route('/library/test/success.html')      # Apple
@app.route('/generate_204')                    # Android / Chrome
@app.route('/gen_204')                         # Android
@app.route('/canonical.html')                  # Android
@app.route('/ncsi.txt')                        # Windows
@app.route('/connecttest.txt')                 # Windows
def captive_portal_auto_popup():
    return redirect("http://10.42.0.1:5000/", code=302)

@app.errorhandler(404)
def captive_portal_catch_all(e):
    if request.host != '10.42.0.1:5000' and request.host != '10.42.0.1':
        return redirect("http://10.42.0.1:5000/", code=302)
    return render_template('menu.html', active_page='menu'), 404

if __name__ == '__main__':
    ip = get_ip_address()
    host = get_hostname()

    # Configure dnsmasq for pillcounter.com domain mapping to 10.42.0.1
    try:
        dns_cmd = 'sudo bash -c \'mkdir -p /etc/NetworkManager/dnsmasq-shared.d && echo -e "address=/pillcounter.com/10.42.0.1\\naddress=/pill.com/10.42.0.1\\naddress=/#/10.42.0.1" > /etc/NetworkManager/dnsmasq-shared.d/pillcounter.conf\''
        subprocess.Popen(dns_cmd, shell=True)
    except Exception:
        pass

    import shutil
    if shutil.which("iptables"):
        try:
            subprocess.Popen("sudo iptables -t nat -F PREROUTING", shell=True)
            subprocess.Popen("sudo iptables -t nat -A PREROUTING -i wlan0 -p tcp --dport 80 -j REDIRECT --to-port 5000", shell=True)
            subprocess.Popen("sudo iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 5000", shell=True)
        except Exception:
            pass

    # Auto-start Hotspot mode by default if running on Raspberry Pi
    if GPIO_AVAILABLE or os.path.exists('/etc/rpi-issue'):
        # 1. Wait up to 30 seconds for NetworkManager to load
        for _ in range(30):
            nm_check = subprocess.run("pgrep NetworkManager", shell=True, capture_output=True)
            if nm_check.returncode == 0:
                break
            time.sleep(1)

        # 2. Wait up to 30 seconds for wlan0 hardware to register in NetworkManager
        for _ in range(30):
            status_check = subprocess.run("nmcli device status", shell=True, capture_output=True, text=True)
            if "wlan0" in status_check.stdout:
                break
            time.sleep(1)

        # 3. Disconnect wlan0 first to release resource
        subprocess.run("sudo nmcli device disconnect wlan0 >/dev/null 2>&1", shell=True)
        time.sleep(2)

        # 4. Activate Hotspot
        subprocess.Popen("sudo nmcli device wifi hotspot ssid PillCounter-Hotspot password 'pillcounter123' ifname wlan0", shell=True)

    print("\n==========================================")
    print(" 💊 PILL COUNTER PRO WEB APPLICATION ONLINE")
    print(" • Domain Name (Universal) : http://pillcounter.com:5000")
    print(" • Default Hotspot Gateway : http://10.42.0.1:5000")
    print(f" • Smartphone IP URL       : http://{ip}:5000")
    print("==========================================\n")
    app.run(host='0.0.0.0', port=5000, threaded=True)