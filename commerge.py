import io
import sys
import json
import urllib.request
import webbrowser
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
import pypdfium2 as pdfium

from PySide6.QtCore import Qt, QRect, QRectF, QPointF, Signal
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor, QBrush, QIcon
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QListWidget,
    QLabel, QFileDialog, QHBoxLayout, QVBoxLayout, QComboBox,
    QMessageBox, QFrame, QRadioButton, QButtonGroup, QSpinBox,
    QCheckBox, QGroupBox, QSlider, QTabWidget, QScrollArea
)
from PySide6.QtPrintSupport import QPrinter, QPrintDialog

# Application Constants
CURRENT_VERSION = "1.0.0"
GITHUB_REPO = "Akash123456780/easycrop-mobile"  # Replace with your actual GitHub username/repo

# Standard Print Presets (Dimensions in mm @ 300 DPI)
PRESETS = {
    "PVC Card / ID Badge (85.6 × 54.0 mm)": {"w_mm": 85.60, "h_mm": 53.98, "dpi": 300, "custom_px": False},
    "Passport Photo (35 × 45 mm)": {"w_mm": 35.00, "h_mm": 45.00, "dpi": 300, "custom_px": False},
    "Stamp Size Photo (20 × 25 mm)": {"w_mm": 20.00, "h_mm": 25.00, "dpi": 300, "custom_px": False},
    "A4 Full Page Document": {"w_mm": 210.0, "h_mm": 297.0, "dpi": 300, "custom_px": False},
    "Custom Exact Pixels (W × H px)": {"w_mm": None, "h_mm": None, "dpi": 300, "custom_px": True},
    "Custom / Free Aspect": {"w_mm": None, "h_mm": None, "dpi": 300, "custom_px": False},
}


def check_for_updates(parent_window, manual_check=False):
    try:
        api_url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
        req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
        
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            latest_version = data["tag_name"].strip("v")
            download_url = None
            
            for asset in data.get("assets", []):
                if asset["name"].endswith(".exe"):
                    download_url = asset["browser_download_url"]
                    break
            
            if latest_version > CURRENT_VERSION:
                msg = QMessageBox(parent_window)
                msg.setIcon(QMessageBox.Information)
                msg.setWindowTitle("Update Available")
                msg.setText(f"A new version ({latest_version}) is available!\nWould you like to download it now?")
                msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                
                if msg.exec() == QMessageBox.Yes and download_url:
                    webbrowser.open(download_url)
            elif manual_check:
                QMessageBox.information(parent_window, "Up to Date", f"You are using the latest version of commerge ({CURRENT_VERSION}).")
                
    except Exception as e:
        if manual_check:
            QMessageBox.warning(parent_window, "Update Error", f"Could not check for updates:\n{str(e)}")


class InteractiveCropCanvas(QWidget):
    boxChanged = Signal()

    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background-color: #0f111a; border-radius: 6px;")
        self.setMouseTracking(True)

        self.cv_raw = None
        self.cv_display = None
        self.pixmap_orig = None
        self.target_ratio = 85.60 / 53.98

        self.crop_rect = None
        self.active_mode = None
        self.drag_start = None
        self.handle_size = 10

        self.brightness = 0
        self.contrast = 0
        self.sharpness = 0
        self.fine_angle = 0
        
        self.splash_pixmap = QPixmap("splash_logo.png")

    def load_pil_image(self, pil_img):
        rgb = np.array(pil_img.convert("RGB"))
        self.cv_raw = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self.fine_angle = 0
        self.brightness = 0
        self.contrast = 0
        self.sharpness = 0
        self.apply_image_pipeline()
        self.crop_rect = None
        self.reset_to_preset()

    def apply_image_pipeline(self):
        if self.cv_raw is None:
            return

        img = self.cv_raw.copy()

        if self.fine_angle != 0:
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, self.fine_angle, 1.0)
            img = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)

        alpha = 1.0 + (self.contrast / 100.0)
        beta = self.brightness
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        if self.sharpness > 0:
            kernel = np.array([[-1, -1, -1],
                               [-1, 9 + (self.sharpness / 20.0), -1],
                               [-1, -1, -1]])
            img = cv2.filter2D(img, -1, kernel)

        self.cv_display = img
        self._update_pixmap_from_cv()

    def rotate_90(self, clockwise=True):
        if self.cv_raw is None:
            return
        code = cv2.ROTATE_90_CLOCKWISE if clockwise else cv2.ROTATE_90_COUNTERCLOCKWISE
        self.cv_raw = cv2.rotate(self.cv_raw, code)
        self.apply_image_pipeline()
        self.reset_to_preset()

    def _update_pixmap_from_cv(self):
        if self.cv_display is None:
            return
        rgb = cv2.cvtColor(self.cv_display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.pixmap_orig = QPixmap.fromImage(q_img)
        self.update()

    def get_render_geometry(self):
        if not self.pixmap_orig or self.pixmap_orig.isNull():
            return None
        scaled = self.pixmap_orig.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        ox = (self.width() - scaled.width()) // 2
        oy = (self.height() - scaled.height()) // 2
        return QRect(ox, oy, scaled.width(), scaled.height())

    def set_target_aspect(self, ratio):
        self.target_ratio = ratio
        self.reset_to_preset()

    def reset_to_preset(self):
        geom = self.get_render_geometry()
        if not geom:
            return

        if self.target_ratio is None:
            w = geom.width() * 0.8
            h = geom.height() * 0.8
        else:
            w = geom.width() * 0.85
            h = w / self.target_ratio
            if h > geom.height() * 0.85:
                h = geom.height() * 0.85
                w = h * self.target_ratio

        cx = geom.center().x()
        cy = geom.center().y()
        self.crop_rect = QRectF(cx - w / 2, cy - h / 2, w, h)
        self.update()
        self.boxChanged.emit()

    def remove_background(self):
        if self.cv_raw is None:
            return False

        rgb_img = cv2.cvtColor(self.cv_raw, cv2.COLOR_BGR2RGB)
        pil_in = Image.fromarray(rgb_img)

        try:
            from rembg import remove
            output_pil = remove(pil_in)
            bg_white = Image.new("RGB", output_pil.size, (255, 255, 255))
            if output_pil.mode == "RGBA":
                bg_white.paste(output_pil, mask=output_pil.split()[3])
            else:
                bg_white = output_pil.convert("RGB")

            self.load_pil_image(bg_white)
            return True
        except Exception:
            mask = np.zeros(self.cv_raw.shape[:2], np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            h, w = self.cv_raw.shape[:2]
            rect = (int(w * 0.05), int(h * 0.05), int(w * 0.9), int(h * 0.9))
            cv2.grabCut(self.cv_raw, mask, rect, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            out_img = self.cv_raw * mask2[:, :, np.newaxis]
            white_bg = np.ones_like(self.cv_raw, dtype=np.uint8) * 255
            final_cv = np.where(mask2[:, :, np.newaxis] == 1, out_img, white_bg)

            self.cv_raw = final_cv
            self.apply_image_pipeline()
            return True

    def auto_detect_and_snap(self):
        geom = self.get_render_geometry()
        if self.cv_display is None or not geom:
            return False

        gray = cv2.cvtColor(self.cv_display, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        found_box = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.contourArea(c) > (gray.shape[0] * gray.shape[1] * 0.08):
                found_box = cv2.boundingRect(approx)
                break

        if found_box:
            bx, by, bw, bh = found_box
            scale_x = geom.width() / self.cv_display.shape[1]
            scale_y = geom.height() / self.cv_display.shape[0]

            self.crop_rect = QRectF(
                geom.left() + (bx * scale_x),
                geom.top() + (by * scale_y),
                bw * scale_x,
                bh * scale_y
            )
            self.update()
            self.boxChanged.emit()
            return True

        self.reset_to_preset()
        return False

    def get_cropped_pil(self):
        if self.cv_display is None or not self.crop_rect:
            return None

        geom = self.get_render_geometry()
        r = self.crop_rect.normalized()

        rel_x = max(0, r.left() - geom.left())
        rel_y = max(0, r.top() - geom.top())
        rel_w = min(geom.width(), r.width())
        rel_h = min(geom.height(), r.height())

        scale_x = self.cv_display.shape[1] / geom.width()
        scale_y = self.cv_display.shape[0] / geom.height()

        crop_box = (
            int(round(rel_x * scale_x)),
            int(round(rel_y * scale_y)),
            int(round((rel_x + rel_w) * scale_x)),
            int(round((rel_y + rel_h) * scale_y))
        )

        rgb_full = cv2.cvtColor(self.cv_display, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_full)
        return img.crop(crop_box)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        geom = self.get_render_geometry()
        if self.pixmap_orig and geom:
            scaled = self.pixmap_orig.scaled(
                geom.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            painter.drawPixmap(geom.topLeft(), scaled)
        elif not self.pixmap_orig and not self.splash_pixmap.isNull():
            scaled_splash = self.splash_pixmap.scaled(
                self.size() * 0.75, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            sx = (self.width() - scaled_splash.width()) // 2
            sy = (self.height() - scaled_splash.height()) // 2
            painter.drawPixmap(sx, sy, scaled_splash)

        if not self.crop_rect or not geom:
            painter.end()
            return

        r = self.crop_rect.normalized()

        painter.setBrush(QBrush(QColor(0, 0, 0, 160)))
        painter.setPen(Qt.NoPen)
        painter.drawRect(QRectF(0, 0, self.width(), r.top()))
        painter.drawRect(QRectF(0, r.bottom(), self.width(), self.height() - r.bottom()))
        painter.drawRect(QRectF(0, r.top(), r.left(), r.height()))
        painter.drawRect(QRectF(r.right(), r.top(), self.width() - r.right(), r.height()))

        pen = QPen(QColor(0, 230, 118), 2, Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(r)

        handles = self._get_handles(r)
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.setBrush(QBrush(QColor(0, 230, 118)))
        for pt in handles.values():
            painter.drawRect(QRectF(pt.x() - self.handle_size / 2, pt.y() - self.handle_size / 2,
                                    self.handle_size, self.handle_size))
        painter.end()

    def _get_handles(self, r):
        return {
            "nw": r.topLeft(), "n": QPointF(r.center().x(), r.top()), "ne": r.topRight(),
            "e": QPointF(r.right(), r.center().y()), "se": r.bottomRight(),
            "s": QPointF(r.center().x(), r.bottom()), "sw": r.bottomLeft(),
            "w": QPointF(r.left(), r.center().y())
        }

    def _hit_test(self, pos):
        if not self.crop_rect:
            return None
        r = self.crop_rect.normalized()
        handles = self._get_handles(r)
        for name, pt in handles.items():
            if (pos - pt).manhattanLength() <= self.handle_size + 4:
                return name
        if r.contains(pos):
            return "inside"
        return None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.crop_rect:
            self.drag_start = event.position()
            self.active_mode = self._hit_test(self.drag_start)

    def mouseMoveEvent(self, event):
        pos = event.position()
        if not self.active_mode:
            mode = self._hit_test(pos)
            cursors = {
                "nw": Qt.SizeFDiagCursor, "se": Qt.SizeFDiagCursor,
                "ne": Qt.SizeBDiagCursor, "sw": Qt.SizeBDiagCursor,
                "n": Qt.SizeVerCursor, "s": Qt.SizeVerCursor,
                "e": Qt.SizeHorCursor, "w": Qt.SizeHorCursor,
                "inside": Qt.SizeAllCursor
            }
            self.setCursor(cursors.get(mode, Qt.CrossCursor))
            return

        geom = self.get_render_geometry()
        if not geom:
            return

        clamped_x = max(geom.left(), min(geom.right(), pos.x()))
        clamped_y = max(geom.top(), min(geom.bottom(), pos.y()))

        r = self.crop_rect
        if self.active_mode == "inside":
            dx = pos.x() - self.drag_start.x()
            dy = pos.y() - self.drag_start.y()

            nr = r.translated(dx, dy)
            if nr.left() >= geom.left() and nr.right() <= geom.right() and \
               nr.top() >= geom.top() and nr.bottom() <= geom.bottom():
                self.crop_rect = nr
                self.drag_start = pos
        else:
            x1, y1, x2, y2 = r.left(), r.top(), r.right(), r.bottom()
            if "w" in self.active_mode: x1 = clamped_x
            if "e" in self.active_mode: x2 = clamped_x
            if "n" in self.active_mode: y1 = clamped_y
            if "s" in self.active_mode: y2 = clamped_y

            if self.target_ratio:
                w = abs(x2 - x1)
                h = w / self.target_ratio
                y2 = y1 + h if y2 >= y1 else y1 - h

            self.crop_rect = QRectF(x1, y1, x2 - x1, y2 - y1).normalized()

        self.update()
        self.boxChanged.emit()

    def mouseReleaseEvent(self, event):
        self.active_mode = None
        self.drag_start = None


class CommergeMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("commerge - Smart File Processing v" + CURRENT_VERSION)
        self.resize(1400, 940)
        self.setStyleSheet("QMainWindow { background-color: #161822; color: #f1f2f6; }")
        
        self.setWindowIcon(QIcon("app_icon.ico"))

        self.loaded_items = []
        self.image_merge_queue = []
        self.pdf_merge_queue = []

        self._init_ui()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_vbox = QVBoxLayout(central)
        main_vbox.setContentsMargins(10, 10, 10, 10)
        main_vbox.setSpacing(6)

        # Header Bar with App Title and Up-Right Update Button
        header_bar = QHBoxLayout()
        header_bar.setContentsMargins(5, 0, 5, 0)
        
        lbl_brand = QLabel("commerge")
        lbl_brand.setStyleSheet("font-size: 17px; font-weight: bold; color: #00e676; letter-spacing: 1px;")
        lbl_sub = QLabel("Smart File Processing")
        lbl_sub.setStyleSheet("font-size: 11px; color: #8890b5; margin-left: 6px; padding-top: 3px;")
        header_bar.addWidget(lbl_brand)
        header_bar.addWidget(lbl_sub)
        header_bar.addStretch()

        # Update button positioned in the upper right corner
        btn_top_update = QPushButton("🔄 Check for Updates")
        btn_top_update.setStyleSheet("""
            QPushButton {
                background-color: #2b2f45;
                color: #00e676;
                border: 1px solid #3d4465;
                padding: 6px 14px;
                font-weight: bold;
                font-size: 11px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #383d59;
                border-color: #00e676;
            }
        """)
        btn_top_update.clicked.connect(lambda: check_for_updates(self, manual_check=True))
        header_bar.addWidget(btn_top_update)

        main_vbox.addLayout(header_bar)

        # Main Workspace Tab Widget
        self.workspace_tabs = QTabWidget()
        self.workspace_tabs.setStyleSheet("""
            QTabBar::tab { background: #2b2f45; color: white; padding: 10px 20px; border-top-left-radius: 6px; border-top-right-radius: 6px; font-weight: bold; font-size: 12px; }
            QTabBar::tab:selected { background: #00e676; color: #090a0f; }
        """)

        # TAB 1: Document Editor Workspace
        tab_editor = QWidget()
        editor_layout = QHBoxLayout(tab_editor)
        editor_layout.setContentsMargins(5, 5, 5, 5)

        editor_sidebar = QFrame()
        editor_sidebar.setFixedWidth(400)
        editor_sidebar.setStyleSheet("background-color: #1e2130; border-radius: 8px; padding: 4px;")
        es_layout = QVBoxLayout(editor_sidebar)

        es_layout.addWidget(QLabel("DIMENSION PRESETS", styleSheet="color: #00e676; font-size: 11px; font-weight: bold;"))
        self.preset_box = QComboBox()
        self.preset_box.addItems(list(PRESETS.keys()))
        self.preset_box.setStyleSheet("background-color: #2b2f45; color: white; padding: 4px; border-radius: 4px;")
        self.preset_box.currentIndexChanged.connect(self.on_preset_change)
        es_layout.addWidget(self.preset_box)

        self.grp_custom_px = QGroupBox("Custom Pixel Resolution")
        self.grp_custom_px.setStyleSheet("QGroupBox { color: #feca57; font-weight: bold; border: 1px solid #576574; }")
        px_layout = QHBoxLayout(self.grp_custom_px)
        px_layout.addWidget(QLabel("W:", styleSheet="color: white;"))
        self.spin_px_w = QSpinBox()
        self.spin_px_w.setRange(50, 10000)
        self.spin_px_w.setValue(1011)
        self.spin_px_w.setSuffix(" px")
        self.spin_px_w.setStyleSheet("background-color: #2b2f45; color: #00e676;")
        self.spin_px_w.valueChanged.connect(self.on_custom_px_change)
        px_layout.addWidget(self.spin_px_w)

        px_layout.addWidget(QLabel("H:", styleSheet="color: white;"))
        self.spin_px_h = QSpinBox()
        self.spin_px_h.setRange(50, 10000)
        self.spin_px_h.setValue(638)
        self.spin_px_h.setSuffix(" px")
        self.spin_px_h.setStyleSheet("background-color: #2b2f45; color: #00e676;")
        self.spin_px_h.valueChanged.connect(self.on_custom_px_change)
        px_layout.addWidget(self.spin_px_h)
        es_layout.addWidget(self.grp_custom_px)
        self.grp_custom_px.hide()

        self.orient_frame = QFrame()
        orient_layout = QHBoxLayout(self.orient_frame)
        orient_layout.setContentsMargins(0, 0, 0, 0)
        self.btn_land = QRadioButton("Landscape")
        self.btn_port = QRadioButton("Portrait")
        self.btn_land.setChecked(True)
        self.btn_land.setStyleSheet("color: white;")
        self.btn_port.setStyleSheet("color: white;")
        self.orient_group = QButtonGroup()
        self.orient_group.addButton(self.btn_land)
        self.orient_group.addButton(self.btn_port)
        self.btn_land.toggled.connect(self.on_preset_change)
        orient_layout.addWidget(self.btn_land)
        orient_layout.addWidget(self.btn_port)
        es_layout.addWidget(self.orient_frame)

        ai_box = QGroupBox("AI & Computer Vision Tools")
        ai_box.setStyleSheet("QGroupBox { color: #81ecec; font-weight: bold; border: 1px solid #3d4465; }")
        ai_layout = QVBoxLayout(ai_box)
        btn_rmbg = QPushButton("🪄 Remove Background (AI)")
        btn_rmbg.setStyleSheet("background-color: #6c5ce7; color: white; font-weight: bold; padding: 6px;")
        btn_rmbg.clicked.connect(self.handle_bg_remove)
        ai_layout.addWidget(btn_rmbg)

        btn_auto_snap = QPushButton("⚡ Auto-Detect Document Boundary")
        btn_auto_snap.setStyleSheet("background-color: #3d4465; color: #00e676; font-weight: bold; padding: 6px;")
        btn_auto_snap.clicked.connect(self.handle_auto_detect)
        ai_layout.addWidget(btn_auto_snap)
        es_layout.addWidget(ai_box)

        tune_box = QGroupBox("Image Adjustments & Deskew")
        tune_box.setStyleSheet("QGroupBox { color: #81ecec; font-weight: bold; border: 1px solid #3d4465; }")
        tb_layout = QVBoxLayout(tune_box)

        btn_rot_row = QHBoxLayout()
        btn_r90_ccw = QPushButton("↺ 90° CCW")
        btn_r90_ccw.setStyleSheet("background-color: #3d4465; color: white;")
        btn_r90_ccw.clicked.connect(lambda: self.canvas.rotate_90(False))
        btn_r90_cw = QPushButton("↻ 90° CW")
        btn_r90_cw.setStyleSheet("background-color: #3d4465; color: white;")
        btn_r90_cw.clicked.connect(lambda: self.canvas.rotate_90(True))
        btn_rot_row.addWidget(btn_r90_ccw)
        btn_rot_row.addWidget(btn_r90_cw)
        tb_layout.addLayout(btn_rot_row)

        fine_row = QHBoxLayout()
        fine_row.addWidget(QLabel("Deskew Angle:", styleSheet="color: white; font-size: 10px;"))
        self.slider_angle = QSlider(Qt.Horizontal)
        self.slider_angle.setRange(-15, 15)
        self.slider_angle.setValue(0)
        self.slider_angle.valueChanged.connect(self.on_slider_change)
        fine_row.addWidget(self.slider_angle)
        self.lbl_angle_val = QLabel("0°", styleSheet="color: #00e676; font-weight: bold;")
        fine_row.addWidget(self.lbl_angle_val)
        tb_layout.addLayout(fine_row)

        b_row = QHBoxLayout()
        b_row.addWidget(QLabel("Brightness:", styleSheet="color: white; font-size: 10px;"))
        self.slider_bright = QSlider(Qt.Horizontal)
        self.slider_bright.setRange(-80, 80)
        self.slider_bright.setValue(0)
        self.slider_bright.valueChanged.connect(self.on_slider_change)
        b_row.addWidget(self.slider_bright)
        tb_layout.addLayout(b_row)

        c_row = QHBoxLayout()
        c_row.addWidget(QLabel("Contrast:", styleSheet="color: white; font-size: 10px;"))
        self.slider_contrast = QSlider(Qt.Horizontal)
        self.slider_contrast.setRange(-50, 100)
        self.slider_contrast.setValue(0)
        self.slider_contrast.valueChanged.connect(self.on_slider_change)
        c_row.addWidget(self.slider_contrast)
        tb_layout.addLayout(c_row)

        s_row = QHBoxLayout()
        s_row.addWidget(QLabel("Sharpness:", styleSheet="color: white; font-size: 10px;"))
        self.slider_sharp = QSlider(Qt.Horizontal)
        self.slider_sharp.setRange(0, 100)
        self.slider_sharp.setValue(0)
        self.slider_sharp.valueChanged.connect(self.on_slider_change)
        s_row.addWidget(self.slider_sharp)
        tb_layout.addLayout(s_row)
        es_layout.addWidget(tune_box)

        grp_compress = QGroupBox("Target Size & Multi-Export")
        grp_compress.setStyleSheet("QGroupBox { color: #81ecec; font-weight: bold; border: 1px solid #3d4465; }")
        comp_layout = QVBoxLayout(grp_compress)
        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Format:", styleSheet="color: white; font-size: 11px;"))
        self.combo_format = QComboBox()
        self.combo_format.addItems(["PNG (300 DPI)", "JPEG (.jpg)", "PDF Document (.pdf)"])
        self.combo_format.setStyleSheet("background-color: #2b2f45; color: white; padding: 3px;")
        fmt_row.addWidget(self.combo_format)
        comp_layout.addLayout(fmt_row)

        self.chk_limit_size = QCheckBox("Compress to target size")
        self.chk_limit_size.setStyleSheet("color: white; font-size: 11px;")
        comp_layout.addWidget(self.chk_limit_size)

        limit_row = QHBoxLayout()
        limit_row.addWidget(QLabel("Max Size:", styleSheet="color: #dfe6e9; font-size: 11px;"))
        self.spin_target_kb = QSpinBox()
        self.spin_target_kb.setRange(10, 50000)
        self.spin_target_kb.setValue(200)
        self.spin_target_kb.setSuffix(" KB")
        self.spin_target_kb.setStyleSheet("background-color: #2b2f45; color: #00e676; font-weight: bold;")
        limit_row.addWidget(self.spin_target_kb)
        comp_layout.addLayout(limit_row)

        self.chk_simul_export = QCheckBox("Export Both High-Res Print + Web (<100KB)")
        self.chk_simul_export.setStyleSheet("color: #feca57; font-size: 10px; font-weight: bold;")
        comp_layout.addWidget(self.chk_simul_export)
        es_layout.addWidget(grp_compress)

        self.file_list = QListWidget()
        self.file_list.setFixedHeight(80)
        self.file_list.setStyleSheet("background-color: #12141d; color: white; border-radius: 4px;")
        self.file_list.currentRowChanged.connect(self.load_selected_item)
        es_layout.addWidget(self.file_list)

        btn_row = QHBoxLayout()
        btn_add = QPushButton("＋ Add")
        btn_add.setStyleSheet("background-color: #2b2f45; color: white; padding: 5px;")
        btn_add.clicked.connect(self.add_files)
        btn_folder = QPushButton("📁 Folder")
        btn_folder.setStyleSheet("background-color: #2b2f45; color: white; padding: 5px;")
        btn_folder.clicked.connect(self.add_folder)
        btn_scan = QPushButton("🖨️ Scan")
        btn_scan.setStyleSheet("background-color: #0984e3; color: white; padding: 5px; font-weight: bold;")
        btn_scan.clicked.connect(self.scan_from_device)
        btn_remove = QPushButton("✕ Remove")
        btn_remove.setStyleSheet("background-color: #d63031; color: white; padding: 5px; font-weight: bold;")
        btn_remove.clicked.connect(self.remove_selected_item)
        btn_row.addWidget(btn_add)
        btn_row.addWidget(btn_folder)
        btn_row.addWidget(btn_scan)
        btn_row.addWidget(btn_remove)
        es_layout.addLayout(btn_row)

        btn_batch_pdf = QPushButton("📑 Batch Extract All PDF Pages")
        btn_batch_pdf.setStyleSheet("background-color: #e17055; color: white; font-weight: bold; padding: 6px;")
        btn_batch_pdf.clicked.connect(self.batch_extract_pdf_pages)
        es_layout.addWidget(btn_batch_pdf)

        act_row = QHBoxLayout()
        btn_print = QPushButton("🖨️ PRINT")
        btn_print.setMinimumHeight(40)
        btn_print.setStyleSheet("background-color: #0984e3; color: white; font-weight: bold; font-size: 12px; border-radius: 6px;")
        btn_print.clicked.connect(self.print_current_card)
        btn_export = QPushButton("💾 EXPORT")
        btn_export.setMinimumHeight(40)
        btn_export.setStyleSheet("background-color: #00e676; color: #090a0f; font-weight: bold; font-size: 12px; border-radius: 6px;")
        btn_export.clicked.connect(self.export_single_processed)
        act_row.addWidget(btn_print)
        act_row.addWidget(btn_export)
        es_layout.addLayout(act_row)

        editor_scroll = QScrollArea()
        editor_scroll.setWidgetResizable(True)
        editor_scroll.setWidget(editor_sidebar)
        editor_scroll.setFixedWidth(420)
        editor_layout.addWidget(editor_scroll)

        self.canvas = InteractiveCropCanvas()
        editor_layout.addWidget(self.canvas, 1)

        self.workspace_tabs.addTab(tab_editor, "Document Editor")

        # TAB 2: Merge Image Workspace
        tab_img_merge = QWidget()
        im_layout = QHBoxLayout(tab_img_merge)
        im_layout.setContentsMargins(5, 5, 5, 5)

        im_sidebar = QFrame()
        im_sidebar.setFixedWidth(400)
        im_sidebar.setStyleSheet("background-color: #1e2130; border-radius: 8px; padding: 8px;")
        ims_layout = QVBoxLayout(im_sidebar)

        ims_layout.addWidget(QLabel("UNLIMITED IMAGE MERGE", styleSheet="color: #00e676; font-size: 13px; font-weight: bold;"))
        btn_add_img_queue = QPushButton("＋ Add Current Canvas Crop to Queue")
        btn_add_img_queue.setStyleSheet("background-color: #3d4465; color: white; padding: 10px; font-weight: bold;")
        btn_add_img_queue.clicked.connect(self.add_to_image_queue)
        ims_layout.addWidget(btn_add_img_queue)

        self.img_queue_list = QListWidget()
        self.img_queue_list.setStyleSheet("background-color: #12141d; color: white; font-size: 12px;")
        ims_layout.addWidget(self.img_queue_list)

        img_reorder_row = QHBoxLayout()
        btn_img_up = QPushButton("▲ Move Up")
        btn_img_up.setStyleSheet("background-color: #2b2f45; color: #00e676; font-weight: bold; padding: 4px;")
        btn_img_up.clicked.connect(lambda: self.move_queue_item(self.img_queue_list, self.image_merge_queue, -1, self.update_image_preview))
        
        btn_img_down = QPushButton("▼ Move Down")
        btn_img_down.setStyleSheet("background-color: #2b2f45; color: #00e676; font-weight: bold; padding: 4px;")
        btn_img_down.clicked.connect(lambda: self.move_queue_item(self.img_queue_list, self.image_merge_queue, 1, self.update_image_preview))
        
        btn_rem_img_queue = QPushButton("✕ Remove")
        btn_rem_img_queue.setStyleSheet("background-color: #d63031; color: white; font-weight: bold; padding: 4px;")
        btn_rem_img_queue.clicked.connect(lambda: self.remove_from_queue_list(self.img_queue_list, self.image_merge_queue, self.update_image_preview))
        
        img_reorder_row.addWidget(btn_img_up)
        img_reorder_row.addWidget(btn_img_down)
        img_reorder_row.addWidget(btn_rem_img_queue)
        ims_layout.addLayout(img_reorder_row)

        ims_layout.addWidget(QLabel("Layout Arrangement:", styleSheet="color: white; font-weight: bold; margin-top: 6px;"))
        self.combo_img_layout = QComboBox()
        self.combo_img_layout.addItems(["Side-by-Side (Horizontal)", "Stacked (Vertical)", "Standard A4 Grid Sheet"])
        self.combo_img_layout.setStyleSheet("background-color: #2b2f45; color: white; padding: 6px;")
        self.combo_img_layout.currentIndexChanged.connect(self.update_image_preview)
        ims_layout.addWidget(self.combo_img_layout)

        btn_export_images = QPushButton("💾 MERGE & SAVE ALL IMAGES")
        btn_export_images.setMinimumHeight(45)
        btn_export_images.setStyleSheet("background-color: #0984e3; color: white; font-weight: bold; font-size: 13px; border-radius: 6px;")
        btn_export_images.clicked.connect(self.export_merged_images)
        ims_layout.addWidget(btn_export_images)

        im_scroll = QScrollArea()
        im_scroll.setWidgetResizable(True)
        im_scroll.setWidget(im_sidebar)
        im_scroll.setFixedWidth(420)
        im_layout.addWidget(im_scroll)

        self.lbl_img_preview = QLabel()
        self.lbl_img_preview.setAlignment(Qt.AlignCenter)
        self.lbl_img_preview.setStyleSheet("background-color: #0f111a; border: 2px dashed #3d4465; border-radius: 6px;")
        
        splash_p = QPixmap("splash_logo.png")
        if not splash_p.isNull():
            self.lbl_img_preview.setPixmap(splash_p.scaled(500, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.lbl_img_preview.setText("Live Visual Preview Area (Empty)")

        im_layout.addWidget(self.lbl_img_preview, 1)
        self.workspace_tabs.addTab(tab_img_merge, "Merge Image")

        # TAB 3: Merge PDF Workspace
        tab_pdf_merge = QWidget()
        pm_layout = QHBoxLayout(tab_pdf_merge)
        pm_layout.setContentsMargins(5, 5, 5, 5)

        pm_sidebar = QFrame()
        pm_sidebar.setFixedWidth(400)
        pm_sidebar.setStyleSheet("background-color: #1e2130; border-radius: 8px; padding: 8px;")
        pms_layout = QVBoxLayout(pm_sidebar)

        pms_layout.addWidget(QLabel("UNLIMITED PDF DOCUMENT MERGE", styleSheet="color: #00e676; font-size: 13px; font-weight: bold;"))
        btn_add_pdf_queue = QPushButton("＋ Add Current Canvas Crop to PDF Queue")
        btn_add_pdf_queue.setStyleSheet("background-color: #3d4465; color: white; padding: 10px; font-weight: bold;")
        btn_add_pdf_queue.clicked.connect(self.add_to_pdf_queue)
        pms_layout.addWidget(btn_add_pdf_queue)

        self.pdf_queue_list = QListWidget()
        self.pdf_queue_list.setStyleSheet("background-color: #12141d; color: white; font-size: 12px;")
        pms_layout.addWidget(self.pdf_queue_list)

        pdf_reorder_row = QHBoxLayout()
        btn_pdf_up = QPushButton("▲ Move Up")
        btn_pdf_up.setStyleSheet("background-color: #2b2f45; color: #00e676; font-weight: bold; padding: 4px;")
        btn_pdf_up.clicked.connect(lambda: self.move_queue_item(self.pdf_queue_list, self.pdf_merge_queue, -1, self.update_pdf_preview))
        
        btn_pdf_down = QPushButton("▼ Move Down")
        btn_pdf_down.setStyleSheet("background-color: #2b2f45; color: #00e676; font-weight: bold; padding: 4px;")
        btn_pdf_down.clicked.connect(lambda: self.move_queue_item(self.pdf_queue_list, self.pdf_merge_queue, 1, self.update_pdf_preview))
        
        btn_rem_pdf_queue = QPushButton("✕ Remove")
        btn_rem_pdf_queue.setStyleSheet("background-color: #d63031; color: white; font-weight: bold; padding: 4px;")
        btn_rem_pdf_queue.clicked.connect(lambda: self.remove_from_queue_list(self.pdf_queue_list, self.pdf_merge_queue, self.update_pdf_preview))
        
        pdf_reorder_row.addWidget(btn_pdf_up)
        pdf_reorder_row.addWidget(btn_pdf_down)
        pdf_reorder_row.addWidget(btn_rem_pdf_queue)
        pms_layout.addLayout(pdf_reorder_row)

        btn_export_pdf = QPushButton("💾 COMBINE & SAVE MULTI-PAGE PDF")
        btn_export_pdf.setMinimumHeight(45)
        btn_export_pdf.setStyleSheet("background-color: #0984e3; color: white; font-weight: bold; font-size: 13px; border-radius: 6px;")
        btn_export_pdf.clicked.connect(self.export_merged_pdf)
        pms_layout.addWidget(btn_export_pdf)
        pms_layout.addStretch()

        pm_scroll = QScrollArea()
        pm_scroll.setWidgetResizable(True)
        pm_scroll.setWidget(pm_sidebar)
        pm_scroll.setFixedWidth(420)
        pm_layout.addWidget(pm_scroll)

        self.pdf_preview_scroll = QScrollArea()
        self.pdf_preview_scroll.setWidgetResizable(True)
        self.pdf_preview_scroll.setStyleSheet("background-color: #0f111a; border: 2px dashed #3d4465; border-radius: 6px;")
        
        self.pdf_preview_content = QWidget()
        self.pdf_preview_vbox = QVBoxLayout(self.pdf_preview_content)
        self.pdf_preview_vbox.setAlignment(Qt.AlignCenter)
        self.pdf_preview_vbox.setSpacing(10)
        
        self.pdf_preview_placeholder = QLabel()
        self.pdf_preview_placeholder.setAlignment(Qt.AlignCenter)
        self.pdf_preview_placeholder.setStyleSheet("background: transparent;")
        if not splash_p.isNull():
            self.pdf_preview_placeholder.setPixmap(splash_p.scaled(500, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.pdf_preview_placeholder.setText("PDF Stack Preview Area (Empty)")
        self.pdf_preview_vbox.addWidget(self.pdf_preview_placeholder)
        
        self.pdf_preview_scroll.setWidget(self.pdf_preview_content)
        pm_layout.addWidget(self.pdf_preview_scroll, 1)

        self.workspace_tabs.addTab(tab_pdf_merge, "Merge PDF")
        main_vbox.addWidget(self.workspace_tabs)

    # ------------------ Handlers ------------------

    def on_slider_change(self):
        self.canvas.fine_angle = self.slider_angle.value()
        self.canvas.brightness = self.slider_bright.value()
        self.canvas.contrast = self.slider_contrast.value()
        self.canvas.sharpness = self.slider_sharp.value()
        self.lbl_angle_val.setText(f"{self.slider_angle.value()}°")
        self.canvas.apply_image_pipeline()

    def handle_bg_remove(self):
        if self.canvas.cv_raw is None:
            QMessageBox.warning(self, "No Image", "Load an image or PDF page first.")
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        success = self.canvas.remove_background()
        QApplication.restoreOverrideCursor()
        if success:
            QMessageBox.information(self, "Complete", "Background removed successfully.")

    def add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images or PDF Files", "",
            "Supported Files (*.png *.jpg *.jpeg *.bmp *.webp *.tiff *.pdf)"
        )
        for p in paths:
            self._import_path(p)

        if self.file_list.currentRow() == -1 and self.loaded_items:
            self.file_list.setCurrentRow(0)

    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".pdf"}
        for p in sorted(Path(folder).iterdir()):
            if p.suffix.lower() in exts:
                self._import_path(str(p))

        if self.file_list.currentRow() == -1 and self.loaded_items:
            self.file_list.setCurrentRow(0)

    def scan_from_device(self):
        try:
            import pythoncom
            from win32com.client import Dispatch
            pythoncom.CoInitialize()
            wia = Dispatch("WIA.CommonDialog")
            scanned_img_file = wia.ShowAcquireImage()
            
            temp_path = Path.cwd() / "scanned_temp_output.jpg"
            scanned_img_file.SaveFile(str(temp_path))
            self._import_path(str(temp_path))
            if self.file_list.currentRow() == -1 and self.loaded_items:
                self.file_list.setCurrentRow(0)
            QMessageBox.information(self, "Scanner", "Document successfully imported from scanner.")
        except Exception:
            path, _ = QFileDialog.getOpenFileName(self, "Select Scanned Document / Image", "", "Images (*.png *.jpg *.jpeg)")
            if path:
                self._import_path(path)
                if self.file_list.currentRow() == -1 and self.loaded_items:
                    self.file_list.setCurrentRow(0)

    def _import_path(self, path_str):
        p = Path(path_str)
        if p.suffix.lower() == ".pdf":
            try:
                pdf = pdfium.PdfDocument(path_str)
                for i in range(len(pdf)):
                    display_name = f"[PDF p.{i+1}] {p.name}"
                    self.loaded_items.append({"type": "pdf", "path": path_str, "page_idx": i, "label": display_name})
                    self.file_list.addItem(display_name)
            except Exception as e:
                QMessageBox.warning(self, "PDF Error", f"Could not load PDF: {e}")
        else:
            display_name = p.name
            self.loaded_items.append({"type": "img", "path": path_str, "page_idx": 0, "label": display_name})
            self.file_list.addItem(display_name)

    def remove_selected_item(self):
        row = self.file_list.currentRow()
        if not (0 <= row < len(self.loaded_items)):
            QMessageBox.warning(self, "Select Item", "Please select a file from the queue to remove.")
            return

        self.file_list.takeItem(row)
        self.loaded_items.pop(row)

        if not self.loaded_items:
            self.canvas.cv_raw = None
            self.canvas.cv_display = None
            self.canvas.pixmap_orig = None
            self.canvas.crop_rect = None
            self.canvas.update()
        else:
            new_row = min(row, len(self.loaded_items) - 1)
            self.file_list.setCurrentRow(new_row)

    def load_selected_item(self, row):
        if not (0 <= row < len(self.loaded_items)):
            return

        item = self.loaded_items[row]
        if item["type"] == "pdf":
            pdf = pdfium.PdfDocument(item["path"])
            page = pdf[item["page_idx"]]
            pil_img = page.render(scale=300/72).to_pil()
            self.canvas.load_pil_image(pil_img)
        else:
            with Image.open(item["path"]) as img:
                pil_img = ImageOps.exif_transpose(img).convert("RGB")
                self.canvas.load_pil_image(pil_img)

        self.slider_angle.setValue(0)
        self.slider_bright.setValue(0)
        self.slider_contrast.setValue(0)
        self.slider_sharp.setValue(0)
        self.on_preset_change()

    def add_to_image_queue(self):
        cropped = self.canvas.get_cropped_pil()
        if not cropped:
            QMessageBox.warning(self, "Warning", "Crop a document in the Editor tab first.")
            return
        final_img = self._get_final_sized_image(cropped)
        self.image_merge_queue.append(final_img)
        self.img_queue_list.addItem(f"Item #{len(self.image_merge_queue)} ({final_img.width}×{final_img.height} px)")
        self.update_image_preview()
        QMessageBox.information(self, "Added", "Document added to Image Merge Queue.")

    def add_to_pdf_queue(self):
        cropped = self.canvas.get_cropped_pil()
        if not cropped:
            QMessageBox.warning(self, "Warning", "Crop a document in the Editor tab first.")
            return
        final_img = self._get_final_sized_image(cropped)
        self.pdf_merge_queue.append(final_img)
        self.pdf_queue_list.addItem(f"Page #{len(self.pdf_merge_queue)} ({final_img.width}×{final_img.height} px)")
        self.update_pdf_preview()
        QMessageBox.information(self, "Added", "Page added to PDF Merge Queue.")

    def remove_from_queue_list(self, list_widget, queue_list, update_callback=None):
        row = list_widget.currentRow()
        if 0 <= row < len(queue_list):
            list_widget.takeItem(row)
            queue_list.pop(row)
            if update_callback:
                update_callback()

    def move_queue_item(self, list_widget, queue_list, direction, update_callback):
        row = list_widget.currentRow()
        new_row = row + direction
        if 0 <= row < len(queue_list) and 0 <= new_row < len(queue_list):
            queue_list[row], queue_list[new_row] = queue_list[new_row], queue_list[row]
            item_text = list_widget.takeItem(row)
            list_widget.insertItem(new_row, item_text)
            list_widget.setCurrentRow(new_row)
            if update_callback:
                update_callback()

    def _generate_merged_image_object(self):
        if not self.image_merge_queue:
            return None

        mode = self.combo_img_layout.currentText()
        first_w, first_h = self.image_merge_queue[0].size
        standardized = [img.resize((first_w, first_h), Image.Resampling.LANCZOS) for img in self.image_merge_queue]

        if "Side-by-Side" in mode:
            total_w = first_w * len(standardized) + (20 * (len(standardized) + 1))
            total_h = first_h + 40
            merged = Image.new("RGB", (total_w, total_h), (255, 255, 255))
            for idx, img in enumerate(standardized):
                merged.paste(img, (20 + idx * (first_w + 20), 20))
        elif "Stacked" in mode:
            total_w = first_w + 40
            total_h = first_h * len(standardized) + (20 * (len(standardized) + 1))
            merged = Image.new("RGB", (total_w, total_h), (255, 255, 255))
            for idx, img in enumerate(standardized):
                merged.paste(img, (20, 20 + idx * (first_h + 20)))
        else:
            merged = Image.new("RGB", (2480, 3508), (255, 255, 255))
            doc_w, doc_h = 1011, 638
            cols = 2
            start_x, start_y = 200, 300
            gap_x, gap_y = 150, 150

            for idx, img in enumerate(standardized):
                r = idx // cols
                c = idx % cols
                x = start_x + c * (doc_w + gap_x)
                y = start_y + r * (doc_h + gap_y)
                resized_doc = img.resize((doc_w, doc_h), Image.Resampling.LANCZOS)
                merged.paste(resized_doc, (x, y))

        return merged

    def update_image_preview(self):
        merged = self._generate_merged_image_object()
        if not merged:
            splash_p = QPixmap("splash_logo.png")
            if not splash_p.isNull():
                self.lbl_img_preview.setPixmap(splash_p.scaled(500, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                self.lbl_img_preview.setText("Live Visual Preview Area (Empty)")
                self.lbl_img_preview.setPixmap(QPixmap())
            return

        thumb = merged.copy()
        thumb.thumbnail((600, 400))
        rgb = np.array(thumb.convert("RGB"))
        h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.lbl_img_preview.setPixmap(QPixmap.fromImage(q_img))

    def update_pdf_preview(self):
        while self.pdf_preview_vbox.count():
            item = self.pdf_preview_vbox.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        if not self.pdf_merge_queue:
            self.pdf_preview_placeholder = QLabel()
            self.pdf_preview_placeholder.setAlignment(Qt.AlignCenter)
            self.pdf_preview_placeholder.setStyleSheet("background: transparent;")
            splash_p = QPixmap("splash_logo.png")
            if not splash_p.isNull():
                self.pdf_preview_placeholder.setPixmap(splash_p.scaled(500, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                self.pdf_preview_placeholder.setText("PDF Stack Preview Area (Empty)")
                self.pdf_preview_placeholder.setStyleSheet("color: #8890b5; background: transparent;")
            self.pdf_preview_vbox.addWidget(self.pdf_preview_placeholder)
            return

        for idx, page_img in enumerate(self.pdf_merge_queue):
            thumb = page_img.copy()
            thumb.thumbnail((450, 300))
            rgb = np.array(thumb.convert("RGB"))
            h, w, ch = rgb.shape
            q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)

            page_label = QLabel()
            page_label.setAlignment(Qt.AlignCenter)
            page_label.setPixmap(QPixmap.fromImage(q_img))
            page_label.setStyleSheet("background-color: #161822; border: 1px solid #3d4465; border-radius: 4px; padding: 4px;")
            
            caption = QLabel(f"Page #{idx + 1}")
            caption.setAlignment(Qt.AlignCenter)
            caption.setStyleSheet("color: #00e676; font-size: 10px; font-weight: bold; background: transparent;")

            self.pdf_preview_vbox.addWidget(caption)
            self.pdf_preview_vbox.addWidget(page_label)

    def export_merged_images(self):
        merged = self._generate_merged_image_object()
        if not merged:
            QMessageBox.warning(self, "Empty Queue", "Add items to the Image Merge queue first.")
            return

        chosen_dir = QFileDialog.getExistingDirectory(self, "Choose Saving Destination Folder")
        if not chosen_dir:
            return
        out_path, _ = QFileDialog.getSaveFileName(self, "Save Merged Images", str(Path(chosen_dir) / "commerge_Merged_Documents.png"), "PNG (*.png);;JPEG (*.jpg)")
        if out_path:
            if out_path.lower().endswith(".jpg"):
                merged.save(out_path, format="JPEG", quality=92, dpi=(300, 300))
            else:
                merged.save(out_path, format="PNG", dpi=(300, 300))
            QMessageBox.information(self, "Success", f"Merged document saved:\n{out_path}")

    def export_merged_pdf(self):
        if not self.pdf_merge_queue:
            QMessageBox.warning(self, "Empty Queue", "Add pages to the PDF Merge queue first.")
            return

        rgb_pages = [img.convert("RGB") for img in self.pdf_merge_queue]
        
        chosen_dir = QFileDialog.getExistingDirectory(self, "Choose Saving Destination Folder")
        if not chosen_dir:
            return
        out_path, _ = QFileDialog.getSaveFileName(self, "Save Multi-Page PDF", str(Path(chosen_dir) / "commerge_Combined_Document.pdf"), "PDF (*.pdf)")
        
        if out_path:
            rgb_pages[0].save(out_path, format="PDF", save_all=True, append_images=rgb_pages[1:], resolution=300.0)
            QMessageBox.information(self, "Success", f"Multi-page PDF saved successfully ({len(rgb_pages)} pages):\n{out_path}")

    def batch_extract_pdf_pages(self):
        row = self.file_list.currentRow()
        if not (0 <= row < len(self.loaded_items)):
            QMessageBox.warning(self, "Select PDF", "Please select a PDF file from the queue first.")
            return

        item = self.loaded_items[row]
        if item["type"] != "pdf":
            QMessageBox.warning(self, "Not a PDF", "The selected item is not a PDF document.")
            return

        pdf_path = item["path"]
        try:
            pdf = pdfium.PdfDocument(pdf_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open PDF: {e}")
            return

        chosen_dir = QFileDialog.getExistingDirectory(self, "Select Destination Folder for Extracted PDF Pages")
        if not chosen_dir:
            return
        out_dir = Path(chosen_dir) / f"{Path(pdf_path).stem}_Extracted_Pages"
        out_dir.mkdir(exist_ok=True)

        success_count = 0
        for i, page in enumerate(pdf):
            pil_page = page.render(scale=300/72).to_pil()
            rgb = np.array(pil_page.convert("RGB"))
            cv_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            box = None
            for c in contours:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4 and cv2.contourArea(c) > (gray.shape[0] * gray.shape[1] * 0.08):
                    box = cv2.boundingRect(approx)
                    break

            if box:
                bx, by, bw, bh = box
                cropped = pil_page.crop((bx, by, bx + bw, by + bh))
            else:
                cropped = pil_page

            final_img = self._get_final_sized_image(cropped)
            out_file = out_dir / f"page_{i+1}_extracted.png"
            final_img.save(out_file, format="PNG", dpi=(300, 300))
            success_count += 1

        QMessageBox.information(
            self, "Batch PDF Complete",
            f"Successfully extracted and auto-cropped {success_count} pages to:\n{out_dir}"
        )

    def handle_auto_detect(self):
        if not self.canvas.auto_detect_and_snap():
            QMessageBox.information(self, "Auto-Detect", "Could not find a distinct 4-corner document. Reset to preset frame.")

    def on_custom_px_change(self):
        tw = self.spin_px_w.value()
        th = self.spin_px_h.value()
        if th > 0:
            self.canvas.set_target_aspect(tw / th)

    def on_preset_change(self):
        preset_name = self.preset_box.currentText()
        preset = PRESETS.get(preset_name)

        if preset and preset.get("custom_px", False):
            self.grp_custom_px.show()
            self.orient_frame.hide()
            self.on_custom_px_change()
            return
        else:
            self.grp_custom_px.hide()
            self.orient_frame.show()

        if not preset or preset["w_mm"] is None:
            self.canvas.set_target_aspect(None)
            return

        w, h = preset["w_mm"], preset["h_mm"]
        ratio = min(w, h) / max(w, h) if self.btn_port.isChecked() else max(w, h) / min(w, h)
        self.canvas.set_target_aspect(ratio)

    def _execute_print_dialog(self, pil_image):
        printer = QPrinter(QPrinter.HighResolution)
        dialog = QPrintDialog(printer, self)

        if dialog.exec() == QPrintDialog.Accepted:
            rgb = np.array(pil_image.convert("RGB"))
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

            painter = QPainter(printer)
            rect = printer.pageRect(QPrinter.DevicePixel)

            scaled_img = q_img.scaled(rect.width(), rect.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = (rect.width() - scaled_img.width()) // 2
            y = (rect.height() - scaled_img.height()) // 2

            painter.drawImage(x, y, scaled_img)
            painter.end()
            QMessageBox.information(self, "Printing", "Sent to printer successfully.")

    def print_current_card(self):
        cropped = self.canvas.get_cropped_pil()
        if not cropped:
            QMessageBox.warning(self, "No Crop", "Crop a document first.")
            return
        final_img = self._get_final_sized_image(cropped)
        self._execute_print_dialog(final_img)

    def _compress_jpeg_to_kb(self, pil_img, target_kb):
        target_bytes = target_kb * 1024
        low, high = 5, 95
        best_data = None

        if pil_img.mode in ("RGBA", "P"):
            pil_img = pil_img.convert("RGB")

        for _ in range(7):
            mid = (low + high) // 2
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=mid, optimize=True, dpi=(300, 300))
            if buf.tell() <= target_bytes:
                best_data = buf.getvalue()
                low = mid + 1
            else:
                high = mid - 1

        if best_data is None:
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=5, optimize=True)
            best_data = buf.getvalue()

        return best_data

    def _compress_pdf_to_kb(self, pil_img, target_kb):
        target_bytes = target_kb * 1024
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        low, high = 5, 90
        best_pdf = None

        for _ in range(6):
            mid = (low + high) // 2
            jpg_buf = io.BytesIO()
            pil_img.save(jpg_buf, format="JPEG", quality=mid, optimize=True, dpi=(300, 300))
            jpg_buf.seek(0)

            pdf_buf = io.BytesIO()
            with Image.open(jpg_buf) as temp_img:
                temp_img.save(pdf_buf, format="PDF", resolution=300.0)

            if pdf_buf.tell() <= target_bytes:
                best_pdf = pdf_buf.getvalue()
                low = mid + 1
            else:
                high = mid - 1

        if best_pdf is None:
            pdf_buf = io.BytesIO()
            pil_img.save(pdf_buf, format="PDF", resolution=150.0)
            best_pdf = pdf_buf.getvalue()

        return best_pdf

    def _get_final_sized_image(self, cropped):
        preset_name = self.preset_box.currentText()
        preset = PRESETS.get(preset_name)

        if preset and preset.get("custom_px", False):
            target_w = self.spin_px_w.value()
            target_h = self.spin_px_h.value()
            return cropped.resize((target_w, target_h), Image.Resampling.LANCZOS)
        elif preset and preset["w_mm"]:
            w, h = preset["w_mm"], preset["h_mm"]
            if self.btn_port.isChecked():
                tw = int((min(w, h) / 25.4) * preset["dpi"])
                th = int((max(w, h) / 25.4) * preset["dpi"])
            else:
                tw = int((max(w, h) / 25.4) * preset["dpi"])
                th = int((min(w, h) / 25.4) * preset["dpi"])
            return cropped.resize((tw, th), Image.Resampling.LANCZOS)
        return cropped

    def export_single_processed(self):
        cropped = self.canvas.get_cropped_pil()
        if not cropped:
            QMessageBox.warning(self, "Notice", "Select an image/page and crop area first.")
            return

        final_img = self._get_final_sized_image(cropped)
        row = self.file_list.currentRow()
        if not (0 <= row < len(self.loaded_items)):
            QMessageBox.warning(self, "Notice", "Please select a file from the queue.")
            return

        item = self.loaded_items[row]
        src_path = Path(item["path"])

        chosen_dir = QFileDialog.getExistingDirectory(self, "Choose Saving Destination Folder")
        if not chosen_dir:
            return
        out_dir = Path(chosen_dir)

        selected_fmt = self.combo_format.currentText()
        target_kb = self.spin_target_kb.value()
        apply_limit = self.chk_limit_size.isChecked()

        stem = f"{src_path.stem}_p{item['page_idx']+1}" if item["type"] == "pdf" else src_path.stem

        if "PDF" in selected_fmt:
            out_file = out_dir / f"{stem}_doc.pdf"
            if apply_limit:
                out_file.write_bytes(self._compress_pdf_to_kb(final_img, target_kb))
            else:
                if final_img.mode != "RGB":
                    final_img = final_img.convert("RGB")
                final_img.save(out_file, format="PDF", resolution=300.0)
        elif "JPEG" in selected_fmt:
            out_file = out_dir / f"{stem}_output.jpg"
            if apply_limit:
                out_file.write_bytes(self._compress_jpeg_to_kb(final_img, target_kb))
            else:
                if final_img.mode in ("RGBA", "P"):
                    final_img = final_img.convert("RGB")
                final_img.save(out_file, format="JPEG", quality=92, dpi=(300, 300))
        else:
            out_file = out_dir / f"{stem}_300dpi.png"
            final_img.save(out_file, format="PNG", dpi=(300, 300))

        msg_extra = ""
        if self.chk_simul_export.isChecked():
            web_file = out_dir / f"{stem}_web_under100kb.jpg"
            web_bytes = self._compress_jpeg_to_kb(final_img, 95)
            web_file.write_bytes(web_bytes)
            msg_extra = f"\n+ Web Upload Copy: {web_file.name} ({len(web_bytes)/1024:.1f} KB)"

        actual_size_kb = out_file.stat().st_size / 1024
        QMessageBox.information(
            self, "Export Complete",
            f"Saved to:\n{out_file}\n\nFinal Size: {actual_size_kb:.1f} KB{msg_extra}"
        )


if __name__ == "__main__":
    import ctypes
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("org.document.commerge.master.1-0")
    except Exception:
        pass

    app = QApplication(sys.argv)
    window = CommergeMainWindow()
    window.show()
    sys.exit(app.exec())