"""
gui3.py
=======

Live costume-recognition GUI (PyQt6) for the ZED X.

"""

from __future__ import annotations

import sys
from typing import List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from camera_session import CameraSession, Detection
from clip_classifier import ClipCostumeClassifier  # noqa: E402

CHECKPOINT = str("40c_epoch8.pt")  # TODO: make sure this file is in the same folder

BOX_COLOR = QColor(0, 200, 120)          # green boxes
BOX_COLOR_HOVER = QColor(80, 230, 160)   # brighter on hover
BOX_COLOR_SELECTED = QColor(255, 200, 60)  # box whose details are open
REJECT_COLOR = QColor(235, 120, 120)     # below-threshold label color

# Background workers

class CameraThread(QThread):
    """Continuously grabs frames + detections and emits them to the GUI."""
    frameReady = pyqtSignal(object, object)   # (rgb: np.ndarray, dets: list[Detection])
    error = pyqtSignal(str)

    def __init__(self, session: CameraSession):
        super().__init__()
        self.session = session
        self._running = True

    def run(self):
        while self._running:
            try:
                rgb, dets = self.session.grab()
                self.frameReady.emit(rgb, dets)
            except Exception as e:  # keep the app alive, report once
                self.error.emit(str(e))
                break

    def stop(self):
        self._running = False
        self.wait()


class ClassifyThread(QThread):
    """Runs the OpenCLIP classifier on a captured set of crops (one-shot)."""
    resultsReady = pyqtSignal(object)   # list[(Detection, preds)]

    def __init__(self, clf: ClipCostumeClassifier, detections: List[Detection]):
        super().__init__()
        self.clf = clf
        self.detections = detections

    def run(self):
        crops = [d.crop for d in self.detections]
        preds = self.clf.classify_batch(crops, k=3) if crops else []
        self.resultsReady.emit(list(zip(self.detections, preds)))

# Helpers
def rgb_to_qpixmap(rgb: np.ndarray) -> QPixmap:
    h, w, _ = rgb.shape
    img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img.copy())   # copy so it owns its buffer


def pil_to_qpixmap(im) -> QPixmap:
    im = im.convert("RGB")
    w, h = im.size
    data = im.tobytes("raw", "RGB")
    img = QImage(data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img.copy())


def _label_text(top: dict, threshold: int, show_raw: bool) -> Tuple[str, bool]:
    """Returns (text, rejected)."""
    rejected = top["score"] < threshold
    if rejected:
        return "Not recognized", True
    val = f"{top.get('sim', 0.0):.3f}" if show_raw else f"{top['score']}%"
    return f"{top['label']}  ({val})", False


# Detail sidebar: opened on click, full top-3 breakdown for one person
SIDEBAR_QSS = """
QFrame#detailSidebar {
    background: #232323;
    border: 1px solid #3d3d3d;
    border-radius: 10px;
}
/* Every specific rule is prefixed with the frame id: a bare
   `QLabel#sidebarSubtitle` would LOSE to `QFrame#detailSidebar QLabel`
   under CSS specificity (same id count, fewer element names) and every
   label would come out white. */
QFrame#detailSidebar QLabel { color: #ffffff; background: transparent; }
QFrame#detailSidebar QLabel#sidebarTitle    { color: #ffffff; font-size: 14px; font-weight: bold; }
QFrame#detailSidebar QLabel#sidebarSubtitle { color: #b8b8b8; font-size: 11px; }
QFrame#detailSidebar QLabel#sidebarHeadline { color: #ffffff; font-size: 17px; font-weight: bold; }
QFrame#detailSidebar QLabel#sidebarRejected { color: #eb7878; font-size: 17px; font-weight: bold; }
QFrame#detailSidebar QLabel#sidebarHint     { color: #9a9a9a; font-size: 12px; }
QFrame#detailSidebar QPushButton#sidebarClose {
    color: #ffffff; background: #3a3a3a; border: 1px solid #555;
    border-radius: 11px; font-weight: bold; padding: 0px;
}
QFrame#detailSidebar QPushButton#sidebarClose:hover { background: #4d4d4d; }
QProgressBar {
    background: #2c2c2c; border: 1px solid #4a4a4a; border-radius: 7px;
    text-align: center; color: #ffffff;
    min-height: 18px; max-height: 18px;   /* plain `height` is ignored by Qt */
}
QProgressBar::chunk { background: #2d9d68; border-radius: 6px; }
QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; border: none; }
"""


class DetailSidebar(QFrame):
    """Right-hand panel with the crop + the full top-3 for one detection."""

    closed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("detailSidebar")
        self.setStyleSheet(SIDEBAR_QSS)
        self.setFixedWidth(360)
        self.setVisible(False)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 14)
        outer.setSpacing(10)

        # ---- header row: title + close button ----
        head = QHBoxLayout()
        self.title = QLabel("Detection details")
        self.title.setObjectName("sidebarTitle")
        head.addWidget(self.title)
        head.addStretch(1)
        close = QPushButton("✕")
        close.setObjectName("sidebarClose")
        close.setFixedSize(22, 22)
        close.clicked.connect(self._on_close)
        head.addWidget(close)
        outer.addLayout(head)

        self.subtitle = QLabel("")
        self.subtitle.setObjectName("sidebarSubtitle")
        outer.addWidget(self.subtitle)

        # ---- crop thumbnail ----
        self.thumb = QLabel()
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb.setMinimumHeight(230)
        outer.addWidget(self.thumb)

        # ---- top-1 headline ----
        self.headline = QLabel("")
        self.headline.setObjectName("sidebarHeadline")
        self.headline.setWordWrap(True)
        outer.addWidget(self.headline)

        # ---- top-3 rows (scrollable, in case the label set grows) ----
        self._rows_host = QWidget()
        self.rows = QVBoxLayout(self._rows_host)
        self.rows.setContentsMargins(0, 0, 0, 0)
        self.rows.setSpacing(6)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._rows_host)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll, stretch=1)

        self.hint = QLabel("Click another box to switch, or ✕ to close.")
        self.hint.setObjectName("sidebarHint")
        self.hint.setWordWrap(True)
        outer.addWidget(self.hint)

    # ---- content ----
    def show_detection(self, index: int, det: Detection, preds: List[dict],
                       show_raw: bool, threshold: int):
        self.subtitle.setText(
            f"Person #{index + 1}  ·  detector confidence {det.score:.0f}%"
        )
        self.thumb.setPixmap(
            pil_to_qpixmap(det.crop).scaled(
                200, 300, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

        top = preds[0] if preds else {"label": "—", "score": 0.0, "sim": 0.0}
        rejected = top["score"] < threshold
        name = "sidebarRejected" if rejected else "sidebarHeadline"
        if self.headline.objectName() != name:
            self.headline.setObjectName(name)
            self.setStyleSheet(SIDEBAR_QSS)
        self.headline.setText("Not recognized as a costume" if rejected else top["label"])

        self._clear_rows()
        top_sim = max((p.get("sim", 0.0) for p in preds), default=1.0) or 1.0
        for p in preds:
            line = QHBoxLayout()
            line.setSpacing(8)
            name = QLabel(p["label"])
            name.setMinimumWidth(130)
            name.setWordWrap(True)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setTextVisible(True)
            if show_raw:
                sim = p.get("sim", 0.0)
                bar.setValue(0)
                bar.setFormat(f"{sim:.3f}")
            else:
                bar.setValue(int(round(p["score"])))
                bar.setFormat(f"{p['score']}%")
            line.addWidget(name)
            line.addWidget(bar, stretch=1)
            self.rows.addLayout(line)
        self.rows.addStretch(1)

        self.setVisible(True)

    def _clear_rows(self):
        while self.rows.count():
            item = self.rows.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
                continue
            sub = item.layout()
            if sub is not None:
                while sub.count():
                    sub_item = sub.takeAt(0)
                    sw = sub_item.widget()
                    if sw is not None:
                        sw.setParent(None)
                        sw.deleteLater()
                sub.deleteLater()

    def _on_close(self):
        self.setVisible(False)
        self.closed.emit()


# Live / frozen camera view with in-image labels
class LiveView(QWidget):
    boxClicked = pyqtSignal(int)   # detection index (the sidebar needs nothing else)

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setMinimumSize(640, 400)
        self.setStyleSheet("background:#111;")

        self.pixmap: Optional[QPixmap] = None
        self.dets: List[Detection] = []
        self.preds: List[List[dict]] = []   # empty until classified

        self.frozen = False
        self.show_raw = False
        self.threshold = 50
        self.hover_index: Optional[int] = None
        self.selected_index: Optional[int] = None   # details open in the sidebar

        self._label_rects: List[QRectF] = []   # screen-space, box + label bar

    # ---- state updates from outside ----
    def set_frame(self, rgb: np.ndarray, dets: List[Detection]):
        if self.frozen:
            return
        self.pixmap = rgb_to_qpixmap(rgb)
        self.dets = dets
        self.preds = []
        self.update()

    def freeze_with_results(self, paired: List[Tuple[Detection, List[dict]]]):
        self.frozen = True
        self.dets = [d for d, _ in paired]
        self.preds = [p for _, p in paired]
        self.hover_index = None
        self.selected_index = None
        self.update()

    def unfreeze(self):
        self.frozen = False
        self.preds = []
        self.hover_index = None
        self.selected_index = None
        self.update()

    def set_display_opts(self, show_raw: bool, threshold: int):
        self.show_raw = show_raw
        self.threshold = threshold
        self.update()

    def set_selected(self, index: Optional[int]):
        self.selected_index = index
        self.update()

    # ---- painting ----
    def paintEvent(self, event):
        painter = QPainter(self)
        if not self.pixmap:
            painter.setPen(QColor(170, 170, 170))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Starting camera…")
            return

        scaled = self.pixmap.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x0 = (self.width() - scaled.width()) / 2
        y0 = (self.height() - scaled.height()) / 2
        painter.drawPixmap(int(x0), int(y0), scaled)

        sx = scaled.width() / self.pixmap.width()
        sy = scaled.height() / self.pixmap.height()
        img_right = x0 + scaled.width()

        self._label_rects = []
        label_font = QFont()
        label_font.setPointSize(11)
        label_font.setBold(True)

        for i, d in enumerate(self.dets):
            x1, y1, x2, y2 = d.box
            box = QRectF(x0 + x1 * sx, y0 + y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy)

            hovered = (i == self.hover_index)
            selected = (i == self.selected_index)
            if selected:
                pen_color, pen_w = BOX_COLOR_SELECTED, 3
            elif hovered:
                pen_color, pen_w = BOX_COLOR_HOVER, 3
            else:
                pen_color, pen_w = BOX_COLOR, 2
            painter.setPen(QPen(pen_color, pen_w))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(box)

            combined = QRectF(box)
            if i < len(self.preds) and self.preds[i]:
                top = self.preds[i][0]
                text, rejected = _label_text(top, self.threshold, self.show_raw)

                painter.setFont(label_font)
                fm = painter.fontMetrics()
                bar_h = min(fm.height() + 8, max(box.height(), 1.0))
                
                bar_w = max(box.width(), fm.horizontalAdvance(text) + 16)
                bar_x = box.left()
                if bar_x + bar_w > img_right:          # keep it on the image
                    bar_x = max(x0, img_right - bar_w)
                bar_rect = QRectF(bar_x, box.bottom() - bar_h, bar_w, bar_h)

                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(0, 0, 0, 205))
                painter.drawRect(bar_rect)
                painter.setBrush(pen_color)
                painter.drawRect(QRectF(bar_rect.left(), bar_rect.top(), bar_rect.width(), 2))

                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(REJECT_COLOR if rejected else QColor(255, 255, 255))
                painter.drawText(bar_rect, Qt.AlignmentFlag.AlignCenter, text)
                combined = box.united(bar_rect)

            self._label_rects.append(combined)

    def _index_at(self, pos: QPointF) -> Optional[int]:
        best: Optional[int] = None
        best_area = None
        for i, r in enumerate(self._label_rects):
            if r.contains(pos):
                area = r.width() * r.height()
                if best_area is None or area < best_area:
                    best, best_area = i, area
        return best

    def mouseMoveEvent(self, event):
        idx = self._index_at(event.position())
        if idx != self.hover_index:
            self.hover_index = idx
            self.update()

    def leaveEvent(self, event):
        self.hover_index = None
        self.update()

    def mousePressEvent(self, event):
        idx = self._index_at(event.position())
        if idx is not None and idx < len(self.preds) and self.preds[idx]:
            self.boxClicked.emit(idx)


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────
MODE_TOGGLE_QSS = """
QPushButton#modeLeft, QPushButton#modeRight {
    background: #3a3a3a;
    color: #ffffff;
    border: 1px solid #5a5a5a;
    padding: 5px 18px;
}
QPushButton#modeLeft {
    border-top-left-radius: 15px; border-bottom-left-radius: 15px;
    border-top-right-radius: 0px; border-bottom-right-radius: 0px;
}
QPushButton#modeRight {
    border-top-right-radius: 15px; border-bottom-right-radius: 15px;
    border-top-left-radius: 0px; border-bottom-left-radius: 0px;
    border-left: none;
}
QPushButton#modeLeft:hover, QPushButton#modeRight:hover { background: #4a4a4a; }
QPushButton#modeLeft:checked, QPushButton#modeRight:checked {
    background: #2d9d68; color: #ffffff; font-weight: bold;
}
"""


class MainWindow(QWidget):
    def __init__(self, session: CameraSession, clf: ClipCostumeClassifier):
        super().__init__()
        self.session = session
        self.clf = clf
        self.latest: Optional[Tuple[np.ndarray, List[Detection]]] = None
        self.classify_thread: Optional[ClassifyThread] = None
        self.detail_index: Optional[int] = None   # which box the sidebar shows

        self.setWindowTitle("Costume Recognition — ZED X Live Demo")
        self.resize(1200, 800)

        root = QVBoxLayout(self)

        # ---- top: display options ----
        opts = QHBoxLayout()
        opts.addWidget(self._build_mode_toggle())
        opts.addStretch(1)
        opts.addWidget(QLabel("Costume threshold:"))
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(50)
        self.threshold_slider.setMinimumWidth(140)
        self.threshold_slider.valueChanged.connect(self._on_display_opts_changed)
        opts.addWidget(self.threshold_slider)
        self.threshold_label = QLabel("50%")
        self.threshold_label.setMinimumWidth(40)
        opts.addWidget(self.threshold_label)
        root.addLayout(opts)

        # ---- center: live / frozen view + detail sidebar ----
        center = QHBoxLayout()
        center.setSpacing(10)
        self.view = LiveView()
        self.view.boxClicked.connect(self.on_box_clicked)
        center.addWidget(self.view, stretch=1)

        self.sidebar = DetailSidebar()
        self.sidebar.closed.connect(self.on_sidebar_closed)
        center.addWidget(self.sidebar)
        root.addLayout(center, stretch=1)

        # ---- bottom: capture / live toggle ----
        self.capture_btn = QPushButton("Capture")
        self.capture_btn.setMinimumHeight(48)
        self.capture_btn.clicked.connect(self.on_capture_button)
        root.addWidget(self.capture_btn)

        # ---- start the camera thread ----
        self.cam_thread = CameraThread(session)
        self.cam_thread.frameReady.connect(self.on_frame)
        self.cam_thread.error.connect(self.on_cam_error)
        self.cam_thread.start()

    # ---- display-mode toggle (segmented two-button control) ----
    def _build_mode_toggle(self) -> QWidget:
        """A two-sided dark pill: 'Percentage' | 'Similarity'."""
        box = QWidget()
        lay = QHBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self.btn_pct = QPushButton("Percentage")
        self.btn_pct.setObjectName("modeLeft")
        self.btn_sim = QPushButton("Similarity")
        self.btn_sim.setObjectName("modeRight")
        self.btn_sim.setToolTip("Show raw cosine similarity instead of softmax %")

        self.mode_group = QButtonGroup(self)
        for b in (self.btn_pct, self.btn_sim):
            b.setCheckable(True)
            b.setMinimumHeight(30)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            self.mode_group.addButton(b)
            lay.addWidget(b)
        self.btn_pct.setChecked(True)   # default: percentages
        self.mode_group.setExclusive(True)
        self.mode_group.buttonToggled.connect(self._on_display_opts_changed)

        box.setStyleSheet(MODE_TOGGLE_QSS)
        return box

    def _show_raw(self) -> bool:
        return self.btn_sim.isChecked()

    def _on_display_opts_changed(self, *args):
        self.threshold_label.setText(f"{self.threshold_slider.value()}%")
        self.view.set_display_opts(self._show_raw(), self.threshold_slider.value())
        self._refresh_sidebar()

    # ---- live preview ----
    def on_frame(self, rgb: np.ndarray, dets: List[Detection]):
        self.latest = (rgb, dets)   # kept fresh even while frozen, for the next capture
        self.view.set_frame(rgb, dets)

    def on_cam_error(self, msg: str):
        self.capture_btn.setText("Camera error")
        self.capture_btn.setToolTip(msg)   # the message was being thrown away
        self.capture_btn.setEnabled(False)

    # ---- capture / live toggle ----
    def on_capture_button(self):
        if self.view.frozen:
            self.view.unfreeze()
            self._close_sidebar()
            self.capture_btn.setText("Capture")
            return

        if not self.latest:
            return
        rgb, dets = self.latest
        if not dets:
            # used to fail silently; say something and stay live
            self.capture_btn.setText("No people detected")
            QTimer.singleShot(1200, lambda: self.capture_btn.setText(
                "Live" if self.view.frozen else "Capture"))
            return

        self.view.set_frame(rgb, dets)   # make sure the frame about to freeze is current
        self.view.frozen = True          # freeze immediately for visual feedback
        self._close_sidebar()
        self.capture_btn.setEnabled(False)
        self.capture_btn.setText("Classifying…")

        self.classify_thread = ClassifyThread(self.clf, list(dets))
        self.classify_thread.resultsReady.connect(self.on_results)
        self.classify_thread.start()

    def on_results(self, paired: List[Tuple[Detection, List[dict]]]):
        self.view.freeze_with_results(paired)
        self.view.set_display_opts(self._show_raw(), self.threshold_slider.value())
        self.capture_btn.setEnabled(True)
        self.capture_btn.setText("Live")

    # ---- detail sidebar ----
    def on_box_clicked(self, index: int):
        self.detail_index = index
        self.view.set_selected(index)
        self._refresh_sidebar()

    def _refresh_sidebar(self):
        i = self.detail_index
        if i is None or i >= len(self.view.dets) or i >= len(self.view.preds):
            return
        self.sidebar.show_detection(
            i, self.view.dets[i], self.view.preds[i],
            self._show_raw(), self.threshold_slider.value(),
        )

    def _close_sidebar(self):
        self.detail_index = None
        self.view.set_selected(None)
        self.sidebar.setVisible(False)

    def on_sidebar_closed(self):
        self.detail_index = None
        self.view.set_selected(None)

    # ---- shutdown ----
    def closeEvent(self, event):
        self.cam_thread.stop()
        if self.classify_thread:
            self.classify_thread.wait()
        self.session.close()
        super().closeEvent(event)


def main():
    print("Opening ZED camera…")
    session = CameraSession(conf=40, pad=0.08)
    # Heavy one-time setup BEFORE the window opens:

    print("Loading OpenCLIP model (this takes a few seconds)…")
    clf = ClipCostumeClassifier(checkpoint=CHECKPOINT)

    app = QApplication(sys.argv)
    win = MainWindow(session, clf)
    win.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
