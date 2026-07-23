"""
gui.py
======

Live costume-recognition GUI (PyQt6) for the ZED X on the Jetson.

What you see
-----------
  * Left  : live camera feed with GREEN bounding boxes drawn around every
            detected person, updated continuously.
  * Bottom: a "Capture" button.
  * Right : after you click Capture, one card per person -- the crop, the
            top-1 character in bold, and the top-3 with confidence bars.

Data flow
---------
    CameraThread (background)                    Main / GUI thread
    ------------------------                     -----------------
    session.grab() ---- frameReady(rgb, dets) -> draw feed + boxes, remember
                                                 the latest (frame, dets)

    [click Capture] --------------------------->  ClassifyThread(latest dets)
    ClassifyThread (background)
    --------------------------
    clf.classify_batch(crops) -- resultsReady --> build the result cards

Two background threads keep the UI responsive: one never-ending camera loop,
and a short-lived classification job spawned per capture. The heavy model load
happens once at startup (see main()).

Design choices (agreed up front)
--------------------------------
  * Live detection overlay: object detection runs every preview frame so the
    boxes track in real time. It runs on the ZED's GPU pipeline, so it's cheap.
  * CLIP classification runs ONLY on capture -- it's the expensive part, and
    you only need a label when you ask for one.
"""

from __future__ import annotations

import pathlib
import sys
from typing import List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from camera_session import CameraSession, Detection

from clip_classifier import ClipCostumeClassifier  # noqa: E402

CHECKPOINT = str("40c_epoch8.pt") # TODO: make sure this file is in the same folder

BOX_COLOR = QColor(0, 200, 120)   # green boxes


# ─────────────────────────────────────────────────────────────────────────────
# Background workers
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: numpy/PIL -> Qt
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# One result card (crop + top-3 labels with confidence bars)
# ─────────────────────────────────────────────────────────────────────────────
class ResultCard(QFrame):
    def __init__(self, index: int, det: Detection, preds: List[dict]):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        row = QHBoxLayout(self)

        # crop thumbnail
        thumb = QLabel()
        thumb.setPixmap(
            pil_to_qpixmap(det.crop).scaled(
                120, 200,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        row.addWidget(thumb)

        # labels + confidence bars
        col = QVBoxLayout()
        top = preds[0] if preds else {"label": "—", "score": 0.0}
        header = QLabel(f"Person {index}:  {top['label']}")
        f = QFont()
        f.setPointSize(13)
        f.setBold(True)
        header.setFont(f)
        col.addWidget(header)

        for p in preds:
            line = QHBoxLayout()
            name = QLabel(p["label"])
            name.setMinimumWidth(220)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(int(round(p["score"])))
            bar.setFormat(f"{p['score']}%")
            line.addWidget(name)
            line.addWidget(bar)
            col.addLayout(line)

        row.addLayout(col)


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────
class MainWindow(QWidget):
    def __init__(self, session: CameraSession, clf: ClipCostumeClassifier):
        super().__init__()
        self.session = session
        self.clf = clf
        self.latest: Optional[Tuple[np.ndarray, List[Detection]]] = None
        self.classify_thread: Optional[ClassifyThread] = None

        self.setWindowTitle("Costume Recognition — ZED X Live Demo")
        self.resize(1400, 800)

        root = QHBoxLayout(self)

        # ---- left: live feed + capture button ----
        left = QVBoxLayout()
        self.video = QLabel("Starting camera…")
        self.video.setMinimumSize(900, 700)
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setStyleSheet("background:#111; color:#aaa;")
        left.addWidget(self.video, stretch=1)

        self.capture_btn = QPushButton("Capture")
        self.capture_btn.setMinimumHeight(48)
        self.capture_btn.clicked.connect(self.on_capture)
        left.addWidget(self.capture_btn)
        root.addLayout(left, stretch=3)

        # ---- right: results panel (scrollable) ----
        self.results_box = QVBoxLayout()
        self.results_box.addStretch(1)
        results_container = QWidget()
        results_container.setLayout(self.results_box)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(results_container)
        scroll.setMinimumWidth(430)
        root.addWidget(scroll, stretch=2)

        # ---- start the camera thread ----
        self.cam_thread = CameraThread(session)
        self.cam_thread.frameReady.connect(self.on_frame)
        self.cam_thread.error.connect(self.on_cam_error)
        self.cam_thread.start()

    # ---- live preview ----
    def on_frame(self, rgb: np.ndarray, dets: List[Detection]):
        self.latest = (rgb, dets)
        pix = rgb_to_qpixmap(rgb)

        # draw boxes at full resolution, then scale to the label
        painter = QPainter(pix)
        painter.setPen(QPen(BOX_COLOR, 3))
        font = QFont()
        font.setPointSize(16)
        painter.setFont(font)
        for i, d in enumerate(dets):
            x1, y1, x2, y2 = d.box
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
            painter.drawText(x1 + 5, max(0, y1 - 8), f"{i}  ({d.score:.0f})")
        painter.end()

        self.video.setPixmap(
            pix.scaled(
                self.video.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def on_cam_error(self, msg: str):
        self.video.setText(f"Camera error:\n{msg}")

    # ---- capture -> classify ----
    def on_capture(self):
        if not self.latest:
            return
        _, dets = self.latest
        self._clear_results()
        if not dets:
            self._add_placeholder("No people detected in frame.")
            return

        self.capture_btn.setEnabled(False)
        self.capture_btn.setText("Classifying…")

        # classify the CURRENT detections off the UI thread
        self.classify_thread = ClassifyThread(self.clf, list(dets))
        self.classify_thread.resultsReady.connect(self.on_results)
        self.classify_thread.start()

    def on_results(self, paired: List[Tuple[Detection, List[dict]]]):
        self._clear_results()
        for i, (det, preds) in enumerate(paired):
            self.results_box.insertWidget(self.results_box.count() - 1,
                                          ResultCard(i, det, preds))
        self.capture_btn.setEnabled(True)
        self.capture_btn.setText("Capture")

    # ---- results panel helpers ----
    def _clear_results(self):
        # remove everything except the trailing stretch
        while self.results_box.count() > 1:
            item = self.results_box.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def _add_placeholder(self, text: str):
        lbl = QLabel(text)
        lbl.setStyleSheet("color:#888; padding:12px;")
        self.results_box.insertWidget(0, lbl)

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
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
