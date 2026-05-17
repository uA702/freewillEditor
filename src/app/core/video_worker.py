import cv2
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
import numpy as np

class VideoWorker(QThread):
    # Signals to communicate safely with the Main UI Thread
    frame_processed = Signal(QImage)
    export_progress = Signal(int, int)
    export_finished = Signal(bool)
    video_ended = Signal()

    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline
        self.video_path = ""
        self.cap = None
        self.running = False
        self.fps = 30

    def load_video(self, path):
        self.video_path = path
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        if self.cap.isOpened():
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
            return True
        return False

    def run(self):
        """The main playback loop running entirely inside the background thread."""
        self.running = True
        delay = int(1000 / self.fps)

        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.video_ended.emit()
                self.running = False
                break

            # 1. Process via NumPy Pipeline
            processed = self.pipeline.process(frame)

            # 2. Convert from BGR to RGB
            rgb_frame = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w

            # 3. Create a QImage and emit it safely to the UI
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # .copy() prevents memory corruption since numpy arrays change rapidly in memory
            self.frame_processed.emit(q_img.copy()) 

            #self.msleep(delay)

    def stop(self):
        self.running = False
        self.wait()

    def restart(self):
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def export_video(self, output_path):
        """Runs an offline export sequence without locking up the user interface."""
        if not self.video_path or not output_path:
            self.export_finished.emit(False)
            return

        reader = cv2.VideoCapture(self.video_path)
        if not reader.isOpened():
            self.export_finished.emit(False)
            return

        width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = reader.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = reader.read()
            if not ret:
                break
            
            processed = self.pipeline.process(frame)
            writer.write(processed)
            
            frame_count += 1
            self.export_progress.emit(frame_count, total_frames)

        reader.release()
        writer.release()
        self.export_finished.emit(True)

    def __del__(self):
        if self.cap:
            self.cap.release()