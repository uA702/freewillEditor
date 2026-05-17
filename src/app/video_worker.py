import cv2
import time
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

class VideoWorker(QThread):
    frame_processed = Signal(QImage)
    export_progress = Signal(int, int)
    export_finished = Signal(bool)
    video_ended = Signal()

    def __init__(self, registry):
        super().__init__()
        self.registry = registry
        self.active_stage = list(registry.stages.keys())[0] if registry.stages else ""
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
        self.running = True
        frame_time = 1000.0 / self.fps

        while self.running and self.cap and self.cap.isOpened():
            start_time = time.perf_counter_ns()
            ret, frame = self.cap.read()
            if not ret:
                self.video_ended.emit()
                self.running = False
                break

            # Route the frame through the globally active stage layout
            processed = self.registry.execute_graph(frame)

            rgb_frame = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.frame_processed.emit(q_img.copy())

            elapsed_ms = (time.perf_counter_ns() - start_time) / 1000000
            self.msleep(int(max(1, frame_time - elapsed_ms)))

    def stop(self):
        self.running = False
        self.wait()

    def restart(self):
        self.registry.clear_history()
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def export_video(self, output_path):
        """Runs an offline export sequence using the currently active stage."""
        if not self.video_path or not output_path or not self.active_stage:
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
            
            # Use the active stage for export math routing
            processed = self.registry.process_stage(self.active_stage, frame)
            writer.write(processed)
            
            frame_count += 1
            self.export_progress.emit(frame_count, total_frames)

        reader.release()
        writer.release()
        self.export_finished.emit(True)

    def __del__(self):
        if self.cap:
            self.cap.release()