import cv2
import time
import os
from moviepy.video.io.VideoFileClip import VideoFileClip
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

    def load_video(self, source):
        """
        Loads a video source. 
        'source' can be a file path string OR an integer device index (e.g., 0).
        """
        self.stop()
        
        # Save the raw source reference
        self.video_path = source
        
        if self.cap is not None:
            self.cap.release()
            
        # OpenCV naturally takes an integer for live cameras or a string for files
        self.cap = cv2.VideoCapture(source)
        
        # If it's a live camera, we can trigger playback immediately or leave it to the UI
        return self.cap.isOpened()

    def run(self):
        self.running = True
        frame_time = 1000.0 / self.fps

        while self.running and self.cap and self.cap.isOpened():
            start_time = time.perf_counter_ns()
            ret, frame = self.cap.read()
            if not ret:
                # If we are streaming a live webcam (integer device), a bad frame 
                # means a hardware glitch/disconnect, so we break out.
                if isinstance(self.video_path, int):
                    break
                # Otherwise, it's a file! Rewind the video reader back to frame index 0
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                # Attempt to read the very first frame immediately so the pipeline doesn't skip a beat
                ret, frame = self.cap.read()
                if not ret:
                    # Safety fallback: If it still fails, the file might be corrupted or closed
                    break

            # Route the frame through the globally active stage layout
            processed = self.registry.execute_graph(frame)

            # Detect channels dynamically
            if frame.shape[2] == 4:
                # Frame is BGRA (Transparency Active)
                rgb_frame = cv2.cvtColor(processed, cv2.COLOR_BGRA2RGBA)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGBA8888)
            else:
                # Frame is standard BGR (Opaque)
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
        """Reads, processes, and writes every video frame out to disk using the graph manager."""
        import cv2
        
        if not self.cap or not self.cap.isOpened():
            self.export_finished.emit(False)
            return

        # Guard: Cannot export a live webcam input device stream
        if isinstance(self.video_path, int):
            print("Export cancelled: Cannot export a live input device stream.")
            self.export_finished.emit(False)
            return

        # 1. Gather raw video properties for the writer setup
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            total_frames = 1 # Prevent divide-by-zero errors

        # 2. Open a separate video capture reader specifically for the export process
        # (This avoids messing up the frame position of the live playback thread)
        export_cap = cv2.VideoCapture(self.video_path)
        
        # 3. Initialize the OpenCV VideoWriter 
        # (Using MP4V encoding for high compatibility with standard .mp4 extensions)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            export_cap.release()
            self.export_finished.emit(False)
            return

        frame_idx = 0
        success = True

        try:
            while True:
                ret, frame = export_cap.read()
                if not ret:
                    break

                # Run the frame through the entire dependency graph up to your chosen node
                processed = self.registry.execute_graph(frame)
                
                # Drop Alpha transparency channel if it exists before saving 
                # (Standard MP4 video files only support 3-channel BGR layouts)
                if processed.shape[2] == 4:
                    processed = cv2.cvtColor(processed, cv2.COLOR_BGRA2BGR)

                # 4. Write the processed glitch matrix frame to file
                writer.write(processed)
                
                # 5. Broadcast rendering update signals to drive the UI progress bar
                frame_idx += 1
                self.export_progress.emit(frame_idx, total_frames)

        except Exception as e:
            print(f"Export Error encountered: {e}")
            success = False
        finally:
            # 6. Clean up resources and close files cleanly
            export_cap.release()
            writer.release()
            self.export_finished.emit(success)

    def __del__(self):
        if self.cap:
            self.cap.release()