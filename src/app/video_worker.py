import cv2
import time
import numpy as np
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
        
        # --- NEW IMAGE HANDLING STATE ---
        self.is_image_mode = False
        self.static_image_frame = None
        self.IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif'}

    def load_video(self, source):
        """
        Loads a video OR image source. 
        'source' can be a file path string OR an integer device index (e.g., 0).
        """
        self.stop()
        
        # Save the raw source reference
        self.video_path = source
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        # Clear out image memory configurations
        self.is_image_mode = False
        self.static_image_frame = None

        # 1. Check if the source is a file path and has an image extension
        if isinstance(source, str):
            _, ext = os.path.splitext(source.lower())
            if ext in self.IMAGE_EXTENSIONS:
                if not os.path.exists(source):
                    return False
                
                # Use IMREAD_UNCHANGED to cleanly preserve 4-channel transparent PNG/WebP files
                img = cv2.imread(source, cv2.IMREAD_UNCHANGED)
                if img is None:
                    return False
                
                # Grayscale protection fallback
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    
                self.is_image_mode = True
                self.static_image_frame = img
                
                # Trigger a single preview generation immediately for the UI window
                self._process_and_emit_frame(img)
                return True

        # 2. Traditional Video / Webcam initialization path
        self.cap = cv2.VideoCapture(source)
        return self.cap.isOpened()

    def _process_and_emit_frame(self, frame):
        """Helper method to pass frames safely through the graph matrix to the UI."""
        if frame is None:
            return

        # Route the frame through the globally active stage layout
        processed = self.registry.execute_graph(frame)

        if processed is not None:
            # GLOBAL DISPLAY BLENDING GUARD:
            if len(processed.shape) == 3 and processed.shape[2] == 4:
                color_data = processed[:, :, :3].astype(np.float32)
                alpha_channel = processed[:, :, 3].astype(np.float32) / 255.0
                alpha_3d = np.expand_dims(alpha_channel, axis=2)
                
                # Blend over a solid background (e.g., Solid Black)
                bg = np.zeros_like(color_data) 
                display_frame = (color_data * alpha_3d + bg * (1.0 - alpha_3d)).astype(np.uint8)
            else:
                display_frame = processed

            h, w, c = display_frame.shape
            bytes_per_line = c * w
            format_type = QImage.Format_BGR888 if c == 3 else QImage.Format_RGBA8888
            
            q_img = QImage(display_frame.data, w, h, bytes_per_line, format_type)
            self.frame_processed.emit(q_img.copy())

    def run(self):
        self.running = True
        frame_time = 1000.0 / self.fps

        while self.running:
            start_time = time.perf_counter_ns()
            
            # --- IMAGE RE-RENDER ROUTINE ---
            # If in image mode, keep re-processing the same image (allows real-time adjustments via sliders)
            if self.is_image_mode:
                self._process_and_emit_frame(self.static_image_frame)
                
            # --- STANDARD VIDEO ROUTINE ---
            elif self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    if isinstance(self.video_path, int):
                        break
                    # Rewind file loop
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        break

                self._process_and_emit_frame(frame)
            else:
                # No media active, avoid aggressive loop cycling
                break

            elapsed_ms = (time.perf_counter_ns() - start_time) / 1000000
            self.msleep(int(max(1, frame_time - elapsed_ms)))

    def stop(self):
        self.running = False
        self.wait()

    def restart(self):
        self.registry.clear_history()
        if self.is_image_mode:
            # For images, simply push a fresh frame slice to reset temporal steps
            self._process_and_emit_frame(self.static_image_frame)
        elif self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def export_video(self, output_path):
        """Reads, processes, and writes every frame out to disk (Supports Video files AND Image snapshots)."""
        
        # -----------------------------------------------------------------
        # STATIC IMAGE EXPORT BRANCH
        # -----------------------------------------------------------------
        if self.is_image_mode:
            if self.static_image_frame is None:
                self.export_finished.emit(False)
                return
                
            try:
                # Process the matrix through the live parameters graph
                processed = self.registry.execute_graph(self.static_image_frame)
                
                _, ext = os.path.splitext(output_path.lower())
                params = []
                
                if ext in {'.jpg', '.jpeg'}:
                    # Drop alpha layout safely for JPEG compatibility
                    if len(processed.shape) == 3 and processed.shape[2] == 4:
                        processed = cv2.cvtColor(processed, cv2.COLOR_BGRA2BGR)
                    params = [cv2.IMWRITE_JPEG_QUALITY, 98]
                elif ext == '.png':
                    params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
                elif ext == '.webp':
                    params = [cv2.IMWRITE_WEBP_QUALITY, 98]

                success = cv2.imwrite(output_path, processed, params)
                
                # Send out immediate completion status to satisfy progress components
                self.export_progress.emit(1, 1)
                self.export_finished.emit(success)
                return
            except Exception as e:
                print(f"Image Export Error encountered: {e}")
                self.export_finished.emit(False)
                return

        # -----------------------------------------------------------------
        # STANDARD VIDEO EXPORT BRANCH
        # -----------------------------------------------------------------
        if not self.cap or not self.cap.isOpened() or isinstance(self.video_path, int):
            print("Export cancelled: Video source not valid or live input stream targeted.")
            self.export_finished.emit(False)
            return

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            total_frames = 1

        export_cap = cv2.VideoCapture(self.video_path)
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

                processed = self.registry.execute_graph(frame)
                
                if len(processed.shape) == 3 and processed.shape[2] == 4:
                    processed = cv2.cvtColor(processed, cv2.COLOR_BGRA2BGR)

                writer.write(processed)
                
                frame_idx += 1
                self.export_progress.emit(frame_idx, total_frames)

        except Exception as e:
            print(f"Export Error encountered: {e}")
            success = False
        finally:
            export_cap.release()
            writer.release()
            self.export_finished.emit(success)

    def __del__(self):
        if self.cap:
            self.cap.release()