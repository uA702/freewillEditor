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
        
        self.is_image_mode = False
        self.static_image_frame = None
        self.IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif'}
        
        # --- PERFORMANCE TUNING ---
        # 1.0 = Original Quality
        # 0.5 = Half Resolution (Dramatically Faster)
        # 0.25 = Quarter Resolution (Maximum Speed / Rough Pixelation)
        self.downscale_factor = 0.5

    def load_video(self, source):
        """Loads a video OR image source with optional hardware-conscious downscaling."""
        self.stop()
        self.video_path = source
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        self.is_image_mode = False
        self.static_image_frame = None

        # --- IMAGE FILE HANDLING ---
        if isinstance(source, str):
            _, ext = os.path.splitext(source.lower())
            if ext in self.IMAGE_EXTENSIONS:
                if not os.path.exists(source):
                    return False
                
                img = cv2.imread(source, cv2.IMREAD_UNCHANGED)
                if img is None:
                    return False
                
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    
                self.is_image_mode = True
                
                # OPTIMIZATION: Apply fast import compression to static image matrix
                if self.downscale_factor != 1.0:
                    h, w = img.shape[:2]
                    new_w = int(w * self.downscale_factor)
                    new_h = int(h * self.downscale_factor)
                    # cv2.INTER_NEAREST is the absolute fastest resizing algorithm available
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

                self.static_image_frame = img
                self._process_and_emit_frame(img)
                return True

        # --- VIDEO / WEBCAM HANDLING ---
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
            
            if self.is_image_mode:
                self._process_and_emit_frame(self.static_image_frame)
                
            elif self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    if isinstance(self.video_path, int):
                        break
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        break

                # OPTIMIZATION: Compress video frame dimensions prior to pipeline processing
                if self.downscale_factor != 1.0:
                    h, w = frame.shape[:2]
                    new_w = int(w * self.downscale_factor)
                    new_h = int(h * self.downscale_factor)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

                self._process_and_emit_frame(frame)
            else:
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

    def export_video(self, output_path, custom_fps=None):
        """Reads, processes, and writes every frame out to disk (Supports downscaled Video/Images and Custom FPS)."""

        # -----------------------------------------------------------------
        # STATIC IMAGE EXPORT BRANCH
        # -----------------------------------------------------------------
        if self.is_image_mode:
            if self.static_image_frame is None:
                self.export_finished.emit(False)
                return
                
            try:
                # The image in memory (self.static_image_frame) is already downscaled 
                # from load_video(), so we just run it straight through the graph!
                processed = self.registry.execute_graph(self.static_image_frame)
                
                _, ext = os.path.splitext(output_path.lower())
                params = []
                
                if ext in {'.jpg', '.jpeg'}:
                    if len(processed.shape) == 3 and processed.shape[2] == 4:
                        processed = cv2.cvtColor(processed, cv2.COLOR_BGRA2BGR)
                    params = [cv2.IMWRITE_JPEG_QUALITY, 85] # Dropping quality slightly to 85 speeds up file compression saving
                elif ext == '.png':
                    params = [cv2.IMWRITE_PNG_COMPRESSION, 1] # 1 is fastest compression, 9 is slowest. Massive PNG save speedup!
                elif ext == '.webp':
                    params = [cv2.IMWRITE_WEBP_QUALITY, 85]

                success = cv2.imwrite(output_path, processed, params)
                
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

        orig_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        export_width = int(orig_width * self.downscale_factor)
        export_height = int(orig_height * self.downscale_factor)
        
        # MODIFICATION: Determine the output frame rate
        if custom_fps and custom_fps > 0:
            fps = float(custom_fps)
        else:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0 # Robust fallback safety net
        
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 1

        export_cap = cv2.VideoCapture(self.video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # The writer will now compile frames using your custom frame timing!
        writer = cv2.VideoWriter(output_path, fourcc, fps, (export_width, export_height))

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

                if self.downscale_factor != 1.0:
                    frame = cv2.resize(frame, (export_width, export_height), interpolation=cv2.INTER_NEAREST)

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