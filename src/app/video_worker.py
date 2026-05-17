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
        """Processes frames via OpenCV, then merges the original video's audio track back in."""
        if not self.cap or not self.cap.isOpened():
            self.export_finished.emit(False)
            return

        # Live webcam check guard
        if isinstance(self.video_path, int):
            print("Export cancelled: Cannot export a live webcam input device.")
            self.export_finished.emit(False)
            return

        # 1. Setup temporary file location for the silent processed video
        base, ext = os.path.splitext(output_path)
        temp_output_path = f"{base}_temp_silent{ext}"

        # 2. Gather raw video properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: 
            total_frames = 1

        # Open dedicated capture context for rendering
        export_cap = cv2.VideoCapture(self.video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Write to our TEMP silent file first
        writer = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            export_cap.release()
            self.export_finished.emit(False)
            return

        frame_idx = 0
        success = True

        # --- STAGE 1: Process and render the visual frames ---
        try:
            while True:
                ret, frame = export_cap.read()
                if not ret:
                    break

                # Run through the entire mixed glitch channel registry graph
                processed = self.registry.execute_graph(frame)
                
                # Convert 4-channel alpha layers to standard 3-channel BGR for container writing
                if processed.shape[2] == 4:
                    processed = cv2.cvtColor(processed, cv2.COLOR_BGRA2BGR)

                writer.write(processed)
                
                frame_idx += 1
                # Allocate 85% of the overall progress track bar to the raw video computation loop
                self.export_progress.emit(int((frame_idx / total_frames) * 85), 100)

        except Exception as e:
            print(f"Visual Rendering Error: {e}")
            success = False
        finally:
            export_cap.release()
            writer.release()

        # --- STAGE 2: Audio Track Extraction and Merging Pipeline ---
        if success:
            try:
                # Signal text or update bar to let users know we are multiplexing audio layers
                self.export_progress.emit(90, 100)
                
                # Load the newly generated silent visual clip and the original audio-carrying file
                processed_visual_clip = VideoFileClip(temp_output_path)
                original_source_clip = VideoFileClip(self.video_path)
                
                if original_source_clip.audio is not None:
                    # Bind the pristine audio track straight to your glitch visual sequence
                    final_merged_output = processed_visual_clip.set_audio(original_source_clip.audio)
                    
                    # Write out the complete final file container
                    final_merged_output.write_videofile(
                        output_path,
                        codec="libx264",
                        audio_codec="aac",
                        logger=None # Suppresses massive text spam in your Python terminal window
                    )
                    
                    # Close handlers to drop file system access locks
                    final_merged_output.close()
                else:
                    # Fallback if original asset has no native soundtrack data
                    processed_visual_clip.close()
                    if os.path.exists(output_path): 
                        os.remove(output_path)
                    os.rename(temp_output_path, output_path)
                    
                original_source_clip.close()
                processed_visual_clip.close()

            except Exception as audio_err:
                print(f"Audio Multiplex Muxing Error: {audio_err}")
                success = False
            finally:
                # Clean up and discard the intermediate temporary file asset safely
                if os.path.exists(temp_output_path):
                    try:
                        os.remove(temp_output_path)
                    except Exception:
                        pass

        # Complete track bar feedback signal and emit final execution completion status back to main window UI
        self.export_progress.emit(100 if success else 0, 100)
        self.export_finished.emit(success)

    def __del__(self):
        if self.cap:
            self.cap.release()