import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                             QFileDialog, QGroupBox, QSlider, QMessageBox, QProgressDialog,
                             QCheckBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

import app.core.effects as fx
from app.core.pipeline import EffectsPipeline
from app.core.video_worker import VideoWorker

class VideoEditorApp(QMainWindow):
    def __init__(self, pipeline):
        super().__init__()
        self.setWindowTitle("Freewill Video Pipeline - PySide6")
        self.setGeometry(100, 100, 1100, 650)
        
        self.pipeline = pipeline
        
        # Initialize background video processing worker thread
        self.worker = VideoWorker(self.pipeline)
        self.worker.frame_processed.connect(self.update_video_canvas)
        self.worker.video_ended.connect(self.on_video_ended)

        self.setup_ui()

    def setup_ui(self):
        # Main central structural frame
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- 1. Top File Routing Bar ---
        top_group = QGroupBox("File Routing Setup")
        top_layout = QHBoxLayout(top_group)

        self.txt_input = QLineEdit()
        self.txt_input.setPlaceholderText("Select raw input file...")
        btn_browse_in = QPushButton("Browse Input")
        btn_browse_in.clicked.connect(self.browse_input)

        self.txt_output = QLineEdit()
        self.txt_output.setPlaceholderText("Select path to export file...")
        btn_browse_out = QPushButton("Browse Output")
        btn_browse_out.clicked.connect(self.browse_output)

        btn_export = QPushButton("Export Video")
        btn_export.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold;")
        btn_export.clicked.connect(self.start_export)

        top_layout.addWidget(QLabel("Input:"))
        top_layout.addWidget(self.txt_input)
        top_layout.addWidget(btn_browse_in)
        top_layout.addWidget(QLabel("Output:"))
        top_layout.addWidget(self.txt_output)
        top_layout.addWidget(btn_browse_out)
        top_layout.addWidget(btn_export)
        main_layout.addWidget(top_group)

        # --- Sub-Layout Body: Controls Left, Screen Right ---
        body_layout = QHBoxLayout()
        
        # --- 2. Left Parameter Controls Panel ---
        self.controls_box = QGroupBox("Dynamic Pipeline Variables")
        self.controls_layout = QVBoxLayout(self.controls_box)
        self.controls_box.setFixedWidth(280)
        self.generate_effect_sliders()
        body_layout.addWidget(self.controls_box)

        # --- 3. Center Screen Viewport ---
        screen_layout = QVBoxLayout()
        self.lbl_video = QLabel("No Video Loaded")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setStyleSheet("background-color: #121212; border: 2px solid #333; border-radius: 4px;")
        screen_layout.addWidget(self.lbl_video, stretch=1)

        # Playback row buttons
        playback_layout = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self.toggle_play)
        btn_restart = QPushButton("Restart")
        btn_restart.clicked.connect(self.restart_video)
        
        playback_layout.addStretch()
        playback_layout.addWidget(self.btn_play)
        playback_layout.addWidget(btn_restart)
        playback_layout.addStretch()
        screen_layout.addLayout(playback_layout)

        body_layout.addLayout(screen_layout, stretch=1)
        main_layout.addLayout(body_layout)

    def browse_input(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Videos (*.mp4 *.avi *.mkv *.mov)")
        if path:
            self.txt_input.setText(path)
            if self.worker.load_video(path):
                self.lbl_video.setText("Video loaded ready for playback thread context.")

    def browse_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "Specify Target Destination", "", "MP4 Video (*.mp4)")
        if path:
            self.txt_output.setText(path)

    def generate_effect_sliders(self):
        """Iterates down operational pipeline modules to map parameter adjustments."""
        for layer in self.pipeline.layers:
            lbl_title = QLabel(layer.__class__.__name__)
            lbl_title.setStyleSheet("font-weight: bold; color: #1976D2; font-size: 13px; margin-top: 10px;")
            self.controls_layout.addWidget(lbl_title)

            for param, meta in layer.parameters_metadata.items():
                if meta.get("type") == "int":
                    self.controls_layout.addWidget(QLabel(f"{param}:"))
                    
                    slider = QSlider(Qt.Horizontal)
                    slider.setRange(meta["min"], meta["max"])
                    slider.setValue(layer.parameters[param])
                    
                    # Lambda assigns target modification pointers efficiently on layout update schedules
                    slider.valueChanged.connect(lambda val, l=layer, p=param: l.set_paarameter(p, val))
                    self.controls_layout.addWidget(slider)

                elif meta.get("type") == "bool":
                    checkbox = QCheckBox(f"Enable {param}")
                    checkbox.setChecked(layer.parameters[param])
                    checkbox.stateChanged.connect(lambda state, l=layer, p=param: l.set_paarameter(p, state == 2))
                    self.controls_layout.addWidget(checkbox)
        
        self.controls_layout.addStretch()

    def update_video_canvas(self, q_img):
        """Safely called via signals sent from our VideoWorker thread execution loops."""
        scaled_pixmap = QPixmap.fromImage(q_img).scaled(
            self.lbl_video.width() - 10, self.lbl_video.height() - 10,
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.lbl_video.setPixmap(scaled_pixmap)

    def toggle_play(self):
        if not self.txt_input.text():
            return
        if self.worker.isRunning():
            self.worker.stop()
            self.btn_play.setText("Play")
        else:
            self.btn_play.setText("Pause")
            self.worker.start()

    def restart_video(self):
        self.worker.restart()

    def on_video_ended(self):
        self.btn_play.setText("Play")
        QMessageBox.information(self, "Playback Finished", "Video reached final frame index loop successfully.")

    def start_export(self):
        if not self.txt_input.text() or not self.txt_output.text():
            QMessageBox.warning(self, "Paths Missing", "Please select valid file routes before attempting export workflows.")
            return

        if self.worker.isRunning():
            self.toggle_play()

        # Set up a native Qt progress overlay
        self.progress_dialog = QProgressDialog("Rendering pipeline frames to file...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        
        # Connect background worker signals directly to updating the ProgressDialog
        self.worker.export_progress.connect(lambda current, total: self.progress_dialog.setValue(int((current / total) * 100)))
        self.worker.export_finished.connect(self.on_export_finished)

        # Fire export processing function inside a safe background environment thread execution block
        # We invoke this using lambda/threading directly to avoid blocking the window engine loop context
        import threading
        t = threading.Thread(target=lambda: self.worker.export_video(self.txt_output.text()))
        t.start()

    def on_export_finished(self, success):
        self.progress_dialog.close()
        if success:
            QMessageBox.information(self, "Complete", "Pipeline video successfully written to specified file target context.")
        else:
            QMessageBox.critical(self, "Error", "An unexpected exception stopped the engine file write sequence.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Setup our basic testing pipeline parameters
    pipeline = EffectsPipeline()
    pipeline.add_layer(fx.BrightnessEffect())
    pipeline.add_layer(fx.DirectionalBlurEffect())

    editor = VideoEditorApp(pipeline)
    editor.show()
    sys.exit(app.exec())