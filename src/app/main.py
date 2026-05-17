import sys
import threading
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                               QFileDialog, QGroupBox, QSlider, QComboBox, 
                               QCheckBox, QProgressDialog, QMessageBox, QScrollArea)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

from pipeline import graph_manager
from video_worker import VideoWorker

class VideoEditorApp(QMainWindow):
    def __init__(self, registry):
        super().__init__()
        self.setWindowTitle("Freewill Editor Pipeline Framework")
        self.setGeometry(100, 100, 1200, 670) # Widened window frame slightly
        
        self.registry = registry
        self.worker = VideoWorker(self.registry)
        self.worker.frame_processed.connect(self.update_video_canvas)
        self.worker.video_ended.connect(self.on_video_ended)

        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- 1. Top File Routing Bar (RESTORED EXPORT UI) ---
        top_group = QGroupBox("Routing Options")
        top_layout = QHBoxLayout(top_group)
        
        self.txt_input = QLineEdit()
        self.txt_input.setPlaceholderText("Select raw input file...")
        btn_browse_in = QPushButton("Browse Input")
        btn_browse_in.clicked.connect(self.browse_input)

        self.txt_output = QLineEdit()
        self.txt_output.setPlaceholderText("Select export file target destination...")
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

        body_layout = QHBoxLayout()
        
        # --- 2. Left Dynamic Controls Container ---
        self.controls_box = QGroupBox("Pipeline Variables")
        self.controls_layout = QVBoxLayout(self.controls_box)
        self.controls_box.setFixedWidth(360) # Expanded to 360px for slider row layout components
        
        self.controls_layout.addWidget(QLabel("Select Active Execution Stage:"))
        self.combo_stages = QComboBox()
        self.combo_stages.addItems(list(self.registry.stages.keys()))
        self.combo_stages.currentTextChanged.connect(self.on_stage_changed)
        self.controls_layout.addWidget(self.combo_stages)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        self.sliders_container = QWidget()
        self.sliders_layout = QVBoxLayout(self.sliders_container)
        self.sliders_layout.setAlignment(Qt.AlignTop)
        self.sliders_layout.setContentsMargins(5, 5, 5, 5)
        
        scroll_area.setWidget(self.sliders_container)
        self.controls_layout.addWidget(scroll_area)
        
        body_layout.addWidget(self.controls_box)

        # --- 3. Center Screen Viewport ---
        screen_layout = QVBoxLayout()
        self.lbl_video = QLabel("Load a video file to begin processing paths...")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setStyleSheet("background-color: #121212; border-radius: 4px; border: 2px solid #333;")
        screen_layout.addWidget(self.lbl_video, stretch=1)

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

        self.generate_effect_sliders()

    def on_stage_changed(self, target_node_name):
        """Changes the root endpoint node target of our rendering tree."""
        self.registry.set_output_node(target_node_name)
        self.generate_effect_sliders()

    def toggle_clipping_override(self, layer, button):
        """Toggles the mathematical clip safeguard of a core matrix layer layer."""
        layer.clipping_disabled = not layer.clipping_disabled
        if layer.clipping_disabled:
            button.setStyleSheet("font-weight: bold; color: #FFF; background-color: #E65100; border: 1px solid #FF9800;")
            button.setToolTip("OVERFLOW ACTIVE: Safeguards disabled! Expect native array overflows.")
        else:
            button.setStyleSheet("font-weight: bold; color: #777; background-color: #222; border: 1px solid #444;")
            button.setToolTip("Safeguards Active: Hardware matrix securely clamped (0-255).")

    def generate_effect_sliders(self):
        """Cleans and builds parameters with value labels, resets, and chaos overrides."""
        while self.sliders_layout.count():
            item = self.sliders_layout.takeAt(0)
            widget = item.widget()
            if widget: 
                widget.deleteLater()

        current_stage_name = self.combo_stages.currentText()
        if not current_stage_name: 
            return
        
        stage = self.registry.stages.get(current_stage_name)
        if not stage: 
            return

        lbl_stage = QLabel(f"STAGE: {current_stage_name}")
        lbl_stage.setStyleSheet("color: #00C000; font-weight: bold; margin-top: 12px; font-size: 11pt;")
        self.sliders_layout.addWidget(lbl_stage)

        # --- PART A: Selected Node Properties (e.g., Mixer Master Faders) ---
        if stage.parameters_metadata:
            for param, meta in stage.parameters_metadata.items():
                if meta.get("type") == "int":
                    clean_param_name = param.replace("volume_", "")
                    self.sliders_layout.addWidget(QLabel(f"   [Mixer] {clean_param_name}:"))
                    
                    row_widget = QWidget()
                    row_layout = QHBoxLayout(row_widget)
                    row_layout.setContentsMargins(0, 0, 0, 0)
                    
                    slider = QSlider(Qt.Horizontal)
                    slider.setRange(meta["min"], meta["max"])
                    slider.setValue(stage.parameters[param])
                    
                    lbl_val = QLabel(str(stage.parameters[param]))
                    lbl_val.setFixedWidth(35)
                    lbl_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    
                    btn_override = QPushButton("R")
                    btn_override.setFixedSize(22, 22)
                    btn_override.setToolTip("Reset to default layout value")
                    btn_override.setStyleSheet("font-weight: bold; color: #D32F2F; background-color: #222; border: 1px solid #444;")
                    
                    default_val = meta.get("default", meta["min"])
                    
                    slider.valueChanged.connect(lambda val, s=stage, p=param, l=lbl_val: [
                        s.set_parameter(p, val),
                        l.setText(str(val))
                    ])
                    btn_override.clicked.connect(lambda checked=False, s=slider, d=default_val: s.setValue(d))
                    
                    row_layout.addWidget(slider)
                    row_layout.addWidget(lbl_val)
                    row_layout.addWidget(btn_override)
                    self.sliders_layout.addWidget(row_widget)
                    
                elif meta.get("type") == "bool":
                    checkbox = QCheckBox(f"   [Node Property] Enable {param}")
                    checkbox.setChecked(stage.parameters[param])
                    checkbox.stateChanged.connect(lambda state, s=stage, p=param: s.set_parameter(p, state == 2))
                    self.sliders_layout.addWidget(checkbox)

        # --- PART B: Selected Stage Child Filter Layers ---
        for layer in stage.layers:
            lbl_title = QLabel(f"  └─ Layer: {layer.__class__.__name__}")
            lbl_title.setStyleSheet("font-weight: bold; color: #1976D2; margin-top: 6px; margin-left: 5px;")
            self.sliders_layout.addWidget(lbl_title)

            for param, meta in layer.parameters_metadata.items():
                if meta.get("type") == "int":
                    self.sliders_layout.addWidget(QLabel(f"      {param}:"))
                    
                    row_widget = QWidget()
                    row_layout = QHBoxLayout(row_widget)
                    row_layout.setContentsMargins(0, 0, 0, 0)
                    
                    slider = QSlider(Qt.Horizontal)
                    slider.setRange(meta["min"], meta["max"])
                    slider.setValue(layer.parameters[param])
                    
                    lbl_val = QLabel(str(layer.parameters[param]))
                    lbl_val.setFixedWidth(35)
                    lbl_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    
                    # 1. Reset Button [R]
                    btn_override = QPushButton("R")
                    btn_override.setFixedSize(22, 22)
                    btn_override.setStyleSheet("font-weight: bold; color: #D32F2F; background-color: #222; border: 1px solid #444;")
                    layer_default = meta.get("default", meta["min"])
                    btn_override.clicked.connect(lambda checked=False, s=slider, d=layer_default: s.setValue(d))
                    
                    # 2. Chaos/Clipping Overflow Trigger Button [!]
                    btn_chaos = QPushButton("!")
                    btn_chaos.setFixedSize(22, 22)
                    
                    # Initialize stylesheet color rules based on layer property status on render
                    if getattr(layer, "clipping_disabled", False):
                        btn_chaos.setStyleSheet("font-weight: bold; color: #FFF; background-color: #E65100; border: 1px solid #FF9800;")
                        btn_chaos.setToolTip("OVERFLOW ACTIVE: Safeguards disabled! Expect native array overflows.")
                    else:
                        btn_chaos.setStyleSheet("font-weight: bold; color: #777; background-color: #222; border: 1px solid #444;")
                        btn_chaos.setToolTip("Safeguards Active: Hardware matrix securely clamped (0-255).")
                    
                    btn_chaos.clicked.connect(lambda checked=False, l=layer, b=btn_chaos: self.toggle_clipping_override(l, b))
                    
                    slider.valueChanged.connect(lambda val, l=layer, p=param, lv=lbl_val: [
                        l.set_parameter(p, val),
                        lv.setText(str(val))
                    ])
                    
                    row_layout.addWidget(slider)
                    row_layout.addWidget(lbl_val)
                    row_layout.addWidget(btn_override)
                    row_layout.addWidget(btn_chaos)
                    self.sliders_layout.addWidget(row_widget)
                    
                elif meta.get("type") == "bool":
                    checkbox = QCheckBox(f"      Enable {param}")
                    checkbox.setChecked(layer.parameters[param])
                    checkbox.stateChanged.connect(lambda state, l=layer, p=param: l.set_parameter(p, state == 2))
                    self.sliders_layout.addWidget(checkbox)
        
        self.sliders_layout.addStretch()

    def browse_input(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Videos (*.mp4 *.avi *.mkv *.mov)")
        if path:
            self.txt_input.setText(path)
            self.worker.load_video(path)

    def browse_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "Specify Destination Target", "", "MP4 Video (*.mp4)")
        if path:
            self.txt_output.setText(path)

    def update_video_canvas(self, q_img):
        scaled_pixmap = QPixmap.fromImage(q_img).scaled(
            self.lbl_video.width() - 10, self.lbl_video.height() - 10, Qt.KeepAspectRatio, Qt.SmoothTransformation
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
        QMessageBox.information(self, "Playback Complete", "Video playback loop finished.")

    def start_export(self):
        if not self.txt_input.text() or not self.txt_output.text():
            QMessageBox.warning(self, "Paths Missing", "Please select file routes before attempting export workflows.")
            return

        if self.worker.isRunning():
            self.toggle_play()

        self.progress_dialog = QProgressDialog("Rendering pipeline frames to file...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        
        self.worker.export_progress.connect(lambda current, total: self.progress_dialog.setValue(int((current / total) * 100)))
        self.worker.export_finished.connect(self.on_export_finished)

        t = threading.Thread(target=lambda: self.worker.export_video(self.txt_output.text()))
        t.start()

    def on_export_finished(self, success):
        self.progress_dialog.close()
        try:
            self.worker.export_progress.disconnect()
            self.worker.export_finished.disconnect()
        except RuntimeError:
            pass

        if success:
            QMessageBox.information(self, "Complete", "Pipeline video successfully written to destination!")
        else:
            QMessageBox.critical(self, "Error", "An unexpected exception stopped the engine file export sequence.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = VideoEditorApp(graph_manager)
    editor.show()
    sys.exit(app.exec())