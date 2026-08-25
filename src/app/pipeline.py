import numpy as np
import cv2
from typing import List, Dict, Callable, Any
import effects as fx

class PipelineStage:
    """
    A generic graph node whose specific frame processing behavior is injected 
    at runtime, allowing full decoupling of custom rendering logic.
    """
    def __init__(self, name: str, processing_func: Callable[['PipelineStage', np.ndarray, Dict[str, np.ndarray]], np.ndarray]):
        self.name = name
        self.process_callback = processing_func
        
        # Graph connections
        self.inputs: List[str] = []
        
        # Linear layer children (if this stage hosts standard base effects)
        self.layers: List[fx.BaseEffect] = []
        
        # Dedicated Node Parameters (Exposes custom sliders for blends, echoes, weights, etc.)
        self.parameters: Dict[str, Any] = {}
        self.parameters_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Temporal State Cache: Independent historical frame buffer
        self.history: List[np.ndarray] = []
        self.max_history = 200 

    def set_inputs(self, input_stage_names: List[str]):
        self.inputs = input_stage_names
        return self

    def add_layers(self, layers: List[fx.BaseEffect]):
        self.layers.extend(layers)
        return self

    def set_parameter(self, name: str, value: Any):
        if name in self.parameters:
            meta = self.parameters_metadata.get(name, {})
            if "min" in meta and "max" in meta:
                value = max(min(value, meta["max"]), meta["min"])
            self.parameters[name] = value

    def process(self, raw_frame: np.ndarray, dependency_outputs: Dict[str, np.ndarray]) -> np.ndarray:
        # Manage the temporal buffer automatically
        self.history.append(raw_frame.copy())
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
        # Delegate execution out to the custom injected strategy behavior function
        return self.process_callback(self, raw_frame, dependency_outputs)


class PipelineGraphRegistry:
    def __init__(self):
        self.stages: Dict[str, PipelineStage] = {}
        self.output_node_name: str = ""

    def add_stage(self, stage: PipelineStage):
        self.stages[stage.name] = stage
        if not self.output_node_name:
            self.output_node_name = stage.name

    def set_output_node(self, name: str):
        self.output_node_name = name

    def clear_history(self):
        for stage in self.stages.values():
            stage.history.clear()

    def execute_graph(self, raw_frame: np.ndarray) -> np.ndarray:
        computed_outputs: Dict[str, np.ndarray] = {}
        unprocessed = list(self.stages.keys())
        
        # Simple evaluation loop to execute parent nodes first
        while unprocessed:
            for name in list(unprocessed):
                stage = self.stages[name]
                if all(dep in computed_outputs for dep in stage.inputs):
                    computed_outputs[name] = stage.process(raw_frame, computed_outputs)
                    unprocessed.remove(name)
                    
        return computed_outputs.get(self.output_node_name, raw_frame)


# =====================================================================
# USER ASSEMBLY AREA: DEFINE CUSTOM STAGE BEHAVIORS EXTERNALLY
# =====================================================================

# --- Custom Behavior 1: Classic Linear Sequence ---
def standard_linear_processor(node: PipelineStage, frame: np.ndarray, deps: Dict[str, np.ndarray]) -> np.ndarray:
    out = deps[node.inputs[0]].copy() if node.inputs else frame.copy()
    for layer in node.layers:
        out = layer.apply(out)
    return out

# --- Custom Behavior 2: Variable Weighted Blender Node ---
def custom_blend_processor(node: PipelineStage, frame: np.ndarray, deps: Dict[str, np.ndarray]) -> np.ndarray:
    if len(node.inputs) < 2: 
        return frame
    img_a = deps[node.inputs[0]]
    img_b = deps[node.inputs[1]]
    
    # Read our exposed node parameter safely
    weight_percent = node.parameters["blend_ratio"] / 100.0
    
    return cv2.addWeighted(img_a, weight_percent, img_b, 1.0 - weight_percent, 0)

def custom_multi_mixer_processor(node, frame: np.ndarray, deps: Dict[str, np.ndarray]) -> np.ndarray:
    if not node.inputs: 
        return frame.copy()
    
    # 1. Dynamically sync blend modes & parameters onto metadata maps
    BLEND_MODES = ["Normal", "Screen", "Multiply", "Difference", "Overlay", "Add", "Darken"]
    
    if "blend_mode" not in node.parameters:
        node.parameters["blend_mode"] = "Normal"
        node.parameters_metadata["blend_mode"] = {
            "type": "str_choice", "default": "Normal", "choices": BLEND_MODES
        }
        
    if "clear_alpha_blackout" not in node.parameters:
        node.parameters["clear_alpha_blackout"] = True
        node.parameters_metadata["clear_alpha_blackout"] = {
            "type": "bool", "default": True
        }

    for input_name in node.inputs:
        param_key = f"volume_{input_name}"
        if param_key not in node.parameters:
            node.parameters[param_key] = 50  # Default to 50% volume
            node.parameters_metadata[param_key] = {
                "type": "int", "default": 50, "min": 0, "max": 100
            }

    # Extract dynamic control states
    selected_blend_mode = node.parameters.get("blend_mode", "Normal")
    force_alpha_blackout = node.parameters.get("clear_alpha_blackout", True)

    # 2. Setup processing canvas (forced 4-channel BGRA float32)
    h, w, _ = frame.shape
    mixed_canvas = np.zeros((h, w, 4), dtype=np.float32)
    alpha_composited_any = False

    # Helper function for non-linear blend modes
    def apply_blend_math(fg: np.ndarray, bg: np.ndarray, mode: str) -> np.ndarray:
        if mode == "Screen":
            return 255.0 - ((255.0 - bg) * (255.0 - fg) / 255.0)
        elif mode == "Multiply":
            return (bg * fg) / 255.0
        elif mode == "Difference":
            return np.abs(bg - fg)
        elif mode == "Overlay":
            # Standard Photoshop Overlay formula
            mask = bg < 128.0
            res = np.empty_like(bg)
            res[mask] = (2.0 * bg[mask] * fg[mask]) / 255.0
            res[~mask] = 255.0 - (2.0 * (255.0 - bg[~mask]) * (255.0 - fg[~mask]) / 255.0)
            return res
        elif mode == "Add":
            return bg + fg
        elif mode == "Darken":
            return np.minimum(bg, fg)
        return fg  # Default / Normal mode

    # 3. Process each track
    for input_name in node.inputs:
        if input_name not in deps:
            continue
            
        volume = node.parameters.get(f"volume_{input_name}", 50) / 100.0
        track_frame = deps[input_name].copy()
        
        # Parse frame dimensions & alpha state
        if track_frame.shape[2] == 4:
            alpha_composited_any = True
            track_fg = track_frame[:, :, :3].astype(np.float32)
            track_alpha = (track_frame[:, :, 3].astype(np.float32) / 255.0) * volume
        else:
            track_fg = track_frame.astype(np.float32)
            track_alpha = np.ones((h, w), dtype=np.float32) * volume

        # Alpha thresholding rule: zero out RGB where alpha is fully transparent
        if force_alpha_blackout:
            zero_alpha_mask = (track_alpha == 0.0)
            track_fg[zero_alpha_mask] = 0.0

        track_alpha_3d = np.expand_dims(track_alpha, axis=2)
        
        bg_rgb = mixed_canvas[:, :, :3]
        bg_alpha = np.expand_dims(mixed_canvas[:, :, 3], axis=2)
        
        # Compute dynamic mode target blend before alpha weighting
        blended_fg = apply_blend_math(track_fg, bg_rgb, selected_blend_mode)
        
        # Calculate new output alpha depth map
        out_alpha = track_alpha_3d + bg_alpha * (1.0 - track_alpha_3d)
        
        # Alpha compositing pass
        with np.errstate(invalid='ignore', divide='ignore'):
            out_rgb = (blended_fg * track_alpha_3d + bg_rgb * bg_alpha * (1.0 - track_alpha_3d)) / np.where(out_alpha == 0, 1.0, out_alpha)
        
        # Update running composite canvas
        mixed_canvas[:, :, :3] = out_rgb
        mixed_canvas[:, :, 3] = np.squeeze(out_alpha)

    # 4. Final Output Render Pass
    if alpha_composited_any:
        final_rgb = mixed_canvas[:, :, :3]
        final_alpha = np.expand_dims(mixed_canvas[:, :, 3], axis=2)
        
        # Black out unmasked pixels on canvas
        rendered_canvas = final_rgb * final_alpha
        
        if force_alpha_blackout:
            rendered_canvas[final_alpha[:, :, 0] == 0.0] = 0.0

        res_int = rendered_canvas.astype(np.int32)
        return (res_int & 0xFF).astype(np.uint8)
    else:
        # Fallback for standard 3-channel mixing
        fallback_canvas = np.zeros((h, w, 3), dtype=np.float32)
        total_weight = 0.0
        
        for input_name in node.inputs:
            if input_name in deps:
                volume = node.parameters.get(f"volume_{input_name}", 50) / 100.0
                track_data = deps[input_name][:, :, :3].astype(np.float32)
                fallback_canvas = apply_blend_math(track_data * volume, fallback_canvas, selected_blend_mode) if total_weight > 0 else track_data * volume
                total_weight += volume
                
        if total_weight > 0.0 and selected_blend_mode == "Normal":
            fallback_canvas /= total_weight
            
        res_int = fallback_canvas.astype(np.int32)
        return (res_int & 0xFF).astype(np.uint8)

# =====================================================================
# 1. INITIALIZE MASTER EFFECT LAYERS (USING THE 'fx.' NAMESPACE)
# =====================================================================
# --- Channel 1 Layers ---
fx_b1_noise    = fx.NoiseEffect()
fx_b1_thresh   = fx.ThresholdingEffect()
fx_b1_bright   = fx.BrightnessEffect()
fx_b1_contrast = fx.ContrastEffect()
fx_b1_sat      = fx.SaturationEffect()
fx_b1_temp     = fx.ColorTempEffect()
fx_b1_add      = fx.AddEffect()
fx_b1_cshift   = fx.ColorShiftEffect()
fx_b1_invert   = fx.InvertEffect()
fx_b1_blur     = fx.LocalBlurEffect()
fx_b1_echo     = fx.LocalEchoEffect()
fx_b1_tblur    = fx.LocalTemporalBlurEffect()
fx_b1_roi      = fx.ROIEffect()

# --- Channel 2 Layers ---
fx_m1_noise    = fx.NoiseEffect()
fx_m1_thresh   = fx.ThresholdingEffect()
fx_m1_edges    = fx.EdgeDetectionEffect()
fx_m1_mono     = fx.MonochromeEffect()
fx_m1_bright   = fx.BrightnessEffect()
fx_m1_contrast = fx.ContrastEffect()
fx_m1_invert   = fx.InvertEffect()
fx_m1_cmap     = fx.ColorMappingEffect()
fx_m1_blur     = fx.LocalBlurEffect()
fx_m1_echo     = fx.LocalEchoEffect()
fx_m1_tblur    = fx.LocalTemporalBlurEffect() 
fx_m1_roi      = fx.ROIEffect()

# --- Post finishing Layers ---
fx_post_bright   = fx.BrightnessEffect()
fx_post_contrast = fx.ContrastEffect()
fx_post_sat      = fx.SaturationEffect()
fx_post_temp     = fx.ColorTempEffect()
fx_post_add      = fx.AddEffect()
fx_post_blur     = fx.LocalBlurEffect()
fx_post_echo     = fx.LocalEchoEffect()
fx_post_tblur    = fx.LocalTemporalBlurEffect()
fx_post_features = fx.FeatureTracker()
fx_post_analog   = fx.AnalogSyncGlitchEffect()

# =====================================================================
# 2. ASSEMBLE ALL-IN-ONE MASTER PLUGINS
# =====================================================================

# --- MASTER STAGE 1: BASIC CHANNEL 1 ---
stage_channel_1 = PipelineStage("channel_1:basic", standard_linear_processor)
stage_channel_1.add_layers([
    fx_b1_noise, fx_b1_thresh, 
    fx_b1_bright, fx_b1_contrast, fx_b1_sat, 
    fx_b1_temp, fx_b1_cshift, fx_b1_add, fx_b1_invert,
    fx_b1_blur, fx_b1_echo, fx_b1_tblur, 
    fx_b1_roi
])

# --- MASTER STAGE 2: MONOCHROME CHANNEL 1 ---
stage_channel_2 = PipelineStage("channel_2:mono", standard_linear_processor)
stage_channel_2.add_layers([
    fx_m1_noise, fx_m1_thresh, fx_m1_edges, 
    fx_m1_mono, fx_m1_invert,
    fx_m1_bright, fx_m1_contrast,
    fx_m1_blur, 
    fx_m1_echo, fx_m1_tblur, 
    fx_m1_cmap, fx_m1_roi
])

# =====================================================================
# 3. CENTRAL AUDIO-STYLE MIXER DESK
# =====================================================================
master_mixer = PipelineStage("master_mixer_desk", custom_multi_mixer_processor)
master_mixer.set_inputs([
    "volume_1:basic",
    "volume_2:mono"
])

# # Define the master volumes (0-100%) directly on the stage parameters
# master_mixer.parameters = {
#     "Volume channel_1:basic": 100,
#     "Volume channel_2:mono": 0
# }

# # Provide the UI metadata so the slider generator knows the ranges
# master_mixer.parameters_metadata = {
#     "Volume channel_1:basic": {"type": "int", "default": 100, "min": 0, "max": 100},
#     "Volume channel_2:mono":  {"type": "int", "default": 0,   "min": 0, "max": 100}
# }

# =====================================================================
# 4. MASTER STAGE 5: POST-MIX FINISHING PATH
# =====================================================================
stage_post_finishing = PipelineStage("finishing", standard_linear_processor)
stage_post_finishing.set_inputs(["master_mixer_desk"])
stage_post_finishing.add_layers([
    fx_post_features,
    fx_post_analog
])

# =====================================================================
# 5. REGISTER TO GRAPH MANAGEMENT ENGINE
# =====================================================================
graph_manager = PipelineGraphRegistry()

graph_manager.add_stage(stage_channel_1)
graph_manager.add_stage(stage_channel_2)
graph_manager.add_stage(master_mixer)
graph_manager.add_stage(stage_post_finishing)

graph_manager.set_output_node("finishing")