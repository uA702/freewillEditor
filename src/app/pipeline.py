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

def custom_multi_mixer_processor(node: PipelineStage, frame: np.ndarray, deps: Dict[str, np.ndarray]) -> np.ndarray:
    if not node.inputs: 
        return frame.copy()
    
    # 1. Dynamically sync parameters and metadata based on current inputs
    for input_name in node.inputs:
        param_key = f"volume_{input_name}"
        if param_key not in node.parameters:
            node.parameters[param_key] = 50  # Default to 50% mix volume
            node.parameters_metadata[param_key] = {
                "type": "int", "default": 50, "min": 0, "max": 100
            }

    # 2. Start with a solid, clean base canvas (forced 4-channel BGRA to support transparency)
    h, w, _ = frame.shape
    mixed_canvas = np.zeros((h, w, 4), dtype=np.float32)
    
    # Track if *any* transparency or alpha keying is actually happening in this stack
    alpha_composited_any = False

    # 3. Process and composite every incoming track
    for input_name in node.inputs:
        if input_name not in deps:
            continue
            
        volume = node.parameters.get(f"volume_{input_name}", 50) / 100.0
        track_frame = deps[input_name].copy()
        
        # Check if this track is outputting transparency (4 channels)
        if track_frame.shape[2] == 4:
            alpha_composited_any = True
            track_fg = track_frame[:, :, :3].astype(np.float32)
            # Normalize alpha mask to 0.0 - 1.0 range, then apply track volume
            track_alpha = (track_frame[:, :, 3].astype(np.float32) / 255.0) * volume
        else:
            # Standard 3-channel track gets a full-screen solid alpha mask
            track_fg = track_frame.astype(np.float32)
            track_alpha = np.ones((h, w), dtype=np.float32) * volume

        # --- ALPHA COMPOSITING (Photoshop Over-Blending Math) ---
        # Reshape alpha for broadcasting: (H, W, 1)
        track_alpha_3d = np.expand_dims(track_alpha, axis=2)
        
        # Isolate the current canvas background color and alpha
        bg_rgb = mixed_canvas[:, :, :3]
        bg_alpha = np.expand_dims(mixed_canvas[:, :, 3], axis=2)
        
        # Compute new alpha layer depth
        out_alpha = track_alpha_3d + bg_alpha * (1.0 - track_alpha_3d)
        
        # Blend the color channels based on the track's alpha transparency mask
        # Guard against zero-division for completely vacant pixels
        with np.errstate(invalid='ignore', divide='ignore'):
            out_rgb = (track_fg * track_alpha_3d + bg_rgb * bg_alpha * (1.0 - track_alpha_3d)) / np.where(out_alpha == 0, 1.0, out_alpha)
        
        # Save back onto our rolling composite canvas
        mixed_canvas[:, :, :3] = out_rgb
        mixed_canvas[:, :, 3] = np.squeeze(out_alpha)

    # 4. Final Render & Display Engine Compatibility Guard
    if alpha_composited_any:
        # If any transparency was used, the unmasked pixels must be forced to a background color
        # (e.g., Solid Black) so that UI windows display them as "disappeared".
        final_rgb = mixed_canvas[:, :, :3]
        final_alpha = np.expand_dims(mixed_canvas[:, :, 3], axis=2)
        
        # Render over a solid black background
        rendered_canvas = final_rgb * final_alpha
        
        # Optional glitch pass (0xFF wrap compatibility)
        res_int = rendered_canvas.astype(np.int32)
        return (res_int & 0xFF).astype(np.uint8)
    else:
        # Fallback: No tracks have transparency, treat like a standard linear mixer
        # We manually process it as standard BGR so weights don't drop exposure.
        fallback_canvas = np.zeros((h, w, 3), dtype=np.float32)
        total_weight = 0.0
        
        for input_name in node.inputs:
            if input_name in deps:
                volume = node.parameters.get(f"volume_{input_name}", 50) / 100.0
                fallback_canvas += deps[input_name][:, :, :3].astype(np.float32) * volume
                total_weight += volume
                
        if total_weight > 0.0:
            fallback_canvas /= total_weight
            
        res_int = fallback_canvas.astype(np.int32)
        return (res_int & 0xFF).astype(np.uint8)

# =====================================================================
# 1. INITIALIZE MASTER EFFECT LAYERS (USING THE 'fx.' NAMESPACE)
# =====================================================================
# --- Channel 1 Layers ---
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
fx_b2_thresh   = fx.ThresholdingEffect()
fx_b2_bright   = fx.BrightnessEffect()
fx_b2_contrast = fx.ContrastEffect()
fx_b2_sat      = fx.SaturationEffect()
fx_b2_temp     = fx.ColorTempEffect()
fx_b2_add      = fx.AddEffect()
fx_b2_cshift   = fx.ColorShiftEffect()
fx_b2_invert   = fx.InvertEffect()
fx_b2_blur     = fx.LocalBlurEffect()
fx_b2_echo     = fx.LocalEchoEffect()
fx_b2_tblur    = fx.LocalTemporalBlurEffect()
fx_b2_roi      = fx.ROIEffect()

# --- Channel 3 Layers ---
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

# --- Channel 4 Layers ---
fx_m2_thresh   = fx.ThresholdingEffect()
fx_m2_edges    = fx.EdgeDetectionEffect()
fx_m2_mono     = fx.MonochromeEffect()
fx_m2_bright   = fx.BrightnessEffect()
fx_m2_contrast = fx.ContrastEffect()
fx_m2_invert   = fx.InvertEffect()
fx_m2_cmap     = fx.ColorMappingEffect()
fx_m2_blur     = fx.LocalBlurEffect()
fx_m2_echo     = fx.LocalEchoEffect()
fx_m2_tblur    = fx.LocalTemporalBlurEffect()
fx_m2_roi      = fx.ROIEffect()

# --- Post Finishing Layers ---
fx_post_bright   = fx.BrightnessEffect()
fx_post_contrast = fx.ContrastEffect()
fx_post_sat      = fx.SaturationEffect()
fx_post_temp     = fx.ColorTempEffect()
fx_post_add      = fx.AddEffect()
fx_post_blur     = fx.LocalBlurEffect()
fx_post_echo     = fx.LocalEchoEffect()
fx_post_tblur    = fx.LocalTemporalBlurEffect()
fx_post_features = fx.FeatureClusterTrackerEffect()
fx_post_analog   = fx.AnalogSyncGlitchEffect()


# =====================================================================
# 2. ASSEMBLE ALL-IN-ONE MASTER PLUGINS
# =====================================================================

# --- MASTER STAGE 1: BASIC CHANNEL 1 ---
stage_channel_1 = PipelineStage("Channel 1: Basic Alpha", standard_linear_processor)
stage_channel_1.add_layers([
    fx_b1_thresh, 
    fx_b1_bright, fx_b1_contrast, fx_b1_sat, 
    fx_b1_temp, fx_b1_cshift, fx_b1_add, fx_b1_invert,
    fx_b1_blur, fx_b1_echo, fx_b1_tblur, 
    fx_b1_roi
])

# --- MASTER STAGE 2: BASIC CHANNEL 2 ---
stage_channel_2 = PipelineStage("Channel 2: Basic Beta", standard_linear_processor)
stage_channel_2.add_layers([
    fx_b2_thresh, 
    fx_b2_bright, fx_b2_contrast, fx_b2_sat, 
    fx_b2_temp, fx_b2_cshift, fx_b2_add, fx_b2_invert,
    fx_b2_echo, fx_b2_tblur,
    fx_b2_roi
])

# --- MASTER STAGE 3: MONOCHROME CHANNEL 1 ---
stage_channel_3 = PipelineStage("Channel 3: Mono Glitch Alpha", standard_linear_processor)
stage_channel_3.add_layers([
    fx_m1_mono, fx_m1_invert,
    fx_m1_bright, fx_m1_contrast,
    fx_m1_thresh, fx_m1_edges, 
    fx_m1_blur, 
    fx_m1_echo, fx_m1_tblur, 
    fx_m1_cmap, fx_m1_roi
])

# --- MASTER STAGE 4: MONOCHROME CHANNEL 2 ---
stage_channel_4 = PipelineStage("Channel 4: Mono Glitch Beta", standard_linear_processor)
stage_channel_4.add_layers([
    fx_m2_mono, fx_m2_invert,
    fx_m2_bright, fx_m2_contrast,
    fx_m2_thresh, fx_m2_edges, 
    fx_m2_blur, 
    fx_m2_echo, fx_m2_tblur, 
    fx_m2_cmap, fx_m2_roi
])


# =====================================================================
# 3. CENTRAL AUDIO-STYLE MIXER DESK
# =====================================================================
master_mixer = PipelineStage("Master Mixer Desk", custom_multi_mixer_processor)
master_mixer.set_inputs([
    "Channel 1: Basic Alpha",
    "Channel 2: Basic Beta",
    "Channel 3: Mono Glitch Alpha",
    "Channel 4: Mono Glitch Beta"
])

# Define the master volumes (0-100%) directly on the stage parameters
master_mixer.parameters = {
    "volume_Channel 1: Basic Alpha": 100,
    "volume_Channel 2: Basic Beta": 0,
    "volume_Channel 3: Mono Glitch Alpha": 0,
    "volume_Channel 4: Mono Glitch Beta": 0
}

# Provide the UI metadata so the slider generator knows the ranges
master_mixer.parameters_metadata = {
    "volume_Channel 1: Basic Alpha":       {"type": "int", "default": 100, "min": 0, "max": 100},
    "volume_Channel 2: Basic Beta":        {"type": "int", "default": 0,   "min": 0, "max": 100},
    "volume_Channel 3: Mono Glitch Alpha": {"type": "int", "default": 0,   "min": 0, "max": 100},
    "volume_Channel 4: Mono Glitch Beta":  {"type": "int", "default": 0,   "min": 0, "max": 100}
}


# =====================================================================
# 4. MASTER STAGE 5: POST-MIX FINISHING PATH
# =====================================================================
stage_post_finishing = PipelineStage("Channel 5: Post Finishing", standard_linear_processor)
stage_post_finishing.set_inputs(["Master Mixer Desk"])
stage_post_finishing.add_layers([
    fx_post_features, fx_post_analog
])


# =====================================================================
# 5. REGISTER TO GRAPH MANAGEMENT ENGINE
# =====================================================================
graph_manager = PipelineGraphRegistry()

graph_manager.add_stage(stage_channel_1)
graph_manager.add_stage(stage_channel_2)
graph_manager.add_stage(stage_channel_3)
graph_manager.add_stage(stage_channel_4)
graph_manager.add_stage(master_mixer)
graph_manager.add_stage(stage_post_finishing)

graph_manager.set_output_node("Channel 5: Post Finishing")