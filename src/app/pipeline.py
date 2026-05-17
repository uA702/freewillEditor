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
        self.layers: List[BaseEffect] = []
        
        # Dedicated Node Parameters (Exposes custom sliders for blends, echoes, weights, etc.)
        self.parameters: Dict[str, Any] = {}
        self.parameters_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Temporal State Cache: Independent historical frame buffer
        self.history: List[np.ndarray] = []
        self.max_history = 100 

    def set_inputs(self, input_stage_names: List[str]):
        self.inputs = input_stage_names
        return self

    def add_layers(self, layers: List[BaseEffect]):
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

# --- Custom Behavior 5: Multi-Channel Audio-Style Video Mixer Console ---
def custom_multi_mixer_processor(node: PipelineStage, frame: np.ndarray, deps: Dict[str, np.ndarray]) -> np.ndarray:
    if not node.inputs: 
        return frame.copy()
    
    # 1. Dynamically sync parameters and metadata based on current inputs
    # If a new input connection is detected, generate a matching slider (0-100%)
    for input_name in node.inputs:
        param_key = f"volume_{input_name}"
        if param_key not in node.parameters:
            node.parameters[param_key] = 50  # Default to 50% mix volume
            node.parameters_metadata[param_key] = {
                "type": "int", "default": 50, "min": 0, "max": 100
            }

    # 2. Initialize an empty floating point canvas for compounding track layers
    h, w, c = frame.shape
    mixed_canvas = np.zeros((h, w, c), dtype=np.float32)
    total_weight = 0.0

    # 3. Accumulate every track based on its active slider "volume"
    for input_name in node.inputs:
        if input_name in deps:
            track_frame = deps[input_name].astype(np.float32)
            volume = node.parameters.get(f"volume_{input_name}", 50) / 100.0
            
            mixed_canvas += track_frame * volume
            total_weight += volume

    # 4. Normalize the mix to avoid unintended blowouts (unless override is on!)
    # We check a global configuration or use the frame's upstream clipping context
    # For a mixer console, normalizing by total weight keeps the image exposure balanced.
    if total_weight > 0.0:
        mixed_canvas /= total_weight

    # 5. Output rendering with your requested overflow glitch compatibility
    # If you want to allow individual channel volumes to glitch past 255, 
    # we cast to int32 and execute the 0xFF mask.
    # (Checking against any parent stage's override, or a local parameter)
    res_int = mixed_canvas.astype(np.int32)
    return (res_int & 0xFF).astype(np.uint8)

# --- Custom Behavior 6: Dynamic ROI Mask with Alpha Transparency ---
def custom_roi_alpha_processor(node: PipelineStage, frame: np.ndarray, deps: Dict[str, np.ndarray]) -> np.ndarray:
    # 1. Get incoming frame (ensure we copy so we don't mutate dependencies)
    input_frame = deps[node.inputs[0]].copy() if node.inputs else frame.copy()
    h, w = input_frame.shape[:2]
    
    # 2. Safely convert incoming BGR frame to BGRA to support alpha transparency
    if input_frame.shape[2] == 3:
        img_bgra = cv2.cvtColor(input_frame, cv2.COLOR_BGR2BGRA)
    else:
        img_bgra = input_frame

    # 3. Read slider parameters
    x_pct = node.parameters.get("center_x", 50) / 100.0
    y_pct = node.parameters.get("center_y", 50) / 100.0
    w_pct = node.parameters.get("width", 30) / 100.0
    h_pct = node.parameters.get("height", 30) / 100.0
    invert = node.parameters.get("invert", False)

    # 4. Convert percentages to absolute pixel coordinates
    center_x, center_y = int(w * x_pct), int(h * y_pct)
    roi_w, roi_h = int(w * w_pct), int(h * h_pct)
    
    x1 = max(0, center_x - (roi_w // 2))
    y1 = max(0, center_y - (roi_h // 2))
    x2 = min(w, center_x + (roi_w // 2))
    y2 = min(h, center_y + (roi_h // 2))

    # 5. Build the alpha mask canvas (0 = transparent, 255 = opaque)
    mask = np.zeros((h, w), dtype=np.uint8)
    if (x2 > x1) and (y2 > y1):
        mask[y1:y2, x1:x2] = 255

    # 6. Apply the inversion toggle
    if invert:
        mask = cv2.bitwise_not(mask)

    # 7. Overwrite the Alpha channel (index 3) with our generated mask
    img_bgra[:, :, 3] = mask

    return img_bgra

# =====================================================================
# 1. INITIALIZE MASTER EFFECT LAYERS (USING THE 'fx.' NAMESPACE)
# =====================================================================
# --- Channel 1 Layers ---
fx_b1_bright   = fx.BrightnessEffect();   fx_b1_contrast = fx.ContrastEffect()
fx_b1_sat      = fx.SaturationEffect();   fx_b1_temp     = fx.ColorTempEffect()
fx_b1_add      = fx.AddEffect();          fx_b1_cshift   = fx.ColorShiftEffect()
fx_b1_blur     = fx.LocalBlurEffect();    fx_b1_echo     = fx.LocalEchoEffect()
fx_b1_tblur    = fx.LocalTemporalBlurEffect()

# --- Channel 2 Layers ---
fx_b2_bright   = fx.BrightnessEffect();   fx_b2_contrast = fx.ContrastEffect()
fx_b2_sat      = fx.SaturationEffect();   fx_b2_temp     = fx.ColorTempEffect()
fx_b2_add      = fx.AddEffect();          fx_b2_cshift   = fx.ColorShiftEffect()
fx_b2_blur     = fx.LocalBlurEffect();    fx_b2_echo     = fx.LocalEchoEffect()
fx_b2_tblur    = fx.LocalTemporalBlurEffect()

# --- Channel 3 Layers ---
fx_m1_mono     = fx.MonochromeEffect();   fx_m1_edges    = fx.EdgeDetectionEffect()
fx_m1_thresh   = fx.ThresholdingEffect(); fx_m1_cmap     = fx.ColorMappingEffect()
fx_m1_blur     = fx.LocalBlurEffect();    fx_m1_echo     = fx.LocalEchoEffect()
fx_m1_tblur    = fx.LocalTemporalBlurEffect()

# --- Channel 4 Layers ---
fx_m2_mono     = fx.MonochromeEffect();   fx_m2_edges    = fx.EdgeDetectionEffect()
fx_m2_thresh   = fx.ThresholdingEffect(); fx_m2_cmap     = fx.ColorMappingEffect()
fx_m2_blur     = fx.LocalBlurEffect();    fx_m2_echo     = fx.LocalEchoEffect()
fx_m2_tblur    = fx.LocalTemporalBlurEffect()

# --- Post Finishing Layers ---
fx_post_bright   = fx.BrightnessEffect();   fx_post_contrast = fx.ContrastEffect()
fx_post_sat      = fx.SaturationEffect();   fx_post_temp     = fx.ColorTempEffect()
fx_post_add      = fx.AddEffect();          fx_post_blur     = fx.LocalBlurEffect()
fx_post_echo     = fx.LocalEchoEffect();    fx_post_tblur    = fx.LocalTemporalBlurEffect()


# =====================================================================
# 2. ASSEMBLE ALL-IN-ONE MASTER PLUGINS
# =====================================================================

# --- MASTER STAGE 1: BASIC CHANNEL 1 ---
stage_channel_1 = PipelineStage("Channel 1: Basic Alpha", standard_linear_processor)
stage_channel_1.add_layers([
    fx_b1_bright, fx_b1_contrast, fx_b1_sat, 
    fx_b1_temp, fx_b1_add, fx_b1_cshift,
    fx_b1_blur, fx_b1_echo, fx_b1_tblur
])

# --- MASTER STAGE 2: BASIC CHANNEL 2 ---
stage_channel_2 = PipelineStage("Channel 2: Basic Beta", standard_linear_processor)
stage_channel_2.add_layers([
    fx_b2_bright, fx_b2_contrast, fx_b2_sat, 
    fx_b2_temp, fx_b2_add, fx_b2_cshift,
    fx_b2_blur, fx_b2_echo, fx_b2_tblur
])

# --- MASTER STAGE 3: MONOCHROME CHANNEL 1 ---
stage_channel_3 = PipelineStage("Channel 3: Mono Glitch Alpha", standard_linear_processor)
stage_channel_3.add_layers([
    fx_m1_mono, fx_m1_edges, fx_m1_thresh, fx_m1_cmap,
    fx_m1_blur, fx_m1_echo, fx_m1_tblur
])

# --- MASTER STAGE 4: MONOCHROME CHANNEL 2 ---
stage_channel_4 = PipelineStage("Channel 4: Mono Glitch Beta", standard_linear_processor)
stage_channel_4.add_layers([
    fx_m2_mono, fx_m2_edges, fx_m2_thresh, fx_m2_cmap,
    fx_m2_blur, fx_m2_echo, fx_m2_tblur
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
    fx_post_bright, fx_post_contrast, fx_post_sat, 
    fx_post_temp, fx_post_add,
    fx_post_blur, fx_post_echo, fx_post_tblur
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