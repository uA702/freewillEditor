import numpy as np
import cv2
from typing import List, Dict, Callable, Any
from effects import BaseEffect, ColorShiftEffect, BrightnessEffect, ContrastEffect

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


# --- Custom Behavior 3: Variable Ghosting Temporal Node ---
def custom_temporal_echo_processor(node: PipelineStage, frame: np.ndarray, deps: Dict[str, np.ndarray]) -> np.ndarray:
    out = deps[node.inputs[0]].copy() if node.inputs else frame.copy()
    
    delay = node.parameters["frame_delay"]
    mix = node.parameters["echo_mix"] / 100.0
    
    # Extract historical frame if buffer has grown deep enough
    if len(node.history) > delay:
        past_frame = node.history[-int(delay)]
        out = cv2.addWeighted(out, 1.0 - mix, past_frame, mix, 0)
    return out


# =====================================================================
# INJECT AND INITIALIZE STAGES
# =====================================================================

fx_bright = BrightnessEffect()
fx_color = ColorShiftEffect()
fx_contrast = ContrastEffect()

# 1. Create Linear Processing Chains
stage_a = PipelineStage("Chain A (Brightness)", standard_linear_processor).add_layers([fx_contrast, fx_bright])
stage_b = PipelineStage("Chain B (Color Shift)", standard_linear_processor).add_layers([fx_color])

# 2. Create Custom Blender with Exposed Sliders
stage_merge = PipelineStage("Alpha Cross-Fader", custom_blend_processor).set_inputs(["Chain A (Brightness)", "Chain B (Color Shift)"])
# Define structural parameters unique to this specific blending node behavior
stage_merge.parameters = {"blend_ratio": 50}
stage_merge.parameters_metadata = {
    "blend_ratio": {"type": "int", "default": 50, "min": 0, "max": 100}
}

# 3. Create Custom Temporal Ghosting Node with Exposed Sliders
stage_echo = PipelineStage("Echo Trail Room", custom_temporal_echo_processor).set_inputs(["Alpha Cross-Fader"])
stage_echo.parameters = {"frame_delay": 5, "echo_mix": 40}
stage_echo.parameters_metadata = {
    "frame_delay": {"type": "int", "default": 5, "min": 1, "max": 25},
    "echo_mix": {"type": "int", "default": 40, "min": 0, "max": 100}
}

# Register everything to the engine
graph_manager = PipelineGraphRegistry()
graph_manager.add_stage(stage_a)
graph_manager.add_stage(stage_b)
graph_manager.add_stage(stage_merge)
graph_manager.add_stage(stage_echo)

graph_manager.set_output_node("Echo Trail Room")