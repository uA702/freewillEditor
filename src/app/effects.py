import cv2
import numpy as np
import cmapy

class BaseEffect:
    """Abstract base class for all frame-processing pipeline layers."""
    def __init__(self):
        self.parameters = {}
        self.parameters_metadata = {}
        self.clipping_disabled = False  # Global chaos switch for integer rollover glitches

    def set_parameter(self, name: str, value):
        """Standardized method to update parameter fields from the UI."""
        if name in self.parameters:
            self.parameters[name] = value

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Process an incoming image frame and return the modified matrix."""
        raise NotImplementedError("Subclasses must implement the apply method.")


# =====================================================================
# 1. NATIVELY NEUTRALIZABLE EFFECTS (With clipping override support)
# =====================================================================
    
class BrightnessEffect(BaseEffect):
    def __init__(self):
        super().__init__()
        self.parameters = {"brightness": 100}
        self.parameters_metadata = {
            "brightness": {"type": "int", "default": 100, "min": 0, "max": 500}
        }

    def apply(self, frame: np.ndarray) -> np.ndarray:
        factor = self.parameters["brightness"] / 100.0
        if factor == 1.0 and not self.clipping_disabled:
            return frame
        
        # If clipping is disabled, bypass native clamping loops to force integer wrap art
        if self.clipping_disabled:
            return (frame * factor).astype(np.uint8)
        return np.clip(frame * factor, 0, 255).astype(np.uint8)


class ContrastEffect(BaseEffect):
    def __init__(self):
        super().__init__()
        self.parameters = {"contrast": 100}
        self.parameters_metadata = {
            "contrast": {"type": "int", "default": 100, "min": 0, "max": 500}
        }

    def apply(self, frame: np.ndarray) -> np.ndarray:
        factor = self.parameters["contrast"] / 100.0
        if factor == 1.0 and not self.clipping_disabled:
            return frame
            
        mean = np.mean(frame, axis=(0, 1), keepdims=True)
        raw_res = mean + (frame - mean) * factor
        
        if self.clipping_disabled:
            return raw_res.astype(np.uint8)
        return np.clip(raw_res, 0, 255).astype(np.uint8)


class SaturationEffect(BaseEffect):
    def __init__(self):
        super().__init__()
        self.parameters = {"saturation": 100}
        self.parameters_metadata = {
            "saturation": {"type": "int", "default": 100, "min": 0, "max": 500}
        }

    def apply(self, frame: np.ndarray) -> np.ndarray:
        factor = self.parameters["saturation"] / 100.0
        if factor == 1.0 and not self.clipping_disabled:
            return frame
            
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= factor
        
        if self.clipping_disabled:
            # Mask out bits past 255 to create sudden neon saturation bands
            hsv_out = (hsv.astype(np.int32) & 0xFF).astype(np.uint8)
            return cv2.cvtColor(hsv_out, cv2.COLOR_HSV2BGR)
            
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


class ColorTempEffect(BaseEffect):
    def __init__(self):
        super().__init__()
        self.parameters = {"temperature": 0}
        self.parameters_metadata = {
            "temperature": {"type": "int", "default": 0, "min": -100, "max": 100}
        }

    def apply(self, frame: np.ndarray) -> np.ndarray:
        temp = self.parameters["temperature"]
        if temp == 0 and not self.clipping_disabled:
            return frame
            
        res = frame.astype(np.int16)
        if temp > 0:
            res[:, :, 2] += temp  # Warm up (Add to Red)
            res[:, :, 0] -= temp // 2 # Sub from Blue
        else:
            res[:, :, 0] -= temp  # Cool down (Add to Blue)
            res[:, :, 2] += temp // 2 # Sub from Red
            
        if self.clipping_disabled:
            return res.astype(np.uint8)
        return np.clip(res, 0, 255).astype(np.uint8)


class AddEffect(BaseEffect):
    def __init__(self):
        super().__init__()
        self.parameters = {"value": 0}
        self.parameters_metadata = {
            "value": {"type": "int", "default": 0, "min": -256, "max": 256}
        }

    def apply(self, frame: np.ndarray) -> np.ndarray:
        val = self.parameters["value"]
        if val == 0 and not self.clipping_disabled:
            return frame
            
        res = frame.astype(np.int16) + val
        if self.clipping_disabled:
            return res.astype(np.uint8)
        return np.clip(res, 0, 255).astype(np.uint8)


# =====================================================================
# 2. TOGGLE-DEPENDENT EFFECTS (Require explicit bypass logic)
# =====================================================================
class InvertEffect(BaseEffect):
    def __init__(self):
        super().__init__()
        self.parameters = {"enabled": False}
        self.parameters_metadata = {
            "enabled": {"type": "bool", "default": False}
        }

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.parameters.get("enabled", True):
            return frame
        return (255 - frame).astype(np.uint8)

class MonochromeEffect(BaseEffect):
    def __init__(self):
        super().__init__()
        self.parameters = {"enabled": False}
        self.parameters_metadata = {
            "enabled": {"type": "bool", "default": False}
        }

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.parameters.get("enabled", True):
            return frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class ColorShiftEffect(BaseEffect):
    def __init__(self):
        super().__init__()
        self.parameters = {"enabled": False, 
                           "channel": 0,
                           "shift_x": 0, 
                           "shift_y": 0}
        self.parameters_metadata = {
            "enabled": {"type": "bool", "default": False},
            "channel": {"type": "int", "default": 0, "min": 0, "max": 3},
            "shift_x": {"type": "int", "default": 0, "min": -100, "max": 100},
            "shift_y": {"type": "int", "default": 0, "min": -100, "max": 100}
        }

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.parameters.get("enabled", True):
            return frame
        
        sx, sy = self.parameters["shift_x"], self.parameters["shift_y"]
        if sx == 0 and sy == 0:
            return frame
        
        rows, cols, _ = frame.shape
        matrix = np.float32([[1, 0, sx], [0, 1, sy]])
        b, g, r = cv2.split(frame)

        channel = self.parameters["channel"]
        if channel == 0: # Override RGB
            b_shifted = cv2.warpAffine(b, matrix, (cols, rows))
            return cv2.merge([b_shifted, g, r])
        elif channel == 1: # Override RGB
            g_shifted = cv2.warpAffine(g, matrix, (cols, rows))
            return cv2.merge([b, g_shifted, r])
        elif channel == 2: # Override RGB
            r_shifted = cv2.warpAffine(r, matrix, (cols, rows))
            return cv2.merge([b, g, r_shifted])
        elif channel == 3: # Override RGB
            return cv2.warpAffine(frame, matrix, (cols, rows))
        else:
            return frame

class EdgeDetectionEffect(BaseEffect):
    def __init__(self):
        super().__init__()
        self.parameters = {"enabled": False, "low_threshold": 50, "high_threshold": 150}
        self.parameters_metadata = {
            "enabled": {"type": "bool", "default": False},
            "low_threshold": {"type": "int", "default": 50, "min": 1, "max": 255},
            "high_threshold": {"type": "int", "default": 150, "min": 1, "max": 255}
        }

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.parameters.get("enabled", True):
            return frame
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.parameters["low_threshold"], self.parameters["high_threshold"])
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


class ThresholdingEffect(BaseEffect):
    def __init__(self):
        super().__init__()
        self.parameters = {"enabled": False, "threshold": 127}
        self.parameters_metadata = {
            "enabled": {"type": "bool", "default": False},
            "threshold": {"type": "int", "default": 127, "min": 0, "max": 255}
        }

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.parameters.get("enabled", True):
            return frame
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, self.parameters["threshold"], 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


class ColorMappingEffect(BaseEffect):
    def __init__(self):
        super().__init__()
        
        # 1. Full Core OpenCV Colormap Catalog (Preserving all variants explicitly)
        self.native_maps = {
            "Autumn (OpenCV)": cv2.COLORMAP_AUTUMN,
            "Bone (OpenCV)": cv2.COLORMAP_BONE,
            "Jet (OpenCV)": cv2.COLORMAP_JET,
            "Winter (OpenCV)": cv2.COLORMAP_WINTER,
            "Rainbow (OpenCV)": cv2.COLORMAP_RAINBOW,
            "Ocean (OpenCV)": cv2.COLORMAP_OCEAN,
            "Summer (OpenCV)": cv2.COLORMAP_SUMMER,
            "Spring (OpenCV)": cv2.COLORMAP_SPRING,
            "Cool (OpenCV)": cv2.COLORMAP_COOL,
            "HSV (OpenCV)": cv2.COLORMAP_HSV,
            "Pink (OpenCV)": cv2.COLORMAP_PINK,
            "Hot (OpenCV)": cv2.COLORMAP_HOT,
            "Parula (OpenCV)": cv2.COLORMAP_PARULA,
            "Magma (OpenCV)": cv2.COLORMAP_MAGMA,
            "Inferno (OpenCV)": cv2.COLORMAP_INFERNO,
            "Plasma (OpenCV)": cv2.COLORMAP_PLASMA,
            "Viridis (OpenCV)": cv2.COLORMAP_VIRIDIS,
            "Cividis (OpenCV)": cv2.COLORMAP_CIVIDIS,
            "Twilight (OpenCV)": cv2.COLORMAP_TWILIGHT,
            "Twilight_Shifted (OpenCV)": cv2.COLORMAP_TWILIGHT_SHIFTED,
            "Turbo (OpenCV)": cv2.COLORMAP_TURBO,
            "Deepgreen (OpenCV)": cv2.COLORMAP_DEEPGREEN
        }

        # 2. Categorized Matplotlib groups array map structure
        cmap_groups = [
            {
                "name": "Perceptually Uniform Sequential",
                "colormaps": ["viridis", "plasma", "inferno", "magma", "cividis"],
            },
            {
                "name": "Sequential",
                "colormaps": [
                    "Greys", "Purples", "Blues", "Greens", "Oranges", "Reds", "YlOrBr",
                    "YlOrRd", "OrRd", "PuRd", "RdPu", "BuPu", "GnBu", "PuBu", "YlGnBu",
                    "PuBuGn", "BuGn", "YlGn",
                ],
            },
            {
                "name": "Sequential (2)",
                "colormaps": [
                    "binary", "gist_yarg", "gist_gray", "gray", "bone", "pink", "spring",
                    "summer", "autumn", "winter", "cool", "Wistia", "hot", "afmhot",
                    "gist_heat", "copper",
                ],
            },
            {
                "name": "Diverging",
                "colormaps": [
                    "PiYG", "PRGn", "BrBG", "PuOr", "RdGy", "RdBu", "RdYlBu", "RdYlGn",
                    "Spectral", "coolwarm", "bwr", "seismic",
                ],
            },
            {
                "name": "Cyclic",
                "colormaps": ["twilight", "twilight_shifted"],
            },
            {
                "name": "Qualitative",
                "colormaps": [
                    "Pastel1", "Pastel2", "Paired", "Accent", "Dark2", "Set1", "Set2",
                    "Set3", "tab10", "tab20", "tab20b", "tab20c",
                ],
            },
            {
                "name": "Miscellaneous",
                "colormaps": [
                    "flag", "prism", "ocean", "gist_earth", "terrain", "gist_stern",
                    "gnuplot", "gnuplot2", "CMRmap", "cubehelix", "brg", "hsv",
                    "gist_rainbow", "rainbow", "jet", "nipy_spectral", "gist_ncar", "turbo",
                ],
            },
        ]

        # 3. Assemble complete list with explicit labeling to ensure both versions coexist
        dropdown_options = list(self.native_maps.keys())
        for group in cmap_groups:
            for cmap in group["colormaps"]:
                dropdown_options.append(f"{cmap} (Cmapy)")
                
        dropdown_options.append("Custom Terminal (LUT)")

        self.parameters = {"enabled": False, "colormap_name": "Viridis (OpenCV)"}
        self.parameters_metadata = {
            "enabled": {"type": "bool", "default": False},
            "colormap_name": {
                "type": "str_choice", 
                "default": "Viridis (OpenCV)", 
                "choices": dropdown_options
            }
        }

    def _generate_terminal_lut(self) -> np.ndarray:
        lut = np.zeros((1, 256, 3), dtype=np.uint8)
        
        # Adjust this to change where the "black threshold" cutoff finishes
        # Higher values shift the cutoff further into the brighter values
        threshold_cutoff1 = 256.0-64.0 
        threshold_cutoff2 = 64.0 
        
        for i in range(256):
            # 1. Base Color Generation (BGR Order)
            # To make a bright pink: Red is maxed out, Green is lower, Blue is medium-high
            r = i/2 + 128
            b = 0
            g = 255 - i/2

            # 2. Corrected Exponential Threshold Equation
            # Dividing 'i' by our cutoff spreads the curve across a visible range
            factor1 = 1.0 - np.exp(-((i / threshold_cutoff1)**2))
            factor2 = np.exp(-((i / threshold_cutoff2)**2))

            # 3. Apply the threshold factor and clamp safely between 0-255
            lut[0, i] = [
                np.clip(b, 0, 255).astype(np.uint8),  # Blue
                np.clip(g * factor2, 0, 255).astype(np.uint8),  # Green
                np.clip(r * factor1, 0, 255).astype(np.uint8)   # Red
            ]
            
        return lut

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.parameters.get("enabled", True):
            return frame
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        chosen_map = self.parameters.get("colormap_name", "Viridis (OpenCV)")

        # Route A: Match Core Explicit OpenCV Options
        if chosen_map in self.native_maps:
            return cv2.applyColorMap(gray, self.native_maps[chosen_map])
            
        # Route B: Parse Matplotlib names inside cmapy
        elif "Cmapy" in chosen_map:
            cmap_id = chosen_map.split(" ")[0]
            return cv2.applyColorMap(gray, cmapy.cmap(cmap_id))

        # Route C: Look Up Tables
        elif chosen_map == "Custom Terminal (LUT)":
            custom_lut = self._generate_terminal_lut()
            return cv2.LUT(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), custom_lut)

        return frame

class LocalBlurEffect(BaseEffect):
    def __init__(self):
        super().__init__()
        self.parameters = {"enabled": False, "kernel_x": 5, "kernel_y": 5}
        self.parameters_metadata = {
            "enabled": {"type": "bool", "default": False},
            "kernel_x": {"type": "int", "default": 5, "min": 1, "max": 99},
            "kernel_y": {"type": "int", "default": 5, "min": 1, "max": 99}
        }

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.parameters.get("enabled", True):
            return frame
            
        kx = self.parameters["kernel_x"]
        ky = self.parameters["kernel_y"]
        
        if kx <= 1 and ky <= 1:
            return frame
            
        if kx > 1 and kx % 2 == 0: 
            kx += 1
        if ky > 1 and ky % 2 == 0: 
            ky += 1
        
        kx = max(1, kx)
        ky = max(1, ky)
        
        return cv2.GaussianBlur(frame, (kx, ky), 0)


class ROIEffect(BaseEffect):
    def __init__(self):
        super().__init__()
        self.parameters = {
            "enabled": False,
            "top": 0,
            "bottom": 100,
            "left": 0,
            "right": 100
        }
        self.parameters_metadata = {
            "enabled": {"type": "bool", "default": False},
            "top":     {"type": "int",  "default": 0,   "min": 0, "max": 100},
            "bottom":  {"type": "int",  "default": 100, "min": 0, "max": 100},
            "left":    {"type": "int",  "default": 0,   "min": 0, "max": 100},
            "right":   {"type": "int",  "default": 100, "min": 0, "max": 100}
        }

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.parameters.get("enabled", True):
            return frame
            
        h, w, c = frame.shape
        
        t_pct, b_pct = self.parameters["top"], self.parameters["bottom"]
        l_pct, r_pct = self.parameters["left"], self.parameters["right"]
        
        if t_pct >= b_pct: 
            b_pct = min(100, t_pct + 1)
        if l_pct >= r_pct: 
            r_pct = min(100, l_pct + 1)
        
        y1 = int((t_pct / 100.0) * h)
        y2 = int((b_pct / 100.0) * h)
        x1 = int((l_pct / 100.0) * w)
        x2 = int((r_pct / 100.0) * w)
        
        cropped = frame[y1:y2, x1:x2]
        output_canvas = np.zeros_like(frame)
        output_canvas[y1:y2, x1:x2] = cropped
        
        return output_canvas


# =====================================================================
# 3. ADVANCED BUFFER-BASED TEMPORAL EFFECTS
# =====================================================================

class LocalEchoEffect(BaseEffect):
    def __init__(self):
        super().__init__()
        self.parameters = {"enabled": False, "delay_frames": 3, "feedback": 50}
        self.parameters_metadata = {
            "enabled": {"type": "bool", "default": False},
            "delay_frames": {"type": "int", "default": 3, "min": 1, "max": 60},
            "feedback": {"type": "int", "default": 50, "min": 0, "max": 100}
        }
        self.history_buffer = []

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.parameters.get("enabled", True):
            self.history_buffer.clear()
            return frame
            
        self.history_buffer.append(frame.copy())
        max_delay = self.parameters["delay_frames"]
        
        if len(self.history_buffer) <= max_delay:
            return frame
            
        delayed_frame = self.history_buffer.pop(0)
        alpha = self.parameters["feedback"] / 100.0
        
        return cv2.addWeighted(frame, 1.0 - alpha, delayed_frame, alpha, 0)


class LocalTemporalBlurEffect(BaseEffect):
    def __init__(self):
        super().__init__()
        self.parameters = {"enabled": False, 
                           "blend_history": 50}
        self.parameters_metadata = {
            "enabled": {"type": "bool", "default": False},
            "blend_history": {"type": "int", "default": 50, "min": 0, "max": 100}
        }
        self.accumulator = None

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.parameters.get("enabled", True):
            self.accumulator = None
            return frame
            
        alpha = self.parameters["blend_history"] / 100.0
        if alpha == 0:
            self.accumulator = None
            return frame
            
        if self.accumulator is None or self.accumulator.shape != frame.shape:
            self.accumulator = frame.copy().astype(np.float32)
            return frame
            
        cv2.accumulateWeighted(frame, self.accumulator, alpha)
        return cv2.convertScaleAbs(self.accumulator)