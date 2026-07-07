import cv2
import numpy as np
import cmapy
import random
from scipy.sparse.csgraph import connected_components

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
        
        has_alpha = (frame.shape[2] == 4)
        color_data = frame[:, :, :3] if has_alpha else frame

        if self.clipping_disabled:
            processed_color = (color_data * factor).astype(np.uint8)
        else:
            processed_color = np.clip(color_data * factor, 0, 255).astype(np.uint8)

        if has_alpha:
            return np.concatenate([processed_color, frame[:, :, 3:]], axis=2)
        return processed_color


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
            
        has_alpha = (frame.shape[2] == 4)
        color_data = frame[:, :, :3] if has_alpha else frame

        mean = np.mean(color_data, axis=(0, 1), keepdims=True)
        raw_res = mean + (color_data - mean) * factor
        
        if self.clipping_disabled:
            processed_color = raw_res.astype(np.uint8)
        else:
            processed_color = np.clip(raw_res, 0, 255).astype(np.uint8)

        if has_alpha:
            return np.concatenate([processed_color, frame[:, :, 3:]], axis=2)
        return processed_color


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
            
        has_alpha = (frame.shape[2] == 4)
        color_data = frame[:, :, :3] if has_alpha else frame

        hsv = cv2.cvtColor(color_data, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= factor
        
        if self.clipping_disabled:
            hsv_out = (hsv.astype(np.int32) & 0xFF).astype(np.uint8)
            processed_color = cv2.cvtColor(hsv_out, cv2.COLOR_HSV2BGR)
        else:
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            processed_color = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if has_alpha:
            return np.concatenate([processed_color, frame[:, :, 3:]], axis=2)
        return processed_color


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
            
        has_alpha = (frame.shape[2] == 4)
        color_data = frame[:, :, :3] if has_alpha else frame

        res = color_data.astype(np.int16)
        if temp > 0:
            res[:, :, 2] += temp  
            res[:, :, 0] -= temp // 2 
        else:
            res[:, :, 0] -= temp  
            res[:, :, 2] += temp // 2 
            
        if self.clipping_disabled:
            processed_color = res.astype(np.uint8)
        else:
            processed_color = np.clip(res, 0, 255).astype(np.uint8)

        if has_alpha:
            return np.concatenate([processed_color, frame[:, :, 3:]], axis=2)
        return processed_color


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
            
        has_alpha = (frame.shape[2] == 4)
        color_data = frame[:, :, :3] if has_alpha else frame

        res = color_data.astype(np.int16) + val
        if self.clipping_disabled:
            processed_color = res.astype(np.uint8)
        else:
            processed_color = np.clip(res, 0, 255).astype(np.uint8)

        if has_alpha:
            return np.concatenate([processed_color, frame[:, :, 3:]], axis=2)
        return processed_color


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
            
        has_alpha = (frame.shape[2] == 4)
        if has_alpha:
            processed_color = (255 - frame[:, :, :3]).astype(np.uint8)
            return np.concatenate([processed_color, frame[:, :, 3:]], axis=2)
        
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
            
        has_alpha = (frame.shape[2] == 4)
        color_data = frame[:, :, :3] if has_alpha else frame
            
        gray = cv2.cvtColor(color_data, cv2.COLOR_BGR2GRAY)
        processed_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if has_alpha:
            return np.concatenate([processed_color, frame[:, :, 3:]], axis=2)
        return processed_color


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
        
        has_alpha = (frame.shape[2] == 4)
        color_data = frame[:, :, :3] if has_alpha else frame

        rows, cols, _ = color_data.shape
        matrix = np.float32([[1, 0, sx], [0, 1, sy]])
        channel = self.parameters["channel"]

        if channel == 3:
            processed_color = cv2.warpAffine(color_data, matrix, (cols, rows))
        elif 0 <= channel <= 2:
            processed_color = color_data.copy()
            processed_color[:, :, channel] = cv2.warpAffine(color_data[:, :, channel], matrix, (cols, rows))
        else:
            processed_color = color_data

        if has_alpha:
            return np.concatenate([processed_color, frame[:, :, 3:]], axis=2)
        return processed_color


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
        
        has_alpha = (frame.shape[2] == 4)
        color_data = frame[:, :, :3] if has_alpha else frame

        gray = cv2.cvtColor(color_data, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.parameters["low_threshold"], self.parameters["high_threshold"])
        processed_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        if has_alpha:
            return np.concatenate([processed_color, frame[:, :, 3:]], axis=2)
        return processed_color


class ThresholdingEffect(BaseEffect):
    def __init__(self):
        super().__init__()
        
        self.modes = ["Luminance (Range)", "Chroma (HSV Range)", "Green Screen Key"]
        self.outputs = ["Binary Split (0/255 Colors)", "Alpha Channel Mask (Transparent ROI)"]

        self.parameters = {
            "enabled": False,
            "mode": "Luminance (Range)",
            "output_type": "Binary Split (0/255 Colors)",
            "luma_min": 100,        
            "luma_max": 200,        
            "hue_target": 60,       
            "hue_tolerance": 15,     
            "sat_min": 50,
            "val_min": 50,
            "invert_mask": False     
        }
        
        self.parameters_metadata = {
            "enabled": {"type": "bool", "default": False},
            "mode": {"type": "str_choice", "default": "Luminance (Range)", "choices": self.modes},
            "output_type": {"type": "str_choice", "default": "Binary Split (0/255 Colors)", "choices": self.outputs},
            "luma_min": {"type": "int", "default": 100, "min": 0, "max": 255},
            "luma_max": {"type": "int", "default": 200, "min": 0, "max": 255},
            "hue_target": {"type": "int", "default": 60, "min": 0, "max": 180},
            "hue_tolerance": {"type": "int", "default": 15, "min": 1, "max": 90},
            "sat_min": {"type": "int", "default": 50, "min": 0, "max": 255},
            "val_min": {"type": "int", "default": 50, "min": 0, "max": 255},
            "invert_mask": {"type": "bool", "default": False}
        }

    def _sanitize_bool(self, value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "on")
        return bool(value)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        enabled = self._sanitize_bool(self.parameters.get("enabled", True))
        if not enabled:
            return frame

        invert = self._sanitize_bool(self.parameters.get("invert_mask", False))
        
        try:
            l_min = int(self.parameters.get("luma_min", 100))
            l_max = int(self.parameters.get("luma_max", 200))
            target_h = int(self.parameters.get("hue_target", 60))
            tol_h = int(self.parameters.get("hue_tolerance", 15))
            s_min = int(self.parameters.get("sat_min", 50))
            v_min = int(self.parameters.get("val_min", 50))
        except (ValueError, TypeError):
            l_min, l_max = 100, 200
            target_h, tol_h = 60, 15
            s_min, v_min = 50, 50

        mode = str(self.parameters.get("mode", "Luminance (Range)")).strip()
        output_type = str(self.parameters.get("output_type", "Binary Split (0/255 Colors)")).strip()

        color_data = frame[:, :, :3] if frame.shape[2] == 4 else frame

        if mode == "Luminance (Range)":
            gray = cv2.cvtColor(color_data, cv2.COLOR_BGR2GRAY)
            if l_min > l_max:
                l_min, l_max = l_max, l_min
            mask = cv2.inRange(gray, l_min, l_max)

        elif mode in ("Chroma (HSV Range)", "Green Screen Key"):
            hsv = cv2.cvtColor(color_data, cv2.COLOR_BGR2HSV)
            if mode == "Green Screen Key":
                lower_bound = np.array([40, 60, 60])
                upper_bound = np.array([80, 255, 255])
            else:
                lower_bound = np.array([max(0, target_h - tol_h), s_min, v_min])
                upper_bound = np.array([min(180, target_h + tol_h), 255, 255])
            
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            if mode == "Green Screen Key":
                mask = cv2.bitwise_not(mask)
        else:
            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        if invert:
            mask = cv2.bitwise_not(mask)

        if output_type == "Binary Split (0/255 Colors)":
            return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        elif output_type == "Alpha Channel Mask (Transparent ROI)":
            bgra = cv2.cvtColor(color_data, cv2.COLOR_BGR2BGRA)
            bgra[:, :, 3] = mask  
            return bgra

        return frame


class ColorMappingEffect(BaseEffect):
    def __init__(self):
        super().__init__()
        
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

        cmap_groups = [
            {"name": "Perceptually Uniform Sequential", "colormaps": ["viridis", "plasma", "inferno", "magma", "cividis"]},
            {"name": "Sequential", "colormaps": ["Greys", "Purples", "Blues", "Greens", "Oranges", "Reds", "YlOrBr", "YlOrRd", "OrRd", "PuRd", "RdPu", "BuPu", "GnBu", "PuBu", "YlGnBu", "PuBuGn", "BuGn", "YlGn"]},
            {"name": "Sequential (2)", "colormaps": ["binary", "gist_yarg", "gist_gray", "gray", "bone", "pink", "spring", "summer", "autumn", "winter", "cool", "Wistia", "hot", "afmhot", "gist_heat", "copper"]},
            {"name": "Diverging", "colormaps": ["PiYG", "PRGn", "BrBG", "PuOr", "RdGy", "RdBu", "RdYlBu", "RdYlGn", "Spectral", "coolwarm", "bwr", "seismic"]},
            {"name": "Cyclic", "colormaps": ["twilight", "twilight_shifted"]},
            {"name": "Qualitative", "colormaps": ["Pastel1", "Pastel2", "Paired", "Accent", "Dark2", "Set1", "Set2", "Set3", "tab10", "tab20", "tab20b", "tab20c"]},
            {"name": "Miscellaneous", "colormaps": ["flag", "prism", "ocean", "gist_earth", "terrain", "gist_stern", "gnuplot", "gnuplot2", "CMRmap", "cubehelix", "brg", "hsv", "gist_rainbow", "rainbow", "jet", "nipy_spectral", "gist_ncar", "turbo"]}
        ]

        dropdown_options = list(self.native_maps.keys())
        for group in cmap_groups:
            for cmap in group["colormaps"]:
                dropdown_options.append(f"{cmap} (Cmapy)")
                
        dropdown_options.append("Custom Terminal (LUT)")

        self.parameters = {"enabled": False, "colormap_name": "Viridis (OpenCV)"}
        self.parameters_metadata = {
            "enabled": {"type": "bool", "default": False},
            "colormap_name": {"type": "str_choice", "default": "Viridis (OpenCV)", "choices": dropdown_options}
        }
        
        # Precompute Terminal LUT to maximize runtime efficiency
        self.cached_terminal_lut = self._generate_terminal_lut()

    def _generate_terminal_lut(self) -> np.ndarray:
        lut = np.zeros((1, 256, 3), dtype=np.uint8)
        indices = np.arange(256)
        
        r = indices / 2.0 + 128.0
        g = 255.0 - indices / 2.0
        b = np.zeros(256)

        factor1 = 1.0 - np.exp(-((indices / 192.0) ** 2))
        factor2 = np.exp(-((indices / 64.0) ** 2))

        lut[0, :, 0] = np.clip(b, 0, 255).astype(np.uint8)
        lut[0, :, 1] = np.clip(g * factor2, 0, 255).astype(np.uint8)
        lut[0, :, 2] = np.clip(r * factor1, 0, 255).astype(np.uint8)
        return lut

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.parameters.get("enabled", True):
            return frame
            
        has_alpha = (frame.shape[2] == 4)
        color_data = frame[:, :, :3] if has_alpha else frame

        gray = cv2.cvtColor(color_data, cv2.COLOR_BGR2GRAY)
        chosen_map = self.parameters.get("colormap_name", "Viridis (OpenCV)")

        if chosen_map in self.native_maps:
            processed_color = cv2.applyColorMap(gray, self.native_maps[chosen_map])
        elif "Cmapy" in chosen_map:
            cmap_id = chosen_map.split(" ")[0]
            processed_color = cv2.applyColorMap(gray, cmapy.cmap(cmap_id))
        elif chosen_map == "Custom Terminal (LUT)":
            processed_color = cv2.LUT(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), self.cached_terminal_lut)
        else:
            processed_color = color_data

        if has_alpha:
            return np.concatenate([processed_color, frame[:, :, 3:]], axis=2)
        return processed_color


class LocalBlurEffect(BaseEffect):
    def __init__(self):
        super().__init__()
        self.blur_types = ["Gaussian (Standard)", "Centric (Zoom Blur)", "Circular (Spin Blur)"]
        self.parameters = {
            "enabled": False, "blur_type": "Gaussian (Standard)", "kernel_x": 15, "kernel_y": 15,
            "blur_alpha": True, "center_x": 50, "center_y": 50, "strength": 10
        }
        self.parameters_metadata = {
            "enabled": {"type": "bool", "default": False},
            "blur_type": {"type": "str_choice", "default": "Gaussian (Standard)", "choices": self.blur_types},
            "kernel_x": {"type": "int", "default": 15, "min": 1, "max": 199},
            "kernel_y": {"type": "int", "default": 15, "min": 1, "max": 199},
            "blur_alpha": {"type": "bool", "default": True},
            "center_x": {"type": "int", "default": 50, "min": 0, "max": 100},
            "center_y": {"type": "int", "default": 50, "min": 0, "max": 100},
            "strength": {"type": "int", "default": 10, "min": 1, "max": 100}
        }

    def _sanitize_bool(self, value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "on")
        return bool(value)

    def _apply_centric_zoom(self, img: np.ndarray, cx: int, cy: int, strength: float) -> np.ndarray:
        h, w = img.shape[:2]
        steps = 10
        accum = img.astype(np.float32)
        for i in range(1, steps):
            scale = 1.0 + (i / steps) * (strength * 0.02)
            M = cv2.getRotationMatrix2D((cx, cy), 0, scale)
            accum += cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101).astype(np.float32)
        return (accum / steps).astype(np.uint8)

    def _apply_circular_spin(self, img: np.ndarray, cx: int, cy: int, strength: float) -> np.ndarray:
        h, w = img.shape[:2]
        steps = 10
        accum = img.astype(np.float32)
        for i in range(1, steps):
            angle = (i / steps) * strength
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            accum += cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101).astype(np.float32)
        return (accum / steps).astype(np.uint8)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self._sanitize_bool(self.parameters.get("enabled", True)):
            return frame

        blur_type = self.parameters.get("blur_type", "Gaussian (Standard)")
        blur_alpha = self._sanitize_bool(self.parameters.get("blur_alpha", True))
        
        kx = int(self.parameters.get("kernel_x", 15))
        ky = int(self.parameters.get("kernel_y", 15))
        kx = kx + 1 if (kx > 1 and kx % 2 == 0) else max(1, kx)
        ky = ky + 1 if (ky > 1 and ky % 2 == 0) else max(1, ky)

        h, w = frame.shape[:2]
        cx = int((self.parameters.get("center_x", 50) / 100.0) * w)
        cy = int((self.parameters.get("center_y", 50) / 100.0) * h)
        strength = float(self.parameters.get("strength", 10))

        has_alpha = (frame.shape[2] == 4)
        color_data = frame[:, :, :3] if has_alpha else frame

        if blur_type == "Gaussian (Standard)":
            processed_color = cv2.GaussianBlur(color_data, (kx, ky), 0)
        elif blur_type == "Centric (Zoom Blur)":
            processed_color = self._apply_centric_zoom(color_data, cx, cy, strength)
        elif blur_type == "Circular (Spin Blur)":
            processed_color = self._apply_circular_spin(color_data, cx, cy, strength)
        else:
            processed_color = color_data

        if has_alpha:
            if blur_alpha:
                alpha_ch = frame[:, :, 3]
                if blur_type == "Gaussian (Standard)":
                    processed_alpha = cv2.GaussianBlur(alpha_ch, (kx, ky), 0)
                elif blur_type == "Centric (Zoom Blur)":
                    processed_alpha = self._apply_centric_zoom(alpha_ch, cx, cy, strength)
                elif blur_type == "Circular (Spin Blur)":
                    processed_alpha = self._apply_circular_spin(alpha_ch, cx, cy, strength)
                else:
                    processed_alpha = alpha_ch
            else:
                processed_alpha = frame[:, :, 3]
            return np.concatenate([processed_color, processed_alpha[:, :, np.newaxis]], axis=2)

        return processed_color


class ROIEffect(BaseEffect):
    def __init__(self):
        super().__init__()
        self.parameters = {"enabled": False, "top": 0, "bottom": 100, "left": 0, "right": 100}
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
        
        y1, y2 = int((t_pct / 100.0) * h), int((b_pct / 100.0) * h)
        x1, x2 = int((l_pct / 100.0) * w), int((r_pct / 100.0) * w)
        
        output_canvas = np.zeros_like(frame)
        output_canvas[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
        return output_canvas
    
class FeatureTracker(BaseEffect):
    def __init__(self):
        super().__init__()
        self.detector_modes = ["Performance (Shi-Tomasi)", "Extreme Accuracy (SIFT)"]
        self.style_modes = ["Colored", "White", "Black", "Disabled"]
        self.geometry_modes = ["Rectangular Box", "Circle", "Elipsoid", "Connecting Polygon"]
        
        self.parameters = {
            "enabled": False, 
            "detector_type": "Extreme Accuracy (SIFT)", 
            "box_style": "Colored",
            "geometry_mode": "Rectangular Box", # NEW DROPDOWN OPTION
            "max_features": 150, 
            "max_cluster_distance": 50, 
            "min_cluster_density": 4,
            "box_thickness": 2, 
            "mask_alpha_roi": True, 
            "frame_throttle": 2
        }
        
        self.parameters_metadata = {
            "enabled": {"type": "bool", "default": False},
            "detector_type": {"type": "str_choice", "default": "Extreme Accuracy (SIFT)", "choices": self.detector_modes},
            "box_style": {"type": "str_choice", "default": "Colored", "choices": self.style_modes},
            "geometry_mode": {"type": "str_choice", "default": "Rectangular Box", "choices": self.geometry_modes},
            "max_features": {"type": "int", "default": 150, "min": 10, "max": 500},
            "max_cluster_distance": {"type": "int", "default": 50, "min": 5, "max": 200},
            "min_cluster_density": {"type": "int", "default": 4, "min": 2, "max": 20},
            "box_thickness": {"type": "int", "default": 2, "min": 1, "max": 5},
            "mask_alpha_roi": {"type": "bool", "default": True},
            "frame_throttle": {"type": "int", "default": 2, "min": 1, "max": 10}
        }
        self.sift_engine = cv2.SIFT_create()
        self.frame_counter = 0
        self.cached_labels = None
        self.cached_coords = None
        self.cached_num_clusters = 0

    def _sanitize_bool(self, value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "on")
        return bool(value)

    def _get_box_color(self, style: str, index: int, palette: list) -> tuple:
        if style == "White": 
            return (255, 255, 255)
        if style == "Black": 
            return (0, 0, 0)
        return palette[index % len(palette)]

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self._sanitize_bool(self.parameters.get("enabled", True)):
            return frame

        box_style = self.parameters.get("box_style", "Disabled")
        if box_style == "Disabled":
            return frame

        output_frame = frame.copy()
        has_alpha = (output_frame.shape[2] == 4)
        color_data = output_frame[:, :, :3] if has_alpha else output_frame
        h, w = color_data.shape[:2]
        
        detector_type = self.parameters.get("detector_type", "Extreme Accuracy (SIFT)")
        geometry_mode = self.parameters.get("geometry_mode", "Rectangular Box")
        max_feats = int(self.parameters.get("max_features", 150))
        eps = float(self.parameters.get("max_cluster_distance", 50))
        min_samples = int(self.parameters.get("min_cluster_density", 4))
        thick = int(self.parameters.get("box_thickness", 2))
        mask_roi = self._sanitize_bool(self.parameters.get("mask_alpha_roi", True))
        throttle = max(1, int(self.parameters.get("frame_throttle", 2)))

        run_heavy_math = (self.frame_counter % throttle == 0) or (self.cached_coords is None)
        self.frame_counter += 1

        if run_heavy_math:
            gray = cv2.cvtColor(color_data, cv2.COLOR_BGR2GRAY)
            feat_coords = None

            # 1. Execute Detector Backend
            if detector_type == "Extreme Accuracy (SIFT)":
                keypoints = self.sift_engine.detect(gray, None)
                if keypoints:
                    keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)[:max_feats]
                    feat_coords = np.array([kp.pt for kp in keypoints], dtype=np.float32)
            else:
                points = cv2.goodFeaturesToTrack(gray, maxCorners=max_feats, qualityLevel=0.05, minDistance=10)
                if points is not None and len(points) > 0:
                    feat_coords = points.reshape(-1, 2).astype(np.float32)

            self.cached_coords = feat_coords

            if feat_coords is None or len(feat_coords) == 0:
                self.cached_labels = None
                self.cached_num_clusters = 0
            else:
                # 2. HARDWARE-ACCELERATED CLUSTERING
                num_points = feat_coords.shape[0]
                
                diff = feat_coords[:, np.newaxis, :] - feat_coords[np.newaxis, :, :]
                dist_matrix = np.linalg.norm(diff, axis=2)
                
                adj_graph = (dist_matrix <= eps).astype(np.uint8)
                
                num_labels, point_labels = connected_components(csgraph=adj_graph, directed=False, return_labels=True)

                final_labels = np.full(num_points, -1, dtype=np.int32)
                actual_cluster_idx = 0
                
                for lbl in range(num_labels):
                    mask = (point_labels == lbl)
                    if np.sum(mask) >= min_samples:
                        final_labels[mask] = actual_cluster_idx
                        actual_cluster_idx += 1
                        
                self.cached_labels = final_labels
                self.cached_num_clusters = actual_cluster_idx
        
        if self.cached_coords is None or len(self.cached_coords) == 0:
            if mask_roi and has_alpha:
                output_frame[:, :, 3] = 0
            return output_frame

        cluster_colors = [(0, 230, 118), (0, 176, 255), (255, 23, 68), (255, 234, 0), (224, 64, 251), (255, 109, 0)]
        roi_mask = np.zeros((h, w), dtype=np.uint8) if (mask_roi and has_alpha) else None

        for c_grp in range(self.cached_num_clusters):
            cluster_points = self.cached_coords[self.cached_labels == c_grp]
            if len(cluster_points) == 0: 
                continue

            color = self._get_box_color(box_style, c_grp, cluster_colors)
            
            # --- DYNAMIC GEOMETRIC RENDERING INTERFACES ---
            if geometry_mode == "Rectangular Box":
                x_min, y_min = max(0, int(np.min(cluster_points[:, 0]))), max(0, int(np.min(cluster_points[:, 1])))
                x_max, y_max = min(w, int(np.max(cluster_points[:, 0]))), min(h, int(np.max(cluster_points[:, 1])))
                
                cv2.rectangle(color_data, (x_min, y_min), (x_max, y_max), color, thick)
                cv2.putText(color_data, f"TRK #{c_grp + 1}", (x_min, max(y_min - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
                
                if roi_mask is not None:
                    cv2.rectangle(roi_mask, (x_min, y_min), (x_max, y_max), 255, -1)

            elif geometry_mode == "Circle":
                # Automatically calculate the exact minimal enclosing circle parameters
                center, radius = cv2.minEnclosingCircle(cluster_points.astype(np.float32))
                cx, cy, r = int(center[0]), int(center[1]), int(radius)

                cv2.circle(color_data, (cx, cy), r, color, thick)
                cv2.putText(color_data, f"TRK #{c_grp + 1}", (cx - 15, max(cy - r - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

                if roi_mask is not None:
                    cv2.circle(roi_mask, (cx, cy), r, 255, -1) 
                    
            elif geometry_mode == "Elipsoid":
                # Guard: cv2.fitEllipse requires a minimum of 5 points to mathematically calculate a curve
                if len(cluster_points) >= 5:
                    # 1. Fit an optimal rotated bounding ellipse to the feature points cloud
                    # Returns: ((center_x, center_y), (width, height), rotation_angle)
                    ellipse_params = cv2.fitEllipse(cluster_points.astype(np.float32))
                    (cx, cy), (axes_w, axes_h), angle = ellipse_params
                    
                    # 2. Extract integer coordinates for drawing and map axis sizes to radii halves
                    center = (int(cx), int(cy))
                    axes = (int(axes_w / 2), int(axes_h / 2))
                    
                    # 3. Draw the ellipse on the screen layout
                    # Arguments: frame, center, half-axes lengths, tilt angle, start angle, end angle, color, thickness
                    cv2.ellipse(color_data, center, axes, angle, 0, 360, color, thick)
                    
                    # Draw a localized text label anchor offset cleanly from the center point
                    cv2.putText(color_data, f"TRK #{c_grp + 1}", (int(cx) - 15, max(int(cy - (axes_h / 2)) - 5, 15)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
                    
                    # 4. Fill transparency mask array if ROI masking engine is turned on
                    if roi_mask is not None:
                        cv2.ellipse(roi_mask, center, axes, angle, 0, 360, 255, -1)
                else:
                    # Fallback Strategy: If point density is low, fall back gracefully to a circle
                    center, radius = cv2.minEnclosingCircle(cluster_points.astype(np.float32))
                    cx, cy, r = int(center[0]), int(center[1]), int(radius)
                    cv2.circle(color_data, (cx, cy), r, color, thick)
                    cv2.putText(color_data, f"TRK #{c_grp + 1}", (cx - 15, max(cy - r - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
                    if roi_mask is not None:
                        cv2.circle(roi_mask, (cx, cy), r, 255, -1)
                    
            elif geometry_mode == "Connecting Polygon":
                # Compute a Convex Hull boundary contour from the tracking point cluster
                hull = cv2.convexHull(cluster_points.astype(np.int32))
                
                # Reshape array for safe polylines injection
                cv2.polylines(color_data, [hull], isClosed=True, color=color, thickness=thick)
                
                # Extract text anchorage layout placement using first vertex coordinate of polygon
                text_x, text_y = hull[0][0][0], hull[0][0][1]
                cv2.putText(color_data, f"TRK #{c_grp + 1}", (text_x, max(text_y - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
                
                if roi_mask is not None:
                    cv2.fillPoly(roi_mask, [hull], 255)

            # Draw underlying core tracked feature points
            for pt in cluster_points:
                cv2.circle(color_data, (int(pt[0]), int(pt[1])), thick, color, -1)

        if has_alpha:
            output_frame[:, :, :3] = color_data
            if mask_roi and roi_mask is not None:
                output_frame[:, :, 3] = roi_mask
            return output_frame
        return color_data
    

class AnalogSyncGlitchEffect(BaseEffect):
    def __init__(self):
        super().__init__()
        self.parameters = {
            "enabled": False, "h_jitter_amount": 15, "h_jitter_chance": 20, "v_sync_roll": 0, "v_sync_chance": 5
        }
        self.parameters_metadata = {
            "enabled": {"type": "bool", "default": False},
            "h_jitter_amount": {"type": "int", "default": 15, "min": 0, "max": 100},
            "h_jitter_chance": {"type": "int", "default": 20, "min": 0, "max": 100},
            "v_sync_roll": {"type": "int", "default": 0, "min": 0, "max": 100},
            "v_sync_chance": {"type": "int", "default": 5, "min": 0, "max": 100}
        }

    def _sanitize_bool(self, value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "on")
        return bool(value)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self._sanitize_bool(self.parameters.get("enabled", True)):
            return frame

        glitched_frame = frame.copy()
        h, w = glitched_frame.shape[:2]

        h_amount = int(self.parameters.get("h_jitter_amount", 15))
        h_chance = int(self.parameters.get("h_jitter_chance", 20))
        v_roll_pct = int(self.parameters.get("v_sync_roll", 0))
        v_chance = int(self.parameters.get("v_sync_chance", 5))

        # 1. VERTICAL SYNC ROLL (Vectorized Shift Pass)
        if v_roll_pct > 0 and random.randint(1, 100) <= v_chance:
            roll_pixels = int((v_roll_pct / 100.0) * h)
            roll_pixels = (roll_pixels + random.randint(-10, 10)) % h
            glitched_frame = np.roll(glitched_frame, shift=roll_pixels, axis=0)

        # 2. HORIZONTAL JITTER SCANLINES (OPTIMIZED: Replaced python for-loop with vectorized masking)
        if h_amount > 0 and h_chance > 0:
            random_chances = np.random.randint(1, 101, size=h)
            jitter_mask = random_chances <= h_chance
            jitter_rows = np.where(jitter_mask)[0]
            
            if len(jitter_rows) > 0:
                shifts = np.random.randint(-h_amount, h_amount + 1, size=len(jitter_rows))
                for row_idx, shift in zip(jitter_rows, shifts):
                    glitched_frame[row_idx] = np.roll(glitched_frame[row_idx], shift=shift, axis=0)

        return glitched_frame


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
            
        if len(self.history_buffer) > 0 and self.history_buffer[0].shape != frame.shape:
            self.history_buffer.clear()

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
        self.parameters = {"enabled": False, "blend_history": 50}
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
    
class NoiseEffect(BaseEffect):
    def __init__(self):
        super().__init__()
        self.parameters = {
            "enabled": False,
            "intensity": 25,
            "noise_type": "Gaussian",
            "monochromatic": True
        }
        self.parameters_metadata = {
            "enabled": {"type": "bool", "default": False},
            "intensity": {"type": "int", "default": 25, "min": 0, "max": 100},
            "noise_type": {"type": "str_choice", "default": "Gaussian", "choices": ["Gaussian", "Salt & Pepper"]},
            "monochromatic": {"type": "bool", "default": True}
        }

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.parameters.get("enabled", False):
            return frame
            
        intensity = self.parameters.get("intensity", 25)
        if intensity == 0:
            return frame

        # Separate alpha channel safely if present
        has_alpha = (frame.shape[2] == 4)
        color_data = frame[:, :, :3] if has_alpha else frame
        
        h, w, c = color_data.shape
        noise_type = self.parameters.get("noise_type", "Gaussian")
        is_mono = self.parameters.get("monochromatic", True)

        if noise_type == "Gaussian":
            # 1. Generate normal/Gaussian distributed math noise
            noise_channels = 1 if is_mono else c
            raw_noise = np.random.normal(0, intensity, (h, w, noise_channels)).astype(np.float32)
            
            # If monochromatic, broadcast the single noise layer across all BGR channels
            if is_mono:
                raw_noise = np.repeat(raw_noise, c, axis=2)

            # 2. Add noise to the image frame and clamp safely between 0-255 bounds
            # Note: If your system global chaos switch `self.clipping_disabled` is True, 
            # we can skip native clamping to let integer rollover artifacts generate!
            if getattr(self, "clipping_disabled", False):
                processed_color = (color_data.astype(np.float32) + raw_noise).astype(np.uint8)
            else:
                processed_color = np.clip(color_data.astype(np.float32) + raw_noise, 0, 255).astype(np.uint8)

        elif noise_type == "Salt & Pepper":
            # Calculate a scaling factor for how many pixels get corrupted
            prob = intensity / 400.0  # Normalized map so 100 doesn't completely wipe out the frame
            processed_color = color_data.copy()
            
            if is_mono:
                # White speckles (Salt) and Black speckles (Pepper) uniformly applied across channels
                random_matrix = np.random.rand(h, w)
                processed_color[random_matrix < prob] = 0       # Pepper
                processed_color[random_matrix > (1 - prob)] = 255 # Salt
            else:
                # Independent colorful structural channel corruption
                random_matrix = np.random.rand(h, w, c)
                processed_color[random_matrix < prob] = 0
                processed_color[random_matrix > (1 - prob)] = 255

        # Recombine original alpha transparency layout map if it was dropped
        if has_alpha:
            return np.concatenate([processed_color, frame[:, :, 3:]], axis=2)
        return processed_color