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
        
        # Channel-safe separation to handle BGRA alpha channels gracefully
        has_alpha = (frame.shape[2] == 4)
        if has_alpha:
            color_data = frame[:, :, :3]
            alpha_channel = frame[:, :, 3:]
        else:
            color_data = frame

        # If clipping is disabled, bypass native clamping loops to force integer wrap art
        if self.clipping_disabled:
            processed_color = (color_data * factor).astype(np.uint8)
        else:
            processed_color = np.clip(color_data * factor, 0, 255).astype(np.uint8)

        if has_alpha:
            return np.concatenate([processed_color, alpha_channel], axis=2)
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
        if has_alpha:
            color_data = frame[:, :, :3]
            alpha_channel = frame[:, :, 3:]
        else:
            color_data = frame

        mean = np.mean(color_data, axis=(0, 1), keepdims=True)
        raw_res = mean + (color_data - mean) * factor
        
        if self.clipping_disabled:
            processed_color = raw_res.astype(np.uint8)
        else:
            processed_color = np.clip(raw_res, 0, 255).astype(np.uint8)

        if has_alpha:
            return np.concatenate([processed_color, alpha_channel], axis=2)
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
        if has_alpha:
            color_data = frame[:, :, :3]
            alpha_channel = frame[:, :, 3:]
        else:
            color_data = frame

        hsv = cv2.cvtColor(color_data, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= factor
        
        if self.clipping_disabled:
            # Mask out bits past 255 to create sudden neon saturation bands
            hsv_out = (hsv.astype(np.int32) & 0xFF).astype(np.uint8)
            processed_color = cv2.cvtColor(hsv_out, cv2.COLOR_HSV2BGR)
        else:
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            processed_color = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if has_alpha:
            return np.concatenate([processed_color, alpha_channel], axis=2)
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
        if has_alpha:
            color_data = frame[:, :, :3]
            alpha_channel = frame[:, :, 3:]
        else:
            color_data = frame

        res = color_data.astype(np.int16)
        if temp > 0:
            res[:, :, 2] += temp  # Warm up (Add to Red)
            res[:, :, 0] -= temp // 2 # Sub from Blue
        else:
            res[:, :, 0] -= temp  # Cool down (Add to Blue)
            res[:, :, 2] += temp // 2 # Sub from Red
            
        if self.clipping_disabled:
            processed_color = res.astype(np.uint8)
        else:
            processed_color = np.clip(res, 0, 255).astype(np.uint8)

        if has_alpha:
            return np.concatenate([processed_color, alpha_channel], axis=2)
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
        if has_alpha:
            color_data = frame[:, :, :3]
            alpha_channel = frame[:, :, 3:]
        else:
            color_data = frame

        res = color_data.astype(np.int16) + val
        if self.clipping_disabled:
            processed_color = res.astype(np.uint8)
        else:
            processed_color = np.clip(res, 0, 255).astype(np.uint8)

        if has_alpha:
            return np.concatenate([processed_color, alpha_channel], axis=2)
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
            color_data = frame[:, :, :3]
            alpha_channel = frame[:, :, 3:]
            processed_color = (255 - color_data).astype(np.uint8)
            return np.concatenate([processed_color, alpha_channel], axis=2)
        
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
        if has_alpha:
            color_data = frame[:, :, :3]
            alpha_channel = frame[:, :, 3:]
            gray = cv2.cvtColor(color_data, cv2.COLOR_BGR2GRAY)
            processed_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            return np.concatenate([processed_color, alpha_channel], axis=2)
            
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
        
        has_alpha = (frame.shape[2] == 4)
        if has_alpha:
            color_data = frame[:, :, :3]
            alpha_channel = frame[:, :, 3:]
        else:
            color_data = frame

        rows, cols, _ = color_data.shape
        matrix = np.float32([[1, 0, sx], [0, 1, sy]])
        b, g, r = cv2.split(color_data)

        channel = self.parameters["channel"]
        if channel == 0: # Override B
            b_shifted = cv2.warpAffine(b, matrix, (cols, rows))
            processed_color = cv2.merge([b_shifted, g, r])
        elif channel == 1: # Override G
            g_shifted = cv2.warpAffine(g, matrix, (cols, rows))
            processed_color = cv2.merge([b, g_shifted, r])
        elif channel == 2: # Override R
            r_shifted = cv2.warpAffine(r, matrix, (cols, rows))
            processed_color = cv2.merge([b, g, r_shifted])
        elif channel == 3: # Override Entire Matrix
            processed_color = cv2.warpAffine(color_data, matrix, (cols, rows))
        else:
            processed_color = color_data

        if has_alpha:
            return np.concatenate([processed_color, alpha_channel], axis=2)
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
        if has_alpha:
            color_data = frame[:, :, :3]
            alpha_channel = frame[:, :, 3:]
            gray = cv2.cvtColor(color_data, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, self.parameters["low_threshold"], self.parameters["high_threshold"])
            processed_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            return np.concatenate([processed_color, alpha_channel], axis=2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.parameters["low_threshold"], self.parameters["high_threshold"])
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


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
        """Force mutated UI inputs (like strings or integers) to actual Python booleans."""
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "on")
        return bool(value)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        # --- TAB SWAP DEFENSE / SANITIZATION LAYER ---
        enabled = self._sanitize_bool(self.parameters.get("enabled", True))
        if not enabled:
            return frame

        # Force UI strings back into clean Python primitives
        invert = self._sanitize_bool(self.parameters.get("invert_mask", False))
        
        try:
            l_min = int(self.parameters.get("luma_min", 100))
            l_max = int(self.parameters.get("luma_max", 200))
            target_h = int(self.parameters.get("hue_target", 60))
            tol_h = int(self.parameters.get("hue_tolerance", 15))
            s_min = int(self.parameters.get("sat_min", 50))
            v_min = int(self.parameters.get("val_min", 50))
        except (ValueError, TypeError):
            # Fallback to defaults if UI values are temporarily mangled during state changes
            l_min, l_max = 100, 200
            target_h, tol_h = 60, 15
            s_min, v_min = 50, 50

        mode = str(self.parameters.get("mode", "Luminance (Range)")).strip()
        output_type = str(self.parameters.get("output_type", "Binary Split (0/255 Colors)")).strip()

        # --- CORE ENGINE LOGIC ---
        # Clear out upstream alpha configurations before calculating matrix limits
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
            # Fallback if text choice string is somehow corrupted by UI selection
            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        if invert:
            mask = cv2.bitwise_not(mask)

        # Output conversion pass
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
        threshold_cutoff1 = 256.0-64.0 
        threshold_cutoff2 = 64.0 
        
        for i in range(256):
            r = i/2 + 128
            b = 0
            g = 255 - i/2

            factor1 = 1.0 - np.exp(-((i / threshold_cutoff1)**2))
            factor2 = np.exp(-((i / threshold_cutoff2)**2))

            lut[0, i] = [
                np.clip(b, 0, 255).astype(np.uint8),  
                np.clip(g * factor2, 0, 255).astype(np.uint8),  
                np.clip(r * factor1, 0, 255).astype(np.uint8)   
            ]
            
        return lut

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.parameters.get("enabled", True):
            return frame
            
        has_alpha = (frame.shape[2] == 4)
        if has_alpha:
            color_data = frame[:, :, :3]
            alpha_channel = frame[:, :, 3:]
        else:
            color_data = frame

        gray = cv2.cvtColor(color_data, cv2.COLOR_BGR2GRAY)
        chosen_map = self.parameters.get("colormap_name", "Viridis (OpenCV)")

        if chosen_map in self.native_maps:
            processed_color = cv2.applyColorMap(gray, self.native_maps[chosen_map])
        elif "Cmapy" in chosen_map:
            cmap_id = chosen_map.split(" ")[0]
            processed_color = cv2.applyColorMap(gray, cmapy.cmap(cmap_id))
        elif chosen_map == "Custom Terminal (LUT)":
            custom_lut = self._generate_terminal_lut()
            processed_color = cv2.LUT(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), custom_lut)
        else:
            processed_color = color_data

        if has_alpha:
            return np.concatenate([processed_color, alpha_channel], axis=2)
        return processed_color


class LocalBlurEffect(BaseEffect):
    def __init__(self):
        super().__init__()
        
        self.blur_types = ["Gaussian (Standard)", "Centric (Zoom Blur)", "Circular (Spin Blur)"]

        self.parameters = {
            "enabled": False,
            "blur_type": "Gaussian (Standard)",
            "kernel_x": 15,
            "kernel_y": 15,
            "blur_alpha": True,       # Toggle to blur transparency mask boundaries
            "center_x": 50,           # Center point % for Zoom/Spin blurs
            "center_y": 50,           # Center point % for Zoom/Spin blurs
            "strength": 10            # Intensity multiplier for Zoom/Spin blurs
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
        """Simulates a camera radial lens zoom explosion effect."""
        h, w = img.shape[:2]
        # Generate multi-scale scaled layers and blend them down linearly
        steps = 10
        accum = img.astype(np.float32)
        
        for i in range(1, steps):
            # Scale factor grows outwards based on strength parameter
            scale = 1.0 + (i / steps) * (strength * 0.02)
            M = cv2.getRotationMatrix2D((cx, cy), 0, scale)
            warped = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            accum += warped.astype(np.float32)
            
        return (accum / steps).astype(np.uint8)

    def _apply_circular_spin(self, img: np.ndarray, cx: int, cy: int, strength: float) -> np.ndarray:
        """Simulates a rapid rotational spinning camera blur."""
        h, w = img.shape[:2]
        steps = 10
        accum = img.astype(np.float32)
        
        for i in range(1, steps):
            # Angular offset grows incrementally per step
            angle = (i / steps) * strength
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            warped = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            accum += warped.astype(np.float32)
            
        return (accum / steps).astype(np.uint8)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self._sanitize_bool(self.parameters.get("enabled", True)):
            return frame

        blur_type = self.parameters.get("blur_type", "Gaussian (Standard)")
        blur_alpha = self._sanitize_bool(self.parameters.get("blur_alpha", True))
        
        # Parse standard kernel sizes
        kx = int(self.parameters.get("kernel_x", 15))
        ky = int(self.parameters.get("kernel_y", 15))
        if kx > 1 and kx % 2 == 0: 
            kx += 1
        if ky > 1 and ky % 2 == 0: 
            ky += 1
        kx, ky = max(1, kx), max(1, ky)

        # Parse relative Center positions
        h, w = frame.shape[:2]
        cx = int((self.parameters.get("center_x", 50) / 100.0) * w)
        cy = int((self.parameters.get("center_y", 50) / 100.0) * h)
        strength = float(self.parameters.get("strength", 10))

        # Channel Split Layer
        has_alpha = (frame.shape[2] == 4)
        if has_alpha:
            color_data = frame[:, :, :3]
            alpha_channel = frame[:, :, 3]
        else:
            color_data = frame
            alpha_channel = None

        # -----------------------------------------------------------------
        # STEP 1: Process Selected Blur Routing on Color Data
        # -----------------------------------------------------------------
        if blur_type == "Gaussian (Standard)":
            processed_color = cv2.GaussianBlur(color_data, (kx, ky), 0)
        elif blur_type == "Centric (Zoom Blur)":
            processed_color = self._apply_centric_zoom(color_data, cx, cy, strength)
        elif blur_type == "Circular (Spin Blur)":
            processed_color = self._apply_circular_spin(color_data, cx, cy, strength)
        else:
            processed_color = color_data

        # -----------------------------------------------------------------
        # STEP 2: Handle Alpha Layer Processing Logic
        # -----------------------------------------------------------------
        if has_alpha:
            if blur_alpha:
                # Run the exact same engine selection on the transparency map
                if blur_type == "Gaussian (Standard)":
                    processed_alpha = cv2.GaussianBlur(alpha_channel, (kx, ky), 0)
                elif blur_type == "Centric (Zoom Blur)":
                    processed_alpha = self._apply_centric_zoom(alpha_channel, cx, cy, strength)
                elif blur_type == "Circular (Spin Blur)":
                    processed_alpha = self._apply_circular_spin(alpha_channel, cx, cy, strength)
                else:
                    processed_alpha = alpha_channel
            else:
                # Maintain original sharp, unblurred alpha mask boundary edges
                processed_alpha = alpha_channel

            return np.concatenate([processed_color, processed_alpha[:, :, np.newaxis]], axis=2)

        return processed_color


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


class FeatureDetectorEffect(BaseEffect):
    def __init__(self):
        super().__init__()
        
        self.algorithms = ["ORB (Fast & Light)", "AKAZE (Precise)"]
        self.outputs = [
            "Overlay Boxes & Points", 
            "Binary Split (0/255 Box Isolation)", 
            "Alpha Channel Mask (Transparent Exterior)"
        ]

        self.parameters = {
            "enabled": False,
            "algorithm": "ORB (Fast & Light)",
            "output_type": "Overlay Boxes & Points",
            "max_features": 500,       
            "padding": 15,             
            "min_cluster_points": 4    
        }
        
        self.parameters_metadata = {
            "enabled": {"type": "bool", "default": False},
            "algorithm": {"type": "str_choice", "default": "ORB (Fast & Light)", "choices": self.algorithms},
            "output_type": {"type": "str_choice", "default": "Overlay Boxes & Points", "choices": self.outputs},
            "max_features": {"type": "int", "default": 500, "min": 50, "max": 2000},
            "padding": {"type": "int", "default": 15, "min": 0, "max": 100},
            "min_cluster_points": {"type": "int", "default": 4, "min": 2, "max": 20}
        }

        self.orb = cv2.ORB_create(nfeatures=self.parameters["max_features"])
        self.akaze = cv2.AKAZE_create()

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.parameters.get("enabled", True):
            return frame

        h, w, c = frame.shape
        algo = self.parameters.get("algorithm", "ORB (Fast & Light)")
        output_type = self.parameters.get("output_type", "Overlay Boxes & Points")
        padding = self.parameters.get("padding", 15)
        min_points = self.parameters.get("min_cluster_points", 4)

        # Slice away preexisting alpha if it exists for tracking detection accuracy
        color_data = frame[:, :, :3] if c == 4 else frame
        gray = cv2.cvtColor(color_data, cv2.COLOR_BGR2GRAY)
        keypoints = []

        if algo == "ORB (Fast & Light)":
            self.orb.setMaxFeatures(self.parameters["max_features"])
            keypoints, _ = self.orb.detectAndCompute(gray, None)
        elif algo == "AKAZE (Precise)":
            keypoints, _ = self.akaze.detectAndCompute(gray, None)

        box_found = False
        x1, y1, x2, y2 = 0, 0, 0, 0

        if len(keypoints) >= min_points:
            pts = np.array([kp.pt for kp in keypoints], dtype=np.int32)
            min_x, min_y = np.min(pts, axis=0)
            max_x, max_y = np.max(pts, axis=0)
            
            x1 = max(0, min_x - padding)
            y1 = max(0, min_y - padding)
            x2 = min(w, max_x + padding)
            y2 = min(h, max_y + padding)
            
            if (x2 - x1) > 0 and (y2 - y1) > 0:
                box_found = True

        # Mode A: Standard UI analysis overlay
        if output_type == "Overlay Boxes & Points":
            out_frame = frame.copy()
            for kp in keypoints:
                pt = (int(kp.pt[0]), int(kp.pt[1]))
                cv2.circle(out_frame, pt, 3, (0, 255, 0), -1)
            if box_found:
                cv2.rectangle(out_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            return out_frame

        # Core Mask Generation for Modes B and C
        mask = np.zeros((h, w), dtype=np.uint8)
        if box_found:
            mask[y1:y2, x1:x2] = 255

        # Mode B: High contrast Binary split 
        if output_type == "Binary Split (0/255 Box Isolation)":
            if c == 4:
                # Isolate 3 channels + preserve original alpha values inside the mask bounding box
                color_masked = cv2.bitwise_and(frame[:, :, :3], frame[:, :, :3], mask=mask)
                alpha_masked = cv2.bitwise_and(frame[:, :, 3], frame[:, :, 3], mask=mask)
                return np.concatenate([color_masked, alpha_masked[:, :, np.newaxis]], axis=2)
            return cv2.bitwise_and(frame, frame, mask=mask)

        # Mode C: Transparence Alpha Keyer
        elif output_type == "Alpha Channel Mask (Transparent Exterior)":
            bgra = cv2.cvtColor(color_data, cv2.COLOR_BGR2BGRA)
            bgra[:, :, 3] = mask  
            return bgra

        return frame


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
            
        # Prevent temporal canvas crashes if upstream parameters instantly add/remove alpha channel
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