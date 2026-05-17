import numpy as np
import cv2

class BaseEffect:
    def __init__(self):
        self.parameters = {"override_clipping": False}
        self.parameters_metadata = {
            "override_clipping": {
                "type": "bool",
                "default": False
            }
        }

    def set_paarameter(self, name: str, value):
        if name in self.parameters:
            meta = self.parameters_metadata.get(name, {})
            # Fix nested dictionary structure evaluation
            if "min" in meta and "max" in meta:
                value = max(min(value, meta["max"]), meta["min"])
            self.parameters[name] = value
    
    def apply(self, frame: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# --- 1 & 2: BRIGHTNESS & BRIGHTNESS WITH OVERFLOW ---
# Combined into a single class leveraging the override parameter cleanly.
class BrightnessEffect(BaseEffect):
    def __init__(self):
        super().__init__()
        self.parameters.update({"factor": 100})
        self.parameters_metadata.update({
            "factor": {"type": "int", "default": 100, "min": 0, "max": 500}
        })

    def apply(self, frame: np.ndarray) -> np.ndarray:
        factor = self.parameters["factor"] / 100.0
        override = self.parameters["override_clipping"]

        if override:
            return (frame * factor).astype(np.uint8)
        else:
            return np.clip(frame * factor, 0, 255).astype(np.uint8)


# --- 3: CONTRAST ---
class ContrastEffect(BaseEffect):
    def __init__(self):
        super().__init__()
        self.parameters.update({"contrast": 100})  # 100 = Neutral base
        self.parameters_metadata.update({
            "contrast": {"type": "int", "default": 100, "min": 0, "max": 300}
        })

    def apply(self, frame: np.ndarray) -> np.ndarray:
        factor = self.parameters["contrast"] / 100.0
        override = self.parameters["override_clipping"]
        
        # Adjust values around the mid-tone grayscale boundary center (128)
        result = 128.0 + factor * (frame.astype(np.float32) - 128.0)
        
        if override:
            return result.astype(np.uint8)
        return np.clip(result, 0, 255).astype(np.uint8)


# --- 4: SATURATION ---
class SaturationEffect(BaseEffect):
    def __init__(self):
        super().__init__()
        self.parameters.update({"saturation": 100})
        self.parameters_metadata.update({
            "saturation": {"type": "int", "default": 100, "min": 0, "max": 400}
        })

    def apply(self, frame: np.ndarray) -> np.ndarray:
        factor = self.parameters["saturation"] / 100.0
        override = self.parameters["override_clipping"]

        # Shift to HSV space to scale saturation without breaking luminance channels
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= factor

        if override:
            # Force conversion back allowing values to wrap around natively
            hsv_out = hsv.astype(np.uint8)
            return cv2.cvtColor(hsv_out, cv2.COLOR_HSV2BGR)
        
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# --- 5: COLOR TEMPERATURE ---
class ColorTempEffect(BaseEffect):
    def __init__(self):
        super().__init__()
        self.parameters.update({"temp": 0})  # Negative = Cool (Blue), Positive = Warm (Red)
        self.parameters_metadata.update({
            "temp": {"type": "int", "default": 0, "min": -100, "max": 100}
        })

    def apply(self, frame: np.ndarray) -> np.ndarray:
        shift = self.parameters["temp"]
        override = self.parameters["override_clipping"]
        
        result = frame.astype(np.float32)
        if shift > 0:
            result[:, :, 2] += shift  # Warm up: Add to Red (OpenCV BGR index 2)
            result[:, :, 0] -= shift * 0.5  # Sub Blue
        else:
            result[:, :, 0] -= shift  # Cool down: Add to Blue (subtracting negative)
            result[:, :, 2] += shift * 0.5  # Sub Red

        if override:
            return result.astype(np.uint8)
        return np.clip(result, 0, 255).astype(np.uint8)


# --- 6: COLOR SHIFT (MULTIPLY CHANNELS) ---
class ColorShiftEffect(BaseEffect):
    def __init__(self):
        super().__init__()
        self.parameters.update({"factor": 100, "channel": 0})
        self.parameters_metadata.update({
            "factor": {"type": "int", "default": 100, "min": 0, "max": 500},
            "channel": {"type": "int", "default": 0, "min": 0, "max": 2}
        })

    def apply(self, frame: np.ndarray) -> np.ndarray:
        factor = self.parameters["factor"] / 100.0
        channels = self.parameters["channel"]
        override = self.parameters["override_clipping"]
        
        result = frame.copy()
        if isinstance(channels, (int, np.integer)):
            channels = [channels]
            
        for ch in channels:
            if 0 <= ch < frame.shape[2]:
                if override:
                    result[:, :, ch] = (result[:, :, ch] * factor).astype(np.uint8)
                else:
                    result[:, :, ch] = np.clip(result[:, :, ch] * factor, 0, 255).astype(np.uint8)
        return result


# --- 7: ADD EFFECT (ADD TO CHANNELS) ---
class AddEffect(BaseEffect):
    def __init__(self):
        super().__init__()
        self.parameters.update({"addend": 0, "channel": 0})
        self.parameters_metadata.update({
            "addend": {"type": "int", "default": 0, "min": -255, "max": 255},
            "channel": {"type": "int", "default": 0, "min": 0, "max": 2}
        })

    def apply(self, frame: np.ndarray) -> np.ndarray:
        addend = self.parameters["addend"]
        channels = self.parameters["channel"]
        override = self.parameters["override_clipping"]
        
        result = frame.copy()
        if isinstance(channels, (int, np.integer)):
            channels = [channels]
            
        for ch in channels:
            if 0 <= ch < frame.shape[2]:
                if override:
                    result[:, :, ch] += addend  # Force native integer matrix wrapping
                else:
                    res_ch = result[:, :, ch].astype(np.int32) + addend
                    result[:, :, ch] = np.clip(res_ch, 0, 255).astype(np.uint8)
        return result


# --- 8: MONOCHROME ---
class MonochromeEffect(BaseEffect):
    def __init__(self):
        super().__init__()
        # No extra params needed beyond basic base layer configurations
        pass

    def apply(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Re-broadcast back to 3 standard BGR channels to maintain pipeline conformity
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# --- 9: GRADIENT COLOR MAPPING ---
class ColorMappingEffect(BaseEffect):
    def __init__(self):
        super().__init__()
        self.parameters.update({"colormap": 0})
        self.parameters_metadata.update({
            "colormap": {"type": "int", "default": 0, "min": 0, "max": 11} # Map profiles 0 to 11
        })
        # Internal reference mapping array lookup parameters
        self.maps = [
            cv2.COLORMAP_AUTUMN, cv2.COLORMAP_BONE, cv2.COLORMAP_JET, 
            cv2.COLORMAP_WINTER, cv2.COLORMAP_RAINBOW, cv2.COLORMAP_OCEAN,
            cv2.COLORMAP_SUMMER, cv2.COLORMAP_SPRING, cv2.COLORMAP_COOL,
            cv2.COLORMAP_HSV, cv2.COLORMAP_PINK, cv2.COLORMAP_HOT
        ]

    def apply(self, frame: np.ndarray) -> np.ndarray:
        map_idx = self.parameters["colormap"]
        override = self.parameters["override_clipping"]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if override:
            # Shift the base grayscale spectrum values using addition to scramble color maps
            gray = (gray + 100).astype(np.uint8)

        return cv2.applyColorMap(gray, self.maps[map_idx])


# --- 10: EDGE DETECTION ---
class EdgeDetectionEffect(BaseEffect):
    def __init__(self):
        super().__init__()
        self.parameters.update({"threshold1": 100, "threshold2": 200})
        self.parameters_metadata.update({
            "threshold1": {"type": "int", "default": 100, "min": 0, "max": 255},
            "threshold2": {"type": "int", "default": 200, "min": 0, "max": 255}
        })

    def apply(self, frame: np.ndarray) -> np.ndarray:
        t1 = self.parameters["threshold1"]
        t2 = self.parameters["threshold2"]
        override = self.parameters["override_clipping"]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, t1, t2)
        
        if override:
            # Blend raw frames with inverted edges for a distorted glitch effect
            return (frame + cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)).astype(np.uint8)
            
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


# --- 11: THRESHOLDING (BINARY BINARIZATION) ---
class ThresholdingEffect(BaseEffect):
    def __init__(self):
        super().__init__()
        self.parameters.update({"thresh_value": 127})
        self.parameters_metadata.update({
            "thresh_value": {"type": "int", "default": 127, "min": 0, "max": 255}
        })

    def apply(self, frame: np.ndarray) -> np.ndarray:
        val = self.parameters["thresh_value"]
        override = self.parameters["override_clipping"]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if override:
            # Custom modular math mapping instead of straight threshold cuts
            res_gray = (gray // (val + 1)) * 50
            return cv2.cvtColor(res_gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        _, binary = cv2.threshold(gray, val, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


# --- 12: DIRECTIONAL U/V BLURRING ---
class DirectionalBlurEffect(BaseEffect):
    def __init__(self):
        super().__init__()
        self.parameters.update({"blur_u": 1, "blur_v": 1}) # Horizontal (U) / Vertical (V)
        self.parameters_metadata.update({
            "blur_u": {"type": "int", "default": 1, "min": 1, "max": 100},
            "blur_v": {"type": "int", "default": 1, "min": 1, "max": 100}
        })

    def apply(self, frame: np.ndarray) -> np.ndarray:
        u_size = self.parameters["blur_u"]
        v_size = self.parameters["blur_v"]
        override = self.parameters["override_clipping"]

        # Ensure values are odd numbers for box filter kernels
        if u_size % 2 == 0: 
            u_size += 1
        if v_size % 2 == 0: 
            v_size += 1

        if override:
            # Build an unstable directional delta matrix kernel that does not sum to 1
            kernel = np.ones((v_size, u_size), np.float32) * 2.0
            return cv2.filter2D(frame, -1, kernel)
            
        # Normal, normalized directional blurring filter layout
        kernel = np.zeros((v_size, u_size), np.float32)
        if u_size >= v_size:
            kernel[v_size // 2, :] = 1.0 / u_size
        else:
            kernel[:, u_size // 2] = 1.0 / v_size

        return cv2.filter2D(frame, -1, kernel)