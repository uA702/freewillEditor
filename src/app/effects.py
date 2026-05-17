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
            if "min" in meta and "max" in meta:
                value = max(min(value, meta["max"]), meta["min"])
            self.parameters[name] = value
    
    def apply(self, frame: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# --- 1 & 2: BRIGHTNESS & BRIGHTNESS WITH OVERFLOW ---
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
            # Force intermediate math to float32, then mask to uint8 for overflow art
            res = (frame.astype(np.float32) * factor).astype(np.int32)
            return (res & 0xFF).astype(np.uint8)
        else:
            return np.clip(frame * factor, 0, 255).astype(np.uint8)


# --- 3: CONTRAST ---
class ContrastEffect(BaseEffect):
    def __init__(self):
        super().__init__()
        self.parameters.update({"contrast": 100})  
        self.parameters_metadata.update({
            "contrast": {"type": "int", "default": 100, "min": 0, "max": 300}
        })

    def apply(self, frame: np.ndarray) -> np.ndarray:
        factor = self.parameters["contrast"] / 100.0
        override = self.parameters["override_clipping"]
        
        result = 128.0 + factor * (frame.astype(np.float32) - 128.0)
        
        if override:
            # Convert float matrix to int32 before masking out bounds
            return (result.astype(np.int32) & 0xFF).astype(np.uint8)
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

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= factor

        if override:
            # Mask the Saturation layer explicitly to avoid conversion errors back to BGR
            hsv_out = (hsv.astype(np.int32) & 0xFF).astype(np.uint8)
            return cv2.cvtColor(hsv_out, cv2.COLOR_HSV2BGR)
        
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# --- 5: COLOR TEMPERATURE ---
class ColorTempEffect(BaseEffect):
    def __init__(self):
        super().__init__()
        self.parameters.update({"temp": 0})  
        self.parameters_metadata.update({
            "temp": {"type": "int", "default": 0, "min": -100, "max": 100}
        })

    def apply(self, frame: np.ndarray) -> np.ndarray:
        shift = self.parameters["temp"]
        override = self.parameters["override_clipping"]
        
        result = frame.astype(np.float32)
        if shift > 0:
            result[:, :, 2] += shift  
            result[:, :, 0] -= shift * 0.5  
        else:
            result[:, :, 0] -= shift  
            result[:, :, 2] += shift * 0.5  

        if override:
            return (result.astype(np.int32) & 0xFF).astype(np.uint8)
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
                    res_ch = (result[:, :, ch].astype(np.float32) * factor).astype(np.int32)
                    result[:, :, ch] = (res_ch & 0xFF).astype(np.uint8)
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
            "channel": {"type": "int", "default": 0, "min": 0, "max": 3}
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
                    # Explicit integer widening prevents python scale boundary errors
                    res_ch = result[:, :, ch].astype(np.int32) + addend
                    result[:, :, ch] = (res_ch & 0xFF).astype(np.uint8)
                else:
                    res_ch = result[:, :, ch].astype(np.int32) + addend
                    result[:, :, ch] = np.clip(res_ch, 0, 255).astype(np.uint8)
            else: 
                if override:
                    # RGB wide matrix calculation
                    res_mat = result.astype(np.int32) + addend
                    result = (res_mat & 0xFF).astype(np.uint8)
                else:
                    res_ch = result.astype(np.int32) + addend
                    result = np.clip(res_ch, 0, 255).astype(np.uint8)
        return result


# --- 8: MONOCHROME ---
class MonochromeEffect(BaseEffect):
    def __init__(self):
        super().__init__()

    def apply(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# --- 9: GRADIENT COLOR MAPPING ---
class ColorMappingEffect(BaseEffect):
    def __init__(self):
        super().__init__()
        self.parameters.update({"colormap": 0})
        self.parameters_metadata.update({
            "colormap": {"type": "int", "default": 0, "min": 0, "max": 11}
        })
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
            res_gray = gray.astype(np.int32) + 100
            gray = (res_gray & 0xFF).astype(np.uint8)

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
            edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR).astype(np.int32)
            res = frame.astype(np.int32) + edge_bgr
            return (res & 0xFF).astype(np.uint8)
            
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


# --- 11: THRESHOLDING ---
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
            res_gray = (gray.astype(np.int32) // (val + 1)) * 50
            return cv2.cvtColor((res_gray & 0xFF).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        _, binary = cv2.threshold(gray, val, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


# --- 12: DIRECTIONAL U/V BLURRING ---
class DirectionalBlurEffect(BaseEffect):
    def __init__(self):
        super().__init__()
        self.parameters.update({"blur_u": 1, "blur_v": 1}) 
        self.parameters_metadata.update({
            "blur_u": {"type": "int", "default": 1, "min": 1, "max": 100},
            "blur_v": {"type": "int", "default": 1, "min": 1, "max": 100}
        })

    def apply(self, frame: np.ndarray) -> np.ndarray:
        u_size = self.parameters["blur_u"]
        v_size = self.parameters["blur_v"]
        override = self.parameters["override_clipping"]

        if u_size % 2 == 0: 
            u_size += 1
        if v_size % 2 == 0: 
            v_size += 1

        if override:
            kernel = np.ones((v_size, u_size), np.float32) * 2.0
            # filter2D handles internal float accumulation safely, but we cast out bounds cleanly here
            res = cv2.filter2D(frame.astype(np.float32), -1, kernel).astype(np.int32)
            return (res & 0xFF).astype(np.uint8)
            
        kernel = np.zeros((v_size, u_size), np.float32)
        if u_size >= v_size:
            kernel[v_size // 2, :] = 1.0 / u_size
        else:
            kernel[:, u_size // 2] = 1.0 / v_size

        return cv2.filter2D(frame, -1, kernel)