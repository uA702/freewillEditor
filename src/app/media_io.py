import os
import cv2
import numpy as np

class MediaIOManager:
    """Centralized Media I/O framework supporting unified Image & Video operations."""
    
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif'}
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

    @classmethod
    def get_media_type(cls, file_path: str) -> str:
        """Determines if a given filename points to a static image, video file, or is invalid."""
        _, ext = os.path.splitext(file_path.lower())
        if ext in cls.IMAGE_EXTENSIONS:
            return "IMAGE"
        elif ext in cls.VIDEO_EXTENSIONS:
            return "VIDEO"
        return "UNKNOWN"

    @classmethod
    def import_image(cls, file_path: str, preserve_alpha: bool = True) -> np.ndarray:
        """
        Reads an image file from disk and standardizes it into a pipeline-safe matrix.
        
        Args:
            file_path (str): Path to target image asset.
            preserve_alpha (bool): If True, retains alpha transparent layer (BGRA format).
                                   If False, coerces asset into a 3-channel matrix (BGR).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image asset missing at destination: {file_path}")

        # cv2.IMREAD_UNCHANGED ensures alpha channels (.png/.webp) are not discarded by default
        flags = cv2.IMREAD_UNCHANGED if preserve_alpha else cv2.IMREAD_COLOR
        image = cv2.imread(file_path, flags)

        if image is None:
            raise IOError(f"File matrix corrupted or format unsupported: {file_path}")

        # Normalized multi-channel structural safety logic
        if len(image.shape) == 2:  # Single channel grayscale fallback
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4 and not preserve_alpha:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        return image

    @classmethod
    def export_image(cls, file_path: str, frame: np.ndarray, quality: int = 95) -> bool:
        """
        Saves a raw matrix frame as a highly optimized static compressed image.
        
        Args:
            file_path (str): Destination path including target extension type.
            frame (np.ndarray): The processed pipeline matrix frame.
            quality (int): Compression factor logic. 
                           Images (JPEG/WebP): 1 to 100 (Higher is crispier)
                           PNG: 0 to 9 (Higher means more compression time)
        """
        _, ext = os.path.splitext(file_path.lower())
        params = []

        if ext in {'.jpg', '.jpeg'}:
            # Standardize: JPEG has no alpha channel. Strip safely before saving.
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            
        elif ext == '.png':
            # Map quality factor (1-100) down to standard PNG compression ranges (0-9)
            png_comp = int((100 - quality) / 11.1)
            params = [cv2.IMWRITE_PNG_COMPRESSION, np.clip(png_comp, 0, 9)]
            
        elif ext == '.webp':
            params = [cv2.IMWRITE_WEBP_QUALITY, quality]

        # Return True if write succeeded, False if permission/system failure occurred
        return cv2.imwrite(file_path, frame, params)

    @classmethod
    def convert_color_space(cls, frame: np.ndarray, direction: str = "BGR2RGB") -> np.ndarray:
        """
        Helper interface converting pipeline channels to standard engine UI rendering spaces.
        Useful when updating UI previews (PyQt/PySide/Tkinter/Pillow) from raw frame matrices.
        """
        channels = frame.shape[2]
        
        if direction == "BGR2RGB":
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA if channels == 4 else cv2.COLOR_BGR2RGB)
        elif direction == "RGB2BGR":
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA if channels == 4 else cv2.COLOR_RGB2BGR)
        return frame