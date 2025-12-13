import cv2
import numpy as np
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum


# ==================== NOTEBOOK / UTILITY HELPERS ====================
def iter_image_files(
    input_dir: str,
    exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".heic", ".heif"),
) -> List[Path]:
    """Return image files in a directory (non-recursive), filtered by extension."""
    input_path = Path(input_dir)
    if not input_path.exists():
        return []

    exts_lower = tuple(e.lower() for e in exts)
    files: List[Path] = []
    for p in input_path.iterdir():
        if p.is_file() and p.suffix.lower() in exts_lower:
            files.append(p)
    return sorted(files)


def load_image_bgr(image_path: str) -> np.ndarray:
    """Load an image from disk using OpenCV (BGR). Raises on failure."""
    image_path_str = str(image_path)
    img = cv2.imread(image_path_str)
    if img is not None:
        return img

    suffix = Path(image_path_str).suffix.lower()
    if suffix in (".heic", ".heif"):
        # OpenCV often can't read HEIC directly; fall back to pillow-heif + PIL.
        try:
            import pillow_heif  # type: ignore
            from PIL import Image  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Reading HEIC/HEIF requires pillow-heif and Pillow. "
                "Install with: pip install pillow-heif pillow"
            ) from e

        try:
            pillow_heif.register_heif_opener()
            pil_img = Image.open(image_path_str).convert("RGB")
            rgb = np.array(pil_img)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise FileNotFoundError(f"Could not read HEIC/HEIF image: {image_path}") from e

    raise FileNotFoundError(f"Could not read image: {image_path}")


def location_label_to_bbox(
    location: str,
    image_shape: Tuple[int, int],
    frac: float = 0.33,
) -> Tuple[int, int, int, int]:
    """Map coarse location labels (e.g. 'top', 'bottom-left') to an approximate bbox.

    This is meant for visualization when labels are coarse and do not include pixel coordinates.
    """
    h, w = int(image_shape[0]), int(image_shape[1])
    loc = (location or "").strip().lower()

    third_w = max(1, int(w * frac))
    third_h = max(1, int(h * frac))
    half_w = max(1, int(w * 0.5))
    half_h = max(1, int(h * 0.5))

    if loc in ("top", "upper"):
        return (0, 0, w, third_h)
    if loc in ("bottom", "lower"):
        return (0, h - third_h, w, third_h)
    if loc == "left":
        return (0, 0, third_w, h)
    if loc == "right":
        return (w - third_w, 0, third_w, h)
    if loc in ("center", "middle"):
        return (w // 4, h // 4, w // 2, h // 2)

    if loc in ("top-left", "upper-left"):
        return (0, 0, half_w, half_h)
    if loc in ("top-right", "upper-right"):
        return (w - half_w, 0, half_w, half_h)
    if loc in ("bottom-left", "lower-left"):
        return (0, h - half_h, half_w, half_h)
    if loc in ("bottom-right", "lower-right"):
        return (w - half_w, h - half_h, half_w, half_h)

    # Unknown / missing label: cover whole image.
    return (0, 0, w, h)


def load_simple_defects_json(
    json_path: str,
    image_shape: Optional[Tuple[int, int]] = None,
) -> Dict:
    """Load the user's simple annotation JSON format.

    Expected structure:
    {
      "image": "img101.jpg",
      "defects": [
        {"component": "seal", "type": "missing", "location": "top"}
      ]
    }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # If requested, normalize location labels into bbox-like dicts.
    if image_shape is not None and isinstance(data, dict) and isinstance(data.get("defects"), list):
        h, w = int(image_shape[0]), int(image_shape[1])
        normalized = []
        for item in data["defects"]:
            if not isinstance(item, dict):
                continue
            loc = item.get("location")
            if isinstance(loc, str):
                x, y, bw, bh = location_label_to_bbox(loc, (h, w))
                item = dict(item)
                item["location_bbox"] = {"x": x, "y": y, "width": bw, "height": bh}
            normalized.append(item)
        data = dict(data)
        data["defects"] = normalized

    return data


def simple_json_to_defects(
    data: Dict,
    image_shape: Tuple[int, int],
    default_confidence: float = 1.0,
) -> List["Defect"]:
    """Convert the simple annotation JSON dict to a list of Defect objects.

    Coarse string locations are mapped to approximate bounding boxes.
    """
    defects: List[Defect] = []
    if not isinstance(data, dict):
        return defects

    items = data.get("defects", [])
    if not isinstance(items, list):
        return defects

    h, w = int(image_shape[0]), int(image_shape[1])
    for item in items:
        if not isinstance(item, dict):
            continue

        component = str(item.get("component", "unknown"))
        defect_type = str(item.get("type", "unknown"))
        loc = item.get("location", "")
        if isinstance(loc, str):
            x, y, bw, bh = location_label_to_bbox(loc, (h, w))
            desc = f"Annotation: {component} / {defect_type} @ {loc}"
        else:
            x, y, bw, bh = (0, 0, w, h)
            desc = f"Annotation: {component} / {defect_type}"

        defects.append(
            Defect(
                component=component,
                defect_type=defect_type,
                location=(int(x), int(y), int(bw), int(bh)),
                confidence=float(default_confidence),
                description=desc,
            )
        )

    return defects


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR image (OpenCV) to RGB (matplotlib/PIL-friendly)."""
    if image_bgr is None:
        return image_bgr
    if len(image_bgr.shape) == 2:
        return image_bgr
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def show_image(
    image_bgr: np.ndarray,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
):
    """Display an OpenCV image inside Jupyter (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for show_image() in notebooks. "
            "Install it with: pip install matplotlib"
        ) from e

    plt.figure(figsize=figsize)
    plt.imshow(bgr_to_rgb(image_bgr))
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


def defects_to_dict(image_name: str, defects: List["Defect"]) -> Dict:
    """Convert defect list to a JSON-serializable dict (no file I/O)."""
    return {
        "image": image_name,
        "defects": [
            {
                "component": d.component,
                "type": d.defect_type,
                "location": {
                    "x": int(d.location[0]),
                    "y": int(d.location[1]),
                    "width": int(d.location[2]),
                    "height": int(d.location[3]),
                },
                "confidence": float(d.confidence),
                "description": d.description,
            }
            for d in defects
        ],
    }


def run_detector_on_path(
    image_path: str,
    detector: Optional["FacadeDefectDetector"] = None,
    component_mask: Optional[np.ndarray] = None,
) -> Tuple[List["Defect"], np.ndarray, Dict]:
    """Notebook-friendly one-liner: load image, detect defects, return (defects, viz, json_dict)."""
    detector = detector or FacadeDefectDetector()
    image = load_image_bgr(image_path)
    defects = detector.detect_all_defects(image, component_mask=component_mask)
    viz = detector.visualize_defects(image, defects)
    results_dict = defects_to_dict(Path(image_path).name, defects)
    return defects, viz, results_dict


class DefectType(Enum):
    MISSING = "missing"           # Nedostaje
    DAMAGED = "damaged"           # Oštećen
    LOOSE = "loose"               # Na pola pritegnut
    SHIFTED = "shifted"           # Pomaknuta
    EXCESS = "excess"             # Višak
    CRACK = "crack"               # Pukotina
    TORN = "torn"                 # Strgana


class ComponentType(Enum):
    SHEET_METAL = "sheet_metal"   # Lim
    SCREW = "screw"               # Vijak
    HOLE = "hole"                 # Rupa
    GLASS = "glass"               # Staklo
    SEAL = "seal"                 # Brtva


@dataclass
class Defect:
    component: str
    defect_type: str
    location: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    description: str


class FacadeDefectDetector:
    """OpenCV-based defect detector for facade components."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.results = []
        
    def _default_config(self) -> Dict:
        return {
            "canny_low": 50,
            "canny_high": 150,
            "blur_kernel": 5,
            "min_contour_area": 100,
            "crack_threshold": 30,
            "screw_min_radius": 5,
            "screw_max_radius": 30,
            "hole_min_circularity": 0.7,
        }
    
    def detect_all_defects(self, image: np.ndarray, component_mask: Optional[np.ndarray] = None) -> List[Defect]:
        """Run all defect detection methods on the image."""
        defects = []
        
        # Detect cracks (for glass and sheet metal)
        defects.extend(self.detect_cracks(image, component_mask))
        
        # Detect missing/loose screws
        defects.extend(self.detect_screw_defects(image, component_mask))
        
        # Detect hole defects
        defects.extend(self.detect_hole_defects(image, component_mask))
        
        # Detect seal defects
        defects.extend(self.detect_seal_defects(image, component_mask))
        
        # Detect surface damage
        defects.extend(self.detect_surface_damage(image, component_mask))
        
        self.results = defects
        return defects
    
    # ==================== CRACK DETECTION ====================
    def detect_cracks(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> List[Defect]:
        """
        Detektira pukotine na staklu i limu.
        Uses edge detection and line detection to find cracks.
        """
        defects = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Apply mask if provided
        if mask is not None:
            gray = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (self.config["blur_kernel"], self.config["blur_kernel"]), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, self.config["canny_low"], self.config["canny_high"])
        
        # Morphological operations to connect crack segments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config["min_contour_area"]:
                continue
                
            # Calculate aspect ratio - cracks are typically elongated
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
            
            # Cracks have high aspect ratio
            if aspect_ratio > 3:
                # Calculate arc length vs area ratio
                arc_length = cv2.arcLength(contour, True)
                linearity = arc_length / (area + 1e-6)
                
                if linearity > 0.1:  # High linearity suggests crack
                    confidence = min(1.0, aspect_ratio / 10 * linearity)
                    defects.append(Defect(
                        component="glass",
                        defect_type="crack",
                        location=(x, y, w, h),
                        confidence=confidence,
                        description=f"Pukotina detektirana - aspect ratio: {aspect_ratio:.2f}"
                    ))
        
        return defects
    
    # ==================== SCREW DETECTION ====================
    def detect_screw_defects(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> List[Defect]:
        """
        Detektira vijke i njihove defekte (nedostaje, na pola pritegnut).
        Uses Hough Circle detection and template matching.
        """
        defects = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        if mask is not None:
            gray = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Blur for circle detection
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detect circles (potential screw heads)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=self.config["screw_min_radius"],
            maxRadius=self.config["screw_max_radius"]
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            for circle in circles[0, :]:
                x, y, r = circle
                
                # Extract ROI around screw
                roi = self._extract_roi(gray, x, y, r * 2)
                if roi is None:
                    continue
                
                # Analyze screw head pattern
                screw_state = self._analyze_screw_state(roi, image, x, y, r)
                
                if screw_state == "loose":
                    defects.append(Defect(
                        component="screw",
                        defect_type="loose",
                        location=(int(x - r), int(y - r), int(r * 2), int(r * 2)),
                        confidence=0.7,
                        description="Vijak na pola pritegnut"
                    ))
                elif screw_state == "missing":
                    defects.append(Defect(
                        component="screw",
                        defect_type="missing",
                        location=(int(x - r), int(y - r), int(r * 2), int(r * 2)),
                        confidence=0.8,
                        description="Vijak nedostaje"
                    ))
        
        return defects
    
    def _analyze_screw_state(self, roi: np.ndarray, full_image: np.ndarray, 
                             x: int, y: int, r: int) -> str:
        """Analyze if screw is properly tightened, loose, or missing."""
        if roi is None or roi.size == 0:
            return "unknown"
        
        # Calculate variance - loose screws often have shadows
        variance = np.var(roi)
        
        # Check for cross pattern (screw head slot)
        edges = cv2.Canny(roi, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Simple heuristic
        if edge_density < 0.05:
            return "missing"  # No features = might be empty hole
        elif variance > 2000:  # High variance might indicate shadow from loose screw
            return "loose"
        
        return "ok"
    
    def _extract_roi(self, image: np.ndarray, cx: int, cy: int, size: int) -> Optional[np.ndarray]:
        """Extract region of interest around a point."""
        h, w = image.shape[:2]
        half = size // 2
        
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, cx + half)
        y2 = min(h, cy + half)
        
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None
            
        return image[y1:y2, x1:x2]
    
    # ==================== HOLE DETECTION ====================
    def detect_hole_defects(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> List[Defect]:
        """
        Detektira defekte rupa (nedostaje, pomaknuta, višak).
        Uses blob detection and template matching.
        """
        defects = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        if mask is not None:
            gray = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Setup SimpleBlobDetector
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 50
        params.maxArea = 5000
        params.filterByCircularity = True
        params.minCircularity = self.config["hole_min_circularity"]
        params.filterByConvexity = True
        params.minConvexity = 0.8
        params.filterByInertia = True
        params.minInertiaRatio = 0.5
        
        detector = cv2.SimpleBlobDetector_create(params)
        
        # Detect blobs (potential holes)
        keypoints = detector.detect(gray)
        
        # Analyze detected holes
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            r = int(kp.size / 2)
            
            # Check if hole position is correct (would need reference template)
            # For now, just report detected holes
            defects.append(Defect(
                component="hole",
                defect_type="detected",
                location=(x - r, y - r, r * 2, r * 2),
                confidence=0.6,
                description=f"Rupa detektirana na poziciji ({x}, {y})"
            ))
        
        return defects
    
    # ==================== SEAL DETECTION ====================
    def detect_seal_defects(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> List[Defect]:
        """
        Detektira defekte brtve (nedostaje, oštećena/strgana).
        Uses color segmentation and contour analysis.
        """
        defects = []
        
        if len(image.shape) == 2:
            return defects  # Need color image for seal detection
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Seals are often black or dark rubber
        # Define range for dark colors
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 50])
        
        dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
        
        if mask is not None:
            dark_mask = cv2.bitwise_and(dark_mask, dark_mask, mask=mask)
        
        # Find contours of dark regions
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:  # Too small
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate contour smoothness
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            # Damaged seals have irregular edges
            regularity = len(approx)
            
            # Analyze edge regularity
            if regularity > 20:  # Very irregular edge
                defects.append(Defect(
                    component="seal",
                    defect_type="torn",
                    location=(x, y, w, h),
                    confidence=min(1.0, regularity / 30),
                    description=f"Brtva oštećena/strgana - nepravilni rubovi"
                ))
            
            # Check for gaps in seal
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / (hull_area + 1e-6)
            
            if solidity < 0.6:  # Low solidity indicates gaps
                defects.append(Defect(
                    component="seal",
                    defect_type="damaged",
                    location=(x, y, w, h),
                    confidence=1 - solidity,
                    description=f"Brtva s rupama/prazninama - solidnost: {solidity:.2f}"
                ))
        
        return defects
    
    # ==================== SURFACE DAMAGE DETECTION ====================
    def detect_surface_damage(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> List[Defect]:
        """
        Detektira oštećenja površine (za lim i staklo).
        Uses texture analysis and anomaly detection.
        """
        defects = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        if mask is not None:
            gray = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Calculate Local Binary Pattern for texture analysis
        lbp = self._compute_lbp(gray)
        
        # Find texture anomalies using variance
        kernel_size = 15
        local_mean = cv2.blur(gray.astype(float), (kernel_size, kernel_size))
        local_var = cv2.blur((gray.astype(float) - local_mean) ** 2, (kernel_size, kernel_size))
        
        # Threshold for high variance regions (potential damage)
        threshold = np.mean(local_var) + 2 * np.std(local_var)
        anomaly_mask = (local_var > threshold).astype(np.uint8) * 255
        
        # Find contours of anomalies
        contours, _ = cv2.findContours(anomaly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 200:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            defects.append(Defect(
                component="sheet_metal",
                defect_type="damaged",
                location=(x, y, w, h),
                confidence=0.6,
                description="Oštećenje površine detektirano"
            ))
        
        return defects
    
    def _compute_lbp(self, gray: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """Compute Local Binary Pattern for texture analysis."""
        h, w = gray.shape
        lbp = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = gray[i, j]
                code = 0
                
                # Sample points around center
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = j + radius * np.cos(angle)
                    y = i - radius * np.sin(angle)
                    
                    # Bilinear interpolation
                    x1, y1 = int(x), int(y)
                    x2, y2 = min(x1 + 1, w - 1), min(y1 + 1, h - 1)
                    
                    dx, dy = x - x1, y - y1
                    value = (1 - dx) * (1 - dy) * gray[y1, x1] + \
                            dx * (1 - dy) * gray[y1, x2] + \
                            (1 - dx) * dy * gray[y2, x1] + \
                            dx * dy * gray[y2, x2]
                    
                    if value >= center:
                        code |= (1 << k)
                
                lbp[i, j] = code
        
        return lbp
    
    # ==================== VISUALIZATION ====================
    def visualize_defects(self, image: np.ndarray, defects: Optional[List[Defect]] = None) -> np.ndarray:
        """Draw detected defects on the image."""
        if defects is None:
            defects = self.results
        
        result = image.copy()
        
        # Color mapping for defect types
        colors = {
            "crack": (0, 0, 255),      # Red
            "missing": (255, 0, 0),     # Blue
            "loose": (0, 255, 255),     # Yellow
            "damaged": (0, 165, 255),   # Orange
            "torn": (255, 0, 255),      # Magenta
            "shifted": (255, 255, 0),   # Cyan
            "excess": (0, 255, 0),      # Green
            "detected": (128, 128, 128) # Gray
        }
        
        for defect in defects:
            x, y, w, h = defect.location
            color = colors.get(defect.defect_type, (255, 255, 255))
            
            # Draw rectangle
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{defect.component}: {defect.defect_type} ({defect.confidence:.2f})"
            cv2.putText(result, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return result
    
    def save_results(self, output_path: str, image_name: str):
        """Save detection results to JSON format."""
        results_dict = {
            "image": image_name,
            "defects": [
                {
                    "component": d.component,
                    "type": d.defect_type,
                    "location": {
                        "x": d.location[0],
                        "y": d.location[1],
                        "width": d.location[2],
                        "height": d.location[3]
                    },
                    "confidence": d.confidence,
                    "description": d.description
                }
                for d in self.results
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)
        
        return results_dict


# ==================== REFERENCE TEMPLATE MATCHING ====================
class TemplateBasedDefectDetector:
    """
    Detect defects by comparing with reference (good) templates.
    Useful for detecting missing components or positional shifts.
    """
    
    def __init__(self, template_dir: str):
        self.template_dir = Path(template_dir)
        self.templates = {}
        self._load_templates()
    
    def _load_templates(self):
        """Load reference templates for each component type."""
        for template_file in self.template_dir.glob("*.png"):
            name = template_file.stem
            self.templates[name] = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
    
    def detect_missing_component(self, image: np.ndarray, component_name: str, 
                                  expected_location: Tuple[int, int]) -> Optional[Defect]:
        """Check if a component is missing at expected location."""
        if component_name not in self.templates:
            return None
        
        template = self.templates[component_name]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Template matching
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        h, w = template.shape
        
        # Check if match is at expected location
        if max_val < 0.7:  # Low match score
            return Defect(
                component=component_name,
                defect_type="missing",
                location=(expected_location[0], expected_location[1], w, h),
                confidence=1 - max_val,
                description=f"Komponenta {component_name} nedostaje na očekivanoj poziciji"
            )
        
        # Check if position is shifted
        distance = np.sqrt((max_loc[0] - expected_location[0])**2 + 
                          (max_loc[1] - expected_location[1])**2)
        
        if distance > 20:  # Position shifted significantly
            return Defect(
                component=component_name,
                defect_type="shifted",
                location=(max_loc[0], max_loc[1], w, h),
                confidence=min(1.0, distance / 100),
                description=f"Komponenta {component_name} pomaknuta za {distance:.1f} piksela"
            )
        
        return None


# ==================== MAIN PROCESSING FUNCTION ====================
def process_facade_images(input_dir: str, output_dir: str, 
                          annotations_file: Optional[str] = None):
    """
    Process all facade images and detect defects.
    
    Args:
        input_dir: Directory containing facade images
        output_dir: Directory to save results
        annotations_file: Optional JSON file with component locations
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    detector = FacadeDefectDetector()
    
    # Load annotations if provided
    annotations = {}
    if annotations_file and os.path.exists(annotations_file):
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
    
    all_results = []
    
    # Process each image (common formats)
    image_files = iter_image_files(str(input_path))
    for img_file in image_files:
        print(f"Processing: {img_file.name}")
        
        image = cv2.imread(str(img_file))
        if image is None:
            continue
        
        # Detect defects
        defects = detector.detect_all_defects(image)
        
        # Visualize results
        result_image = detector.visualize_defects(image, defects)
        
        # Save visualized image
        output_img_path = output_path / f"defects_{img_file.name}"
        cv2.imwrite(str(output_img_path), result_image)
        
        # Save JSON results
        json_path = output_path / f"{img_file.stem}_defects.json"
        results = detector.save_results(str(json_path), img_file.name)
        all_results.append(results)
        
        print(f"  Found {len(defects)} potential defects")
    
    # Save summary
    summary_path = output_path / "all_defects_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    return all_results


# ==================== DEMO / TEST ====================
if __name__ == "__main__":
    # Example usage
    print("Facade Defect Detection System")
    print("=" * 50)
    
    # Check if we have test images
    test_dirs = [
        "TRAIN_JPG/positive",
        "TRAIN_JPG/negative", 
        "segmented_output/positive",
        "segmented_output/negative"
    ]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            print(f"\nProcessing images from: {test_dir}")
            output_dir = f"defect_results/{os.path.basename(test_dir)}"
            
            try:
                results = process_facade_images(test_dir, output_dir)
                print(f"Processed {len(results)} images")
            except Exception as e:
                print(f"Error processing {test_dir}: {e}")
    
    # Demo with single image
    print("\n" + "=" * 50)
    print("To process a single image:")
    print("""
    detector = FacadeDefectDetector()
    image = cv2.imread("your_image.jpg")
    defects = detector.detect_all_defects(image)
    result = detector.visualize_defects(image, defects)
    cv2.imwrite("result.jpg", result)
    """)
