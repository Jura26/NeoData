"""
OpenCV-based Defect Detection for Facade Elements

This script:
1. Learns baseline features from POSITIVE images (expected screw/hole counts and positions)
2. Analyzes NEGATIVE images and compares against JSON metadata
3. Outputs a confusion/accuracy report

Supports: screws, holes, tin (metal sheet), seal, glass cracks
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# Try to import pillow_heif for HEIC support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    from PIL import Image
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False
    print("Warning: pillow_heif not installed. HEIC files won't be supported.")
    print("Install with: pip install pillow-heif")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    # Paths - adjust these to your setup
    train_positive_dir: str = r"TRAIN\positive"
    train_negative_dir: str = r"TRAIN\negative"
    train_jpg_positive_dir: str = r"TRAIN_JPG\positive"
    train_jpg_negative_dir: str = r"TRAIN_JPG\negative"
    
    # Output
    output_dir: str = "opencv_results"
    save_debug_images: bool = True
    
    # Screw detection parameters (SimpleBlobDetector)
    screw_min_area: int = 50
    screw_max_area: int = 2000
    screw_min_circularity: float = 0.5
    screw_min_convexity: float = 0.6
    screw_min_inertia: float = 0.3
    
    # Hole detection parameters
    hole_min_area: int = 100
    hole_max_area: int = 5000
    hole_min_circularity: float = 0.6
    
    # Seal detection parameters
    seal_min_length: int = 100
    seal_aspect_ratio_thresh: float = 5.0  # length/width ratio
    
    # Glass crack detection
    crack_canny_low: int = 50
    crack_canny_high: int = 150
    crack_min_line_length: int = 30
    crack_max_line_gap: int = 10
    
    # Location quadrant mapping (normalized 0-1 coordinates)
    # Format: (x1, y1, x2, y2) where 0,0 is top-left
    location_zones: Dict = field(default_factory=lambda: {
        # Corners
        "top-left": (0.0, 0.0, 0.5, 0.33),
        "top-right": (0.5, 0.0, 1.0, 0.33),
        "bottom-left": (0.0, 0.66, 0.5, 1.0),
        "bottom-right": (0.5, 0.66, 1.0, 1.0),
        # Edges (full strips)
        "top": (0.0, 0.0, 1.0, 0.33),
        "bottom": (0.0, 0.66, 1.0, 1.0),
        "left": (0.0, 0.0, 0.33, 1.0),
        "right": (0.66, 0.0, 1.0, 1.0),
        # Center variations
        "center": (0.2, 0.2, 0.8, 0.8),  # Generous center
        "center-left": (0.0, 0.33, 0.5, 0.66),
        "center-right": (0.5, 0.33, 1.0, 0.66),
        "bottom-center": (0.33, 0.66, 0.66, 1.0),
        "top-center": (0.33, 0.0, 0.66, 0.33),
    })


# ============================================================================
# IMAGE LOADING
# ============================================================================

def load_image(path: str) -> Optional[np.ndarray]:
    """Load image supporting HEIC, JPG, PNG formats. Returns BGR."""
    path = str(path)
    ext = os.path.splitext(path)[1].lower()
    
    if ext in ['.heic', '.heif']:
        if not HEIC_SUPPORT:
            print(f"  Cannot load HEIC: {path} (pillow_heif not installed)")
            return None
        try:
            pil_img = Image.open(path)
            rgb = np.array(pil_img.convert('RGB'))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"  Error loading HEIC {path}: {e}")
            return None
    else:
        img = cv2.imread(path)
        if img is None:
            print(f"  Could not load: {path}")
        return img


def find_image_file(base_name: str, search_dirs: List[str]) -> Optional[str]:
    """Find image file matching base_name in search directories."""
    extensions = ['.jpg', '.jpeg', '.png', '.heic', '.heif', '.HEIC', '.JPG', '.PNG']
    
    # Try exact match first
    for dir_path in search_dirs:
        if not os.path.exists(dir_path):
            continue
        for ext in extensions:
            # Try with original name
            candidate = os.path.join(dir_path, base_name)
            if os.path.exists(candidate):
                return candidate
            # Try replacing extension
            candidate = os.path.join(dir_path, os.path.splitext(base_name)[0] + ext)
            if os.path.exists(candidate):
                return candidate
    
    return None


# ============================================================================
# FEATURE DETECTORS
# ============================================================================

class ScrewDetector:
    """Detect screw heads using blob detection and circle detection."""
    
    def __init__(self, config: Config):
        self.config = config
        self._setup_blob_detector()
    
    def _setup_blob_detector(self):
        params = cv2.SimpleBlobDetector_Params()
        
        # Filter by area
        params.filterByArea = True
        params.minArea = self.config.screw_min_area
        params.maxArea = self.config.screw_max_area
        
        # Filter by circularity (screws are roughly circular)
        params.filterByCircularity = True
        params.minCircularity = self.config.screw_min_circularity
        
        # Filter by convexity
        params.filterByConvexity = True
        params.minConvexity = self.config.screw_min_convexity
        
        # Filter by inertia (roundness)
        params.filterByInertia = True
        params.minInertiaRatio = self.config.screw_min_inertia
        
        # Color - detect dark blobs (screws are often dark)
        params.filterByColor = False
        
        self.detector = cv2.SimpleBlobDetector_create(params)
    
    def detect(self, img: np.ndarray) -> List[Tuple[float, float, float]]:
        """
        Detect screws in image.
        Returns list of (x, y, radius) for each detection.
        """
        detections = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Method 1: Blob detection
        keypoints = self.detector.detect(gray)
        for kp in keypoints:
            detections.append((kp.pt[0], kp.pt[1], kp.size / 2))
        
        # Method 2: Hough circles (complementary)
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=3,
            maxRadius=25
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for c in circles[0, :]:
                # Check if this circle overlaps with existing detections
                is_duplicate = False
                for det in detections:
                    dist = np.sqrt((c[0] - det[0])**2 + (c[1] - det[1])**2)
                    if dist < max(c[2], det[2]) * 1.5:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    detections.append((float(c[0]), float(c[1]), float(c[2])))
        
        return detections


class HoleDetector:
    """Detect holes using contour analysis."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def detect(self, img: np.ndarray) -> List[Tuple[float, float, float]]:
        """
        Detect holes (circular openings) in image.
        Returns list of (x, y, radius) for each detection.
        """
        detections = []
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Edge detection
        edges = cv2.Canny(gray, 30, 100)
        
        # Dilate to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.hole_min_area or area > self.config.hole_max_area:
                continue
            
            # Check circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            if circularity >= self.config.hole_min_circularity:
                # Get center and radius
                (x, y), radius = cv2.minEnclosingCircle(contour)
                detections.append((float(x), float(y), float(radius)))
        
        return detections


class SealDetector:
    """Detect rubber seals using morphological analysis."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def detect(self, img: np.ndarray) -> List[Dict]:
        """
        Detect seals (long, thin dark regions).
        Returns list of dicts with 'contour', 'bbox', 'length', 'is_intact'.
        """
        detections = []
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Seals are typically dark - threshold to find dark regions
        _, dark_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Seals are long and thin
            length = max(w, h)
            width = min(w, h)
            
            if length < self.config.seal_min_length:
                continue
            
            if width > 0:
                aspect_ratio = length / width
                if aspect_ratio < self.config.seal_aspect_ratio_thresh:
                    continue
            
            # Check if seal is intact (continuous) or damaged (fragmented)
            area = cv2.contourArea(contour)
            rect_area = w * h
            fill_ratio = area / rect_area if rect_area > 0 else 0
            
            is_intact = fill_ratio > 0.3  # Continuous seal fills more of its bbox
            
            detections.append({
                'contour': contour,
                'bbox': (x, y, w, h),
                'length': length,
                'is_intact': is_intact,
                'center': (x + w/2, y + h/2)
            })
        
        return detections


class CrackDetector:
    """Detect glass cracks using line detection."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def detect(self, img: np.ndarray, glass_mask: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Detect cracks (long thin lines with high contrast).
        Returns list of dicts with 'line', 'length', 'center'.
        """
        detections = []
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Apply glass mask if provided
        if glass_mask is not None:
            gray = cv2.bitwise_and(gray, gray, mask=glass_mask)
        
        # Edge detection
        edges = cv2.Canny(
            gray, 
            self.config.crack_canny_low, 
            self.config.crack_canny_high
        )
        
        # Detect lines
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=self.config.crack_min_line_length,
            maxLineGap=self.config.crack_max_line_gap
        )
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                detections.append({
                    'line': (x1, y1, x2, y2),
                    'length': length,
                    'center': ((x1+x2)/2, (y1+y2)/2)
                })
        
        return detections


class TinDetector:
    """Detect tin/metal sheet regions using edge and color analysis."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def detect(self, img: np.ndarray) -> List[Dict]:
        """
        Detect tin/metal regions.
        Returns list of dicts with coverage info.
        """
        detections = []
        h, w = img.shape[:2]
        
        # Convert to HSV for better metal detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Metal is typically low saturation (grayish)
        # Create mask for metallic regions
        lower_metal = np.array([0, 0, 50])
        upper_metal = np.array([180, 50, 200])
        metal_mask = cv2.inRange(hsv, lower_metal, upper_metal)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        metal_mask = cv2.morphologyEx(metal_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        metal_mask = cv2.morphologyEx(metal_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(metal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_area = (w * h) * 0.01  # At least 1% of image
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            x, y, cw, ch = cv2.boundingRect(contour)
            
            detections.append({
                'contour': contour,
                'bbox': (x, y, cw, ch),
                'area': area,
                'coverage': area / (w * h),
                'center': (x + cw/2, y + ch/2)
            })
        
        return detections


# ============================================================================
# LOCATION MAPPING
# ============================================================================

def get_location_zone(x: float, y: float, img_width: int, img_height: int, 
                      zones: Dict) -> str:
    """Map (x, y) coordinates to a location zone name."""
    norm_x = x / img_width
    norm_y = y / img_height
    
    for zone_name, (x1, y1, x2, y2) in zones.items():
        if x1 <= norm_x <= x2 and y1 <= norm_y <= y2:
            return zone_name
    
    return "unknown"


def points_in_zone(points: List[Tuple], zone_name: str, img_width: int, 
                   img_height: int, zones: Dict) -> List[Tuple]:
    """Filter points that fall within a specific zone."""
    if zone_name not in zones:
        return []
    
    x1, y1, x2, y2 = zones[zone_name]
    x1_px, y1_px = x1 * img_width, y1 * img_height
    x2_px, y2_px = x2 * img_width, y2 * img_height
    
    return [p for p in points if x1_px <= p[0] <= x2_px and y1_px <= p[1] <= y2_px]


# ============================================================================
# BASELINE LEARNING (from POSITIVE images)
# ============================================================================

@dataclass
class BaselineFeatures:
    """Learned baseline features from POSITIVE images."""
    screw_count_mean: float = 0.0
    screw_count_std: float = 0.0
    screw_positions: List[Tuple] = field(default_factory=list)  # Normalized positions
    
    hole_count_mean: float = 0.0
    hole_count_std: float = 0.0
    hole_positions: List[Tuple] = field(default_factory=list)
    
    seal_present: bool = True
    seal_typical_length: float = 0.0
    
    tin_coverage_mean: float = 0.0
    tin_coverage_std: float = 0.0
    
    samples_analyzed: int = 0


def learn_baseline(config: Config) -> BaselineFeatures:
    """Learn baseline features from POSITIVE images."""
    print("\n" + "="*60)
    print("LEARNING BASELINE FROM POSITIVE IMAGES")
    print("="*60)
    
    baseline = BaselineFeatures()
    
    # Initialize detectors
    screw_det = ScrewDetector(config)
    hole_det = HoleDetector(config)
    seal_det = SealDetector(config)
    tin_det = TinDetector(config)
    
    # Collect statistics
    screw_counts = []
    hole_counts = []
    seal_lengths = []
    tin_coverages = []
    all_screw_positions = []
    all_hole_positions = []
    
    # Find positive images
    search_dirs = [
        config.train_jpg_positive_dir,
        config.train_positive_dir
    ]
    
    positive_dir = None
    for d in search_dirs:
        if os.path.exists(d):
            positive_dir = d
            break
    
    if not positive_dir:
        print("Warning: No positive directory found!")
        return baseline
    
    print(f"Searching in: {positive_dir}")
    
    extensions = ('.jpg', '.jpeg', '.png', '.heic', '.heif')
    image_files = [f for f in os.listdir(positive_dir) 
                   if f.lower().endswith(extensions)]
    
    print(f"Found {len(image_files)} positive images")
    
    for img_file in image_files:
        img_path = os.path.join(positive_dir, img_file)
        img = load_image(img_path)
        
        if img is None:
            continue
        
        h, w = img.shape[:2]
        print(f"\n  Processing: {img_file} ({w}x{h})")
        
        # Detect screws
        screws = screw_det.detect(img)
        screw_counts.append(len(screws))
        for s in screws:
            all_screw_positions.append((s[0]/w, s[1]/h))  # Normalized
        print(f"    Screws: {len(screws)}")
        
        # Detect holes
        holes = hole_det.detect(img)
        hole_counts.append(len(holes))
        for h_det in holes:
            all_hole_positions.append((h_det[0]/w, h_det[1]/h))
        print(f"    Holes: {len(holes)}")
        
        # Detect seals
        seals = seal_det.detect(img)
        if seals:
            seal_lengths.extend([s['length'] for s in seals])
        print(f"    Seal segments: {len(seals)}")
        
        # Detect tin
        tins = tin_det.detect(img)
        total_coverage = sum(t['coverage'] for t in tins)
        tin_coverages.append(total_coverage)
        print(f"    Tin coverage: {total_coverage:.1%}")
        
        baseline.samples_analyzed += 1
    
    # Compute statistics
    if screw_counts:
        baseline.screw_count_mean = np.mean(screw_counts)
        baseline.screw_count_std = np.std(screw_counts) if len(screw_counts) > 1 else 1.0
        baseline.screw_positions = all_screw_positions
    
    if hole_counts:
        baseline.hole_count_mean = np.mean(hole_counts)
        baseline.hole_count_std = np.std(hole_counts) if len(hole_counts) > 1 else 1.0
        baseline.hole_positions = all_hole_positions
    
    if seal_lengths:
        baseline.seal_typical_length = np.mean(seal_lengths)
        baseline.seal_present = True
    
    if tin_coverages:
        baseline.tin_coverage_mean = np.mean(tin_coverages)
        baseline.tin_coverage_std = np.std(tin_coverages) if len(tin_coverages) > 1 else 0.1
    
    print(f"\n{'='*60}")
    print("BASELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Samples analyzed: {baseline.samples_analyzed}")
    print(f"Screw count: {baseline.screw_count_mean:.1f} ± {baseline.screw_count_std:.1f}")
    print(f"Hole count: {baseline.hole_count_mean:.1f} ± {baseline.hole_count_std:.1f}")
    print(f"Seal typical length: {baseline.seal_typical_length:.0f}px")
    print(f"Tin coverage: {baseline.tin_coverage_mean:.1%} ± {baseline.tin_coverage_std:.1%}")
    
    return baseline


# ============================================================================
# DEFECT ANALYSIS
# ============================================================================

@dataclass
class DefectAnalysisResult:
    """Result of analyzing one image for defects."""
    image_name: str
    json_defects: List[Dict]  # From metadata
    detected_defects: List[Dict]  # From OpenCV
    matches: List[Dict]  # Matched json <-> detected
    false_negatives: List[Dict]  # In JSON but not detected
    false_positives: List[Dict]  # Detected but not in JSON
    debug_image: Optional[np.ndarray] = None


def analyze_image_for_defects(
    img: np.ndarray,
    img_name: str,
    json_defects: List[Dict],
    baseline: BaselineFeatures,
    config: Config
) -> DefectAnalysisResult:
    """Analyze single image for defects and compare against JSON metadata."""
    
    result = DefectAnalysisResult(
        image_name=img_name,
        json_defects=json_defects,
        detected_defects=[],
        matches=[],
        false_negatives=[],
        false_positives=[]
    )
    
    h, w = img.shape[:2]
    
    # Initialize detectors
    screw_det = ScrewDetector(config)
    hole_det = HoleDetector(config)
    seal_det = SealDetector(config)
    tin_det = TinDetector(config)
    crack_det = CrackDetector(config)
    
    # Run all detections
    screws = screw_det.detect(img)
    holes = hole_det.detect(img)
    seals = seal_det.detect(img)
    tins = tin_det.detect(img)
    cracks = crack_det.detect(img)
    
    # Create debug image
    debug_img = img.copy()
    
    # Draw detections on debug image
    for s in screws:
        cv2.circle(debug_img, (int(s[0]), int(s[1])), int(s[2]), (0, 255, 0), 2)
        cv2.putText(debug_img, "S", (int(s[0])-5, int(s[1])-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    for ho in holes:
        cv2.circle(debug_img, (int(ho[0]), int(ho[1])), int(ho[2]), (255, 0, 0), 2)
        cv2.putText(debug_img, "H", (int(ho[0])-5, int(ho[1])-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    for seal in seals:
        cv2.drawContours(debug_img, [seal['contour']], -1, (0, 255, 255), 2)
    
    for tin in tins:
        cv2.drawContours(debug_img, [tin['contour']], -1, (255, 0, 255), 1)
    
    for crack in cracks:
        x1, y1, x2, y2 = crack['line']
        cv2.line(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    result.debug_image = debug_img
    
    # Analyze each JSON defect
    for defect in json_defects:
        component = defect.get('component', '').lower()
        defect_type = defect.get('type', '').lower()
        location = defect.get('location', '').lower()
        
        detected = False
        detection_info = {}
        
        # ---- SCREW defects ----
        if component == 'screw':
            # Get screws in the specified zone
            screw_points = [(s[0], s[1]) for s in screws]
            screws_in_zone = points_in_zone(screw_points, location, w, h, config.location_zones)
            
            if defect_type == 'missing':
                # Missing = we expect screws but found fewer than baseline
                expected_in_zone = baseline.screw_count_mean / 4  # Rough estimate per quadrant
                if len(screws_in_zone) < expected_in_zone * 0.5:
                    detected = True
                    detection_info = {
                        'reason': f'Only {len(screws_in_zone)} screws in {location} (expected ~{expected_in_zone:.0f})'
                    }
            
            elif defect_type in ['half-tightened', 'loose']:
                # Would need more sophisticated analysis (size variation, etc.)
                detected = False
                detection_info = {'reason': 'Half-tightened detection not implemented'}
        
        # ---- HOLE defects ----
        elif component == 'hole':
            hole_points = [(ho[0], ho[1]) for ho in holes]
            holes_in_zone = points_in_zone(hole_points, location, w, h, config.location_zones)
            
            if defect_type == 'missing':
                expected_in_zone = baseline.hole_count_mean / 4
                if len(holes_in_zone) < expected_in_zone * 0.5:
                    detected = True
                    detection_info = {'reason': f'Missing holes in {location}'}
            
            elif defect_type == 'extra':
                expected_in_zone = baseline.hole_count_mean / 4
                if len(holes_in_zone) > expected_in_zone * 1.5:
                    detected = True
                    detection_info = {'reason': f'Extra holes in {location}'}
            
            elif defect_type == 'shifted':
                # Would compare against expected positions
                detected = False
                detection_info = {'reason': 'Shifted detection not implemented'}
        
        # ---- TIN defects ----
        elif component == 'tin':
            if defect_type == 'missing':
                # Check if tin coverage in zone is low
                zone_bounds = config.location_zones.get(location, (0, 0, 1, 1))
                zone_tins = [t for t in tins 
                            if zone_bounds[0] <= t['center'][0]/w <= zone_bounds[2]
                            and zone_bounds[1] <= t['center'][1]/h <= zone_bounds[3]]
                zone_coverage = sum(t['coverage'] for t in zone_tins)
                
                expected_zone_coverage = baseline.tin_coverage_mean / 4
                if zone_coverage < expected_zone_coverage * 0.3:
                    detected = True
                    detection_info = {'reason': f'Low tin coverage in {location}: {zone_coverage:.1%}'}
        
        # ---- SEAL defects ----
        elif component == 'seal':
            if defect_type in ['damaged', 'missing', 'torn']:
                # Check seal continuity in zone
                zone_bounds = config.location_zones.get(location, (0, 0, 1, 1))
                zone_seals = [s for s in seals
                             if zone_bounds[0] <= s['center'][0]/w <= zone_bounds[2]
                             and zone_bounds[1] <= s['center'][1]/h <= zone_bounds[3]]
                
                if not zone_seals:
                    detected = True
                    detection_info = {'reason': f'No seal detected in {location}'}
                elif defect_type in ['damaged', 'torn']:
                    # Check for fragmentation or short segments
                    damaged_seals = [s for s in zone_seals if not s['is_intact']]
                    short_seals = [s for s in zone_seals if s['length'] < baseline.seal_typical_length * 0.3]
                    if damaged_seals or short_seals:
                        detected = True
                        detection_info = {'reason': f'Fragmented/torn seal in {location} ({len(damaged_seals)} damaged, {len(short_seals)} short)'}
        
        # ---- GLASS defects ----
        elif component == 'glass':
            if defect_type in ['cracked', 'crack', 'damaged']:
                zone_bounds = config.location_zones.get(location, (0, 0, 1, 1))
                zone_cracks = [c for c in cracks
                              if zone_bounds[0] <= c['center'][0]/w <= zone_bounds[2]
                              and zone_bounds[1] <= c['center'][1]/h <= zone_bounds[3]]
                
                if zone_cracks:
                    detected = True
                    detection_info = {'reason': f'{len(zone_cracks)} crack lines in {location}'}
        
        # Record result
        result.detected_defects.append({
            'component': component,
            'type': defect_type,
            'location': location,
            'detected': detected,
            'info': detection_info
        })
        
        if detected:
            result.matches.append({
                'json_defect': defect,
                'detection_info': detection_info
            })
        else:
            result.false_negatives.append(defect)
    
    return result


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_negative_images(config: Config, baseline: BaselineFeatures) -> Dict:
    """Process all negative images and generate report."""
    
    print("\n" + "="*60)
    print("ANALYZING NEGATIVE IMAGES")
    print("="*60)
    
    results = []
    summary = {
        'total_images': 0,
        'total_defects': 0,
        'detected': 0,
        'missed': 0,
        'by_component': defaultdict(lambda: {'total': 0, 'detected': 0}),
        'by_type': defaultdict(lambda: {'total': 0, 'detected': 0}),
    }
    
    # Find negative JSON files
    negative_json_dir = config.train_negative_dir
    if not os.path.exists(negative_json_dir):
        print(f"Warning: Negative directory not found: {negative_json_dir}")
        return summary
    
    json_files = [f for f in os.listdir(negative_json_dir) if f.endswith('.json')]
    print(f"Found {len(json_files)} JSON metadata files")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    for json_file in json_files:
        json_path = os.path.join(negative_json_dir, json_file)
        
        # Load JSON metadata
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        
        image_name = metadata.get('image', '')
        defects = metadata.get('defects', [])
        
        if not image_name or not defects:
            continue
        
        # Find corresponding image
        search_dirs = [
            config.train_jpg_negative_dir,
            config.train_negative_dir,
            config.train_jpg_positive_dir,  # Sometimes might be misfiled
        ]
        
        img_path = find_image_file(image_name, search_dirs)
        
        if not img_path:
            print(f"\n  {json_file}: Image not found ({image_name})")
            continue
        
        # Load image
        img = load_image(img_path)
        if img is None:
            continue
        
        print(f"\n  Processing: {json_file}")
        print(f"    Image: {img_path}")
        print(f"    Defects in JSON: {len(defects)}")
        
        # Analyze
        result = analyze_image_for_defects(img, image_name, defects, baseline, config)
        results.append(result)
        
        # Update summary
        summary['total_images'] += 1
        summary['total_defects'] += len(defects)
        summary['detected'] += len(result.matches)
        summary['missed'] += len(result.false_negatives)
        
        for defect in defects:
            comp = defect.get('component', 'unknown')
            dtype = defect.get('type', 'unknown')
            summary['by_component'][comp]['total'] += 1
            summary['by_type'][dtype]['total'] += 1
        
        for match in result.matches:
            comp = match['json_defect'].get('component', 'unknown')
            dtype = match['json_defect'].get('type', 'unknown')
            summary['by_component'][comp]['detected'] += 1
            summary['by_type'][dtype]['detected'] += 1
        
        # Print result
        for det in result.detected_defects:
            status = "✓" if det['detected'] else "✗"
            print(f"    {status} {det['component']}/{det['type']} @ {det['location']}")
            if det['info']:
                print(f"      {det['info'].get('reason', '')}")
        
        # Save debug image
        if config.save_debug_images and result.debug_image is not None:
            debug_path = os.path.join(config.output_dir, f"debug_{os.path.splitext(json_file)[0]}.jpg")
            cv2.imwrite(debug_path, result.debug_image)
    
    return summary


def print_summary(summary: Dict):
    """Print final summary report."""
    
    print("\n" + "="*60)
    print("FINAL SUMMARY REPORT")
    print("="*60)
    
    total = summary['total_defects']
    detected = summary['detected']
    missed = summary['missed']
    
    if total > 0:
        accuracy = detected / total * 100
    else:
        accuracy = 0
    
    print(f"\nOverall Results:")
    print(f"  Images analyzed: {summary['total_images']}")
    print(f"  Total defects in JSON: {total}")
    print(f"  Defects detected: {detected}")
    print(f"  Defects missed: {missed}")
    print(f"  Detection rate: {accuracy:.1f}%")
    
    print(f"\nBy Component:")
    for comp, stats in summary['by_component'].items():
        rate = stats['detected'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {comp}: {stats['detected']}/{stats['total']} ({rate:.0f}%)")
    
    print(f"\nBy Defect Type:")
    for dtype, stats in summary['by_type'].items():
        rate = stats['detected'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {dtype}: {stats['detected']}/{stats['total']} ({rate:.0f}%)")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Main entry point."""
    
    print("="*60)
    print("OpenCV Facade Defect Detection")
    print("="*60)
    
    # Initialize configuration
    config = Config()
    
    # Learn baseline from positive images
    baseline = learn_baseline(config)
    
    if baseline.samples_analyzed == 0:
        print("\nWarning: No positive samples analyzed. Using default baseline.")
        baseline.screw_count_mean = 4
        baseline.screw_count_std = 2
        baseline.hole_count_mean = 4
        baseline.hole_count_std = 2
        baseline.tin_coverage_mean = 0.3
        baseline.tin_coverage_std = 0.1
    
    # Process negative images
    summary = process_negative_images(config, baseline)
    
    # Print summary
    print_summary(summary)
    
    print(f"\nDebug images saved to: {config.output_dir}/")
    print("\nDone!")


if __name__ == "__main__":
    main()
