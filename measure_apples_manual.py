"""
Manual Box Edge Selection Tool
Allows you to manually select the reference edge and then measure apples
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path

# Configuration
ASSETS_DIR = "assets"
OUTPUT_DIR = "measurements_output"
BOX_EDGE_LENGTH_METERS = 1.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global variables for mouse callback
points = []
current_image = None


def mouse_callback(event, x, y, flags, param):
    """Mouse callback to capture two points for the reference edge"""
    global points, current_image
    
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"  Point {len(points)}: ({x}, {y})")
        
        # Draw point on image
        cv2.circle(current_image, (x, y), 5, (0, 0, 255), -1)
        
        # If we have 2 points, draw the line
        if len(points) == 2:
            cv2.line(current_image, points[0], points[1], (255, 0, 0), 2)
            length = np.sqrt((points[1][0] - points[0][0])**2 + 
                           (points[1][1] - points[0][1])**2)
            cv2.putText(current_image, f"Edge: {length:.1f}px = 1m", 
                       (points[0][0], points[0][1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        cv2.imshow('Select Reference Edge', current_image)


def select_reference_edge(image):
    """
    Interactive tool to select reference edge
    User clicks two points to define the 1-meter edge
    """
    global points, current_image
    points = []
    current_image = image.copy()
    
    print("\n  Click two points to define the 1-meter reference edge")
    print("  Press ENTER when done, or 'r' to reset, or 'q' to skip")
    
    cv2.namedWindow('Select Reference Edge')
    cv2.setMouseCallback('Select Reference Edge', mouse_callback)
    cv2.imshow('Select Reference Edge', current_image)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # Enter key
            if len(points) == 2:
                cv2.destroyAllWindows()
                return points
            else:
                print("  Please select 2 points first!")
        
        elif key == ord('r'):  # Reset
            print("  Resetting points...")
            points = []
            current_image = image.copy()
            cv2.imshow('Select Reference Edge', current_image)
        
        elif key == ord('q'):  # Skip
            cv2.destroyAllWindows()
            return None
    
    cv2.destroyAllWindows()
    return None


def detect_apples(image):
    """
    Detect apples using color segmentation
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Red apples
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Green/yellow apples
    lower_green = np.array([20, 40, 40])
    upper_green = np.array([80, 255, 255])
    
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    mask = cv2.bitwise_or(mask_red1, mask_red2)
    mask = cv2.bitwise_or(mask, mask_green)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_area = 500
    apple_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    return apple_contours, mask


def calculate_distance_from_reference(point, ref_p1, ref_p2):
    """
    Calculate perpendicular distance from point to reference line
    """
    x0, y0 = point
    x1, y1 = ref_p1
    x2, y2 = ref_p2
    
    # Line equation coefficients
    a = y2 - y1
    b = x1 - x2
    c = x2*y1 - x1*y2
    
    # Perpendicular distance
    distance = abs(a*x0 + b*y0 + c) / np.sqrt(a**2 + b**2)
    
    return distance


def process_image_with_manual_reference(image_path, output_dir):
    """
    Process image with manually selected reference edge
    """
    print(f"\n{'='*70}")
    print(f"Processing: {image_path}")
    print('='*70)
    
    image = cv2.imread(image_path)
    if image is None:
        print("  Error: Could not read image")
        return None
    
    h, w = image.shape[:2]
    
    # Step 1: Select reference edge manually
    reference_points = select_reference_edge(image)
    
    if reference_points is None:
        print("  Skipped by user")
        return None
    
    p1, p2 = reference_points
    reference_length_pixels = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    scale_factor = BOX_EDGE_LENGTH_METERS / reference_length_pixels
    
    print(f"\n  Reference edge: {reference_length_pixels:.1f} pixels = 1 meter")
    print(f"  Scale: {1/scale_factor:.2f} pixels/meter")
    print(f"  Scale factor: {scale_factor:.6f} meters/pixel")
    
    # Step 2: Detect apples
    print("\n  Detecting apples...")
    apple_contours, apple_mask = detect_apples(image)
    print(f"  Found {len(apple_contours)} apples")
    
    # Step 3: Measure apples
    vis_image = image.copy()
    
    # Draw reference edge
    cv2.line(vis_image, p1, p2, (255, 0, 0), 3)
    cv2.putText(vis_image, f"Reference: 1m ({reference_length_pixels:.1f}px)", 
                (p1[0], p1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    measurements = []
    
    for i, contour in enumerate(apple_contours):
        # Get minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        
        # Calculate measurements
        diameter_pixels = radius * 2
        diameter_meters = diameter_pixels * scale_factor
        area_pixels = cv2.contourArea(contour)
        area_meters_sq = area_pixels * (scale_factor ** 2)
        
        # Distance from reference edge
        distance_pixels = calculate_distance_from_reference(center, p1, p2)
        distance_meters = distance_pixels * scale_factor
        
        apple_data = {
            'apple_id': i + 1,
            'center_x': center[0],
            'center_y': center[1],
            'diameter_cm': round(diameter_meters * 100, 2),
            'area_cm2': round(area_meters_sq * 10000, 2),
            'distance_from_edge_cm': round(distance_meters * 100, 2)
        }
        measurements.append(apple_data)
        
        # Visualize
        cv2.circle(vis_image, center, radius, (0, 255, 0), 2)
        cv2.circle(vis_image, center, 3, (0, 0, 255), -1)
        
        # Add text
        text = f"#{i+1}: {diameter_meters*100:.1f}cm"
        cv2.putText(vis_image, text, (center[0]-50, center[1]-radius-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw distance line to reference
        # Find closest point on reference line
        v = np.array([p2[0]-p1[0], p2[1]-p1[1]])
        u = np.array([center[0]-p1[0], center[1]-p1[1]])
        line_len_sq = np.dot(v, v)
        
        if line_len_sq > 0:
            t = max(0, min(1, np.dot(u, v) / line_len_sq))
            closest = (int(p1[0] + t*v[0]), int(p1[1] + t*v[1]))
            cv2.line(vis_image, center, closest, (255, 255, 0), 1)
            
            # Add distance text on the line
            mid = (int((center[0]+closest[0])/2), int((center[1]+closest[1])/2))
            cv2.putText(vis_image, f"{distance_meters*100:.1f}cm", mid,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    # Save results
    base_name = Path(image_path).stem
    
    vis_path = os.path.join(output_dir, f"{base_name}_measured.jpg")
    cv2.imwrite(vis_path, vis_image)
    print(f"\n  Saved: {vis_path}")
    
    mask_path = os.path.join(output_dir, f"{base_name}_apple_mask.jpg")
    cv2.imwrite(mask_path, apple_mask)
    
    # Save JSON
    json_data = {
        'image': image_path,
        'image_size': {'width': w, 'height': h},
        'reference_edge': {
            'point1': list(p1),
            'point2': list(p2),
            'length_pixels': float(reference_length_pixels),
            'length_meters': BOX_EDGE_LENGTH_METERS
        },
        'scale_factor': float(scale_factor),
        'pixels_per_meter': float(1/scale_factor),
        'apples': measurements
    }
    
    json_path = os.path.join(output_dir, f"{base_name}_measurements.json")
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # Print summary
    print(f"\n  {'='*60}")
    print(f"  MEASUREMENTS SUMMARY")
    print(f"  {'='*60}")
    for m in measurements:
        print(f"  Apple #{m['apple_id']}:")
        print(f"    - Diameter: {m['diameter_cm']:.1f} cm")
        print(f"    - Area: {m['area_cm2']:.1f} cm²")
        print(f"    - Distance from edge: {m['distance_from_edge_cm']:.1f} cm")
    
    return json_data


def main():
    """Main function"""
    print("="*70)
    print("Apple Measurement Tool - Manual Reference Selection")
    print("="*70)
    print("\nInstructions:")
    print("  1. Click two points to define the 1-meter reference edge")
    print("  2. Press ENTER to confirm")
    print("  3. Press 'r' to reset selection")
    print("  4. Press 'q' to skip image")
    print("="*70)
    
    if not os.path.exists(ASSETS_DIR):
        print(f"Error: {ASSETS_DIR} directory not found!")
        return
    
    image_files = [f for f in os.listdir(ASSETS_DIR) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print(f"No images found in {ASSETS_DIR}")
        return
    
    print(f"\nFound {len(image_files)} images")
    
    all_results = []
    
    for image_file in sorted(image_files):
        image_path = os.path.join(ASSETS_DIR, image_file)
        result = process_image_with_manual_reference(image_path, OUTPUT_DIR)
        if result:
            all_results.append(result)
    
    if all_results:
        combined_path = os.path.join(OUTPUT_DIR, "all_measurements.json")
        with open(combined_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n{'='*70}")
        print(f"All measurements saved to: {combined_path}")
        print(f"Processed {len(all_results)} images successfully")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
