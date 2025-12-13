"""
YOLO + SAM Pipeline for Facade Element Segmentation
1. YOLO detects components (seal, tin, screw, hole, glass) with bounding boxes
2. SAM 3 segments each detection using bounding box prompts
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
import pillow_heif

# Register HEIF opener
pillow_heif.register_heif_opener()

# Configuration
TRAIN_DIR = "TRAIN"
POSITIVE_DIR = os.path.join(TRAIN_DIR, "positive")
NEGATIVE_DIR = os.path.join(TRAIN_DIR, "negative")
OUTPUT_DIR = "segmented_output"

# Component classes (must match your YOLO training classes)
COMPONENT_CLASSES = ["seal", "tin", "screw", "hole", "glass"]

# Color scheme for components (BGR for OpenCV)
COMPONENT_COLORS = {
    "seal": (255, 0, 0),      # Blue
    "tin": (0, 255, 255),     # Yellow
    "screw": (0, 255, 0),     # Green
    "hole": (0, 0, 255),      # Red
    "glass": (255, 255, 0),   # Cyan
}

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "positive"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "negative"), exist_ok=True)


class YOLODetector:
    """YOLO object detector for facade components"""
    
    def __init__(self, model_path=None):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to trained YOLO model (e.g., 'runs/detect/train/weights/best.pt')
                       If None, uses a placeholder for demo
        """
        from ultralytics import YOLO
        
        if model_path and os.path.exists(model_path):
            print(f"Loading trained YOLO model: {model_path}")
            self.model = YOLO(model_path)
        else:
            print("No trained model found. Using YOLOv8n as placeholder.")
            print("Train your model first with: yolo train data=dataset.yaml model=yolov8n.pt epochs=100")
            self.model = YOLO('yolov8n.pt')  # Placeholder - replace with your trained model
        
        self.class_names = COMPONENT_CLASSES
    
    def detect(self, image, conf_threshold=0.25):
        """
        Detect components in image
        
        Returns:
            List of detections: [{'box': [x1,y1,x2,y2], 'class': 'seal', 'conf': 0.95}, ...]
        """
        results = self.model(image, conf=conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                    conf = float(boxes.conf[i])
                    cls_id = int(boxes.cls[i])
                    
                    # Map class ID to name
                    if cls_id < len(self.class_names):
                        cls_name = self.class_names[cls_id]
                    else:
                        cls_name = f"class_{cls_id}"
                    
                    detections.append({
                        'box': box.tolist(),
                        'class': cls_name,
                        'conf': conf
                    })
        
        return detections


class SAM3Segmenter:
    """SAM 3 segmenter using bounding box prompts"""
    
    def __init__(self):
        """Initialize SAM 3 from Hugging Face"""
        from transformers import Sam3Processor, Sam3Model
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"SAM 3 using device: {self.device}")
        
        print("Loading SAM 3 model...")
        self.model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")
        print("SAM 3 loaded successfully!")
    
    def segment_with_box(self, image, box):
        """
        Segment object using bounding box prompt
        
        Args:
            image: PIL Image
            box: [x1, y1, x2, y2] bounding box
            
        Returns:
            Binary mask (numpy array)
        """
        # Prepare inputs with box prompt
        # Box format for SAM 3: [[box]] for single box
        input_boxes = [[[box]]]  # [batch, num_boxes, 4]
        input_boxes_labels = [[[1]]]  # 1 = positive box
        
        inputs = self.processor(
            images=image,
            input_boxes=input_boxes,
            input_boxes_labels=input_boxes_labels,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]
        
        masks = results.get('masks', [])
        if len(masks) > 0:
            # Return the first (best) mask
            return masks[0].cpu().numpy().astype(np.uint8)
        return None


def load_image(image_path):
    """Load image supporting HEIC, PNG, JPG formats"""
    try:
        pil_image = Image.open(image_path)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        return pil_image
    except Exception as e:
        print(f"  Error loading image: {e}")
        return None


def process_image(image_path, yolo_detector, sam_segmenter, output_path):
    """Process single image through YOLO + SAM pipeline"""
    print(f"\nProcessing: {os.path.basename(image_path)}")
    
    # Load image
    pil_image = load_image(image_path)
    if pil_image is None:
        return None
    
    # Resize if too large
    max_dim = 2048
    w, h = pil_image.size
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
        print(f"  Resized to {new_w}x{new_h}")
    
    # Convert for visualization
    image_rgb = np.array(pil_image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    annotated_image = image_bgr.copy()
    
    # Step 1: YOLO detection
    print("  Running YOLO detection...")
    detections = yolo_detector.detect(pil_image)
    print(f"  Found {len(detections)} detections")
    
    # Step 2: SAM segmentation for each detection
    all_results = {}
    
    for det in detections:
        box = det['box']
        cls_name = det['class']
        conf = det['conf']
        
        print(f"  Segmenting {cls_name} (conf: {conf:.2f})...")
        
        # Get SAM mask using box prompt
        mask = sam_segmenter.segment_with_box(pil_image, box)
        
        if mask is not None:
            # Store result
            if cls_name not in all_results:
                all_results[cls_name] = []
            all_results[cls_name].append({
                'mask': mask,
                'box': box,
                'conf': conf
            })
            
            # Draw on annotated image
            color = COMPONENT_COLORS.get(cls_name, (128, 128, 128))
            
            # Draw mask overlay
            mask_bool = mask.astype(bool)
            overlay = annotated_image.copy()
            overlay[mask_bool] = color
            annotated_image = cv2.addWeighted(annotated_image, 0.7, overlay, 0.3, 0)
            
            # Draw contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated_image, contours, -1, color, 2)
            
            # Draw bounding box and label
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name}: {conf:.2f}"
            cv2.putText(annotated_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add legend
    y_offset = 30
    for comp, color in COMPONENT_COLORS.items():
        cv2.putText(annotated_image, comp, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset += 30
    
    # Save result
    cv2.imwrite(output_path, annotated_image)
    print(f"  Saved: {output_path}")
    
    return all_results, image_bgr


def save_component_masks(all_results, image_bgr, output_dir, base_name):
    """Save individual component masks"""
    masks_dir = os.path.join(output_dir, f"{base_name}_components")
    os.makedirs(masks_dir, exist_ok=True)
    
    for component, detections in all_results.items():
        comp_dir = os.path.join(masks_dir, component)
        os.makedirs(comp_dir, exist_ok=True)
        
        for idx, det in enumerate(detections):
            mask = det['mask']
            
            # Save colored mask
            colored_mask = np.zeros_like(image_bgr)
            colored_mask[mask.astype(bool)] = image_bgr[mask.astype(bool)]
            cv2.imwrite(os.path.join(comp_dir, f"{component}_{idx:03d}.png"), colored_mask)
            
            # Save binary mask
            binary_mask = mask * 255
            cv2.imwrite(os.path.join(comp_dir, f"{component}_{idx:03d}_binary.png"), binary_mask)
    
    print(f"  Saved masks to {masks_dir}")


def process_folder(folder_path, yolo_detector, sam_segmenter, category):
    """Process all images in folder"""
    supported_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.heic', '.heif')
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(supported_ext) and not f.startswith('.')]
    
    if not image_files:
        print(f"No images in {folder_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing {len(image_files)} images from {category}")
    print(f"{'='*60}")
    
    output_dir = os.path.join(OUTPUT_DIR, category)
    
    for image_file in sorted(image_files):
        image_path = os.path.join(folder_path, image_file)
        base_name = os.path.splitext(image_file)[0].replace(" ", "_")
        
        output_path = os.path.join(output_dir, f"{base_name}_segmented.jpg")
        
        result = process_image(image_path, yolo_detector, sam_segmenter, output_path)
        
        if result:
            all_results, image_bgr = result
            save_component_masks(all_results, image_bgr, output_dir, base_name)


def main():
    """Main pipeline"""
    print("=" * 60)
    print("YOLO + SAM Pipeline for Facade Element Segmentation")
    print("=" * 60)
    
    # Initialize YOLO detector
    # Replace with your trained model path:
    # yolo_model_path = "runs/detect/train/weights/best.pt"
    yolo_model_path = None  # Will use placeholder
    
    print("\n[1/2] Loading YOLO detector...")
    yolo_detector = YOLODetector(yolo_model_path)
    
    # Initialize SAM 3
    print("\n[2/2] Loading SAM 3 segmenter...")
    sam_segmenter = SAM3Segmenter()
    
    # Process folders
    if os.path.exists(POSITIVE_DIR):
        process_folder(POSITIVE_DIR, yolo_detector, sam_segmenter, "positive")
    
    if os.path.exists(NEGATIVE_DIR):
        process_folder(NEGATIVE_DIR, yolo_detector, sam_segmenter, "negative")
    
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
