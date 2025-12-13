"""
Facade Elements Segmentation using SAM 3 (Segment Anything Model 3)
Uses Hugging Face Transformers: https://huggingface.co/facebook/sam3
Segments components: seals, tin, screw, hole, and glass from facade images
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
import pillow_heif  # For HEIC support

# Register HEIF opener with PIL
pillow_heif.register_heif_opener()

# Configuration
TRAIN_DIR = "TRAIN"
POSITIVE_DIR = os.path.join(TRAIN_DIR, "positive")
NEGATIVE_DIR = os.path.join(TRAIN_DIR, "negative")
OUTPUT_DIR = "segmented_output"

# Component classes we're looking for in facade elements
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


def setup_sam3_model():
    """Load and setup the SAM 3 model from Hugging Face"""
    from transformers import Sam3Processor, Sam3Model
    
    # Check for GPU availability
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    
    print("Loading SAM 3 model from Hugging Face...")
    print("Note: First run will download the model (~3.6GB)")
    
    try:
        model = Sam3Model.from_pretrained("facebook/sam3").to(DEVICE)
        processor = Sam3Processor.from_pretrained("facebook/sam3")
        print("SAM 3 model loaded successfully!")
        return model, processor, DEVICE
    except Exception as e:
        print(f"Error loading SAM 3: {e}")
        print("\nMake sure you have:")
        print("1. Accepted the model license at: https://huggingface.co/facebook/sam3")
        print("2. Logged in with: huggingface-cli login")
        print("3. Installed transformers: pip install transformers")
        return None, None, None


def load_image(image_path):
    """Load image supporting HEIC, PNG, JPG formats - returns PIL Image"""
    ext = os.path.splitext(image_path)[1].lower()
    
    try:
        # PIL with pillow-heif handles all formats including HEIC
        pil_image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        return pil_image
    except Exception as e:
        print(f"  Error loading image: {e}")
        return None


def segment_component(image, component_name, model, processor, device, threshold=0.5):
    """Segment a specific component using SAM 3 text prompts"""
    
    # Prepare inputs with text prompt
    inputs = processor(images=image, text=component_name, return_tensors="pt").to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process results
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]
    
    return results


def segment_image(image_path, model, processor, device, output_path):
    """Segment all facade components in an image and save results"""
    print(f"Processing: {os.path.basename(image_path)}")
    
    # Load image
    pil_image = load_image(image_path)
    if pil_image is None:
        print(f"  Error: Could not read image {image_path}")
        return None
    
    # Resize if image is too large (for memory efficiency)
    max_dim = 2048
    w, h = pil_image.size
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
        print(f"  Resized to {new_w}x{new_h}")
    
    # Convert to numpy for OpenCV visualization
    image_rgb = np.array(pil_image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    annotated_image = image_bgr.copy()
    
    # Store all component results
    all_results = {}
    
    # Segment each component type using text prompts
    for component in COMPONENT_CLASSES:
        print(f"  Detecting: {component}...")
        try:
            results = segment_component(pil_image, component, model, processor, device)
            
            masks = results.get('masks', [])
            boxes = results.get('boxes', [])
            scores = results.get('scores', [])
            
            if len(masks) > 0:
                print(f"    Found {len(masks)} {component}(s)")
                all_results[component] = {
                    'masks': masks,
                    'boxes': boxes,
                    'scores': scores
                }
                
                # Draw masks on annotated image
                color = COMPONENT_COLORS[component]
                for mask in masks:
                    mask_np = mask.cpu().numpy().astype(bool)
                    overlay = annotated_image.copy()
                    overlay[mask_np] = color
                    annotated_image = cv2.addWeighted(annotated_image, 0.7, overlay, 0.3, 0)
                    
                    # Draw contour
                    contours, _ = cv2.findContours(
                        mask_np.astype(np.uint8), 
                        cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(annotated_image, contours, -1, color, 2)
            else:
                print(f"    No {component} found")
                
        except Exception as e:
            print(f"    Error detecting {component}: {e}")
    
    # Add legend
    y_offset = 30
    for comp, color in COMPONENT_COLORS.items():
        cv2.putText(annotated_image, comp, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset += 30
    
    # Save the annotated image
    cv2.imwrite(output_path, annotated_image)
    print(f"  Saved: {output_path}")
    
    return all_results, image_bgr


def save_component_masks(all_results, image_bgr, output_dir, base_name):
    """Save individual component masks as separate images"""
    masks_dir = os.path.join(output_dir, f"{base_name}_components")
    os.makedirs(masks_dir, exist_ok=True)
    
    for component, data in all_results.items():
        masks = data['masks']
        if len(masks) == 0:
            continue
            
        comp_dir = os.path.join(masks_dir, component)
        os.makedirs(comp_dir, exist_ok=True)
        
        for idx, mask in enumerate(masks):
            mask_np = mask.cpu().numpy().astype(np.uint8)
            
            # Create colored mask overlay
            colored_mask = np.zeros_like(image_bgr)
            colored_mask[mask_np.astype(bool)] = image_bgr[mask_np.astype(bool)]
            
            # Save individual mask
            mask_path = os.path.join(comp_dir, f"{component}_{idx:03d}.png")
            cv2.imwrite(mask_path, colored_mask)
            
            # Also save binary mask
            binary_mask = mask_np * 255
            binary_path = os.path.join(comp_dir, f"{component}_{idx:03d}_binary.png")
            cv2.imwrite(binary_path, binary_mask)
    
    print(f"  Saved component masks to {masks_dir}")


def create_comparison_image(original_bgr, segmented_path, output_path):
    """Create a side-by-side comparison image"""
    segmented = cv2.imread(segmented_path)
    
    if original_bgr is None or segmented is None:
        return
    
    # Resize original to match segmented
    h2, w2 = segmented.shape[:2]
    original_resized = cv2.resize(original_bgr, (w2, h2))
    
    # Create comparison
    comparison = np.hstack([original_resized, segmented])
    cv2.imwrite(output_path, comparison)
    print(f"  Created comparison: {output_path}")


def process_folder(folder_path, model, processor, device, category):
    """Process all images in a folder"""
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.heic', '.heif')
    
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(supported_extensions) and not f.startswith('.')]
    
    if not image_files:
        print(f"No images found in {folder_path}")
        return
    
    print(f"\nProcessing {len(image_files)} images from {category} folder")
    print("-" * 60)
    
    output_category_dir = os.path.join(OUTPUT_DIR, category)
    
    for image_file in sorted(image_files):
        image_path = os.path.join(folder_path, image_file)
        base_name = os.path.splitext(image_file)[0]
        # Clean up filename for output (remove spaces)
        base_name_clean = base_name.replace(" ", "_")
        
        # Output paths
        segmented_path = os.path.join(output_category_dir, f"{base_name_clean}_segmented.jpg")
        comparison_path = os.path.join(output_category_dir, f"{base_name_clean}_comparison.jpg")
        
        # Segment the image
        result = segment_image(image_path, model, processor, device, segmented_path)
        
        if result:
            all_results, image_bgr = result
            
            # Save individual component masks
            save_component_masks(all_results, image_bgr, output_category_dir, base_name_clean)
            
            # Create comparison image
            create_comparison_image(image_bgr, segmented_path, comparison_path)
        
        print()


def main():
    """Main function to process all facade element images"""
    print("=" * 60)
    print("Facade Elements Segmentation using SAM 3")
    print("https://huggingface.co/facebook/sam3")
    print("Components: seal, tin, screw, hole, glass")
    print("=" * 60)
    
    # Setup SAM 3 model
    model, processor, device = setup_sam3_model()
    if model is None:
        return
    
    # Process positive samples
    if os.path.exists(POSITIVE_DIR):
        process_folder(POSITIVE_DIR, model, processor, device, "positive")
    else:
        print(f"Positive folder not found: {POSITIVE_DIR}")
    
    # Process negative samples
    if os.path.exists(NEGATIVE_DIR):
        process_folder(NEGATIVE_DIR, model, processor, device, "negative")
    else:
        print(f"Negative folder not found: {NEGATIVE_DIR}")
    
    print("=" * 60)
    print("Processing complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
