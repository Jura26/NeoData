import os
import cv2
import torch
import numpy as np
import supervision as sv
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Configuration
ASSETS_DIR = "assets"
OUTPUT_DIR = "segmented_output"
CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def setup_sam_model():
    """Load and setup the SAM model"""
    # Check for GPU availability
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    
    # Check if checkpoint exists
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\nSAM checkpoint not found at {CHECKPOINT_PATH}")
        print("Please download it using:")
        print("wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        print("\nOr using PowerShell:")
        print("Invoke-WebRequest -Uri 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth' -OutFile 'sam_vit_h_4b8939.pth'")
        return None, None
    
    # Load SAM model
    print("Loading SAM model...")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    
    # Create mask generator
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )
    
    return sam, mask_generator


def segment_image(image_path, mask_generator, output_path):
    """Segment objects in an image and save the result"""
    print(f"Processing: {image_path}")
    
    # Read image
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"  Error: Could not read image {image_path}")
        return None
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Generate masks
    print("  Generating masks...")
    result = mask_generator.generate(image_rgb)
    print(f"  Found {len(result)} segments")
    
    # Create annotated image using supervision
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    detections = sv.Detections.from_sam(result)
    annotated_image = mask_annotator.annotate(
        scene=image_bgr.copy(), 
        detections=detections
    )
    
    # Save the annotated image
    cv2.imwrite(output_path, annotated_image)
    print(f"  Saved: {output_path}")
    
    return result


def save_individual_masks(image_path, result, output_dir, base_name):
    """Save individual masks as separate images"""
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return
    
    masks_dir = os.path.join(output_dir, f"{base_name}_masks")
    os.makedirs(masks_dir, exist_ok=True)
    
    for i, mask_data in enumerate(result):
        mask = mask_data['segmentation']
        
        # Create colored mask overlay
        colored_mask = np.zeros_like(image_bgr)
        colored_mask[mask] = image_bgr[mask]
        
        # Save individual mask
        mask_path = os.path.join(masks_dir, f"mask_{i:03d}.png")
        cv2.imwrite(mask_path, colored_mask)
        
        # Also save binary mask
        binary_mask = (mask * 255).astype(np.uint8)
        binary_path = os.path.join(masks_dir, f"mask_{i:03d}_binary.png")
        cv2.imwrite(binary_path, binary_mask)
    
    print(f"  Saved {len(result)} individual masks to {masks_dir}")


def create_comparison_image(original_path, segmented_path, output_path):
    """Create a side-by-side comparison image"""
    original = cv2.imread(original_path)
    segmented = cv2.imread(segmented_path)
    
    if original is None or segmented is None:
        return
    
    # Resize images to same height
    h1, w1 = original.shape[:2]
    h2, w2 = segmented.shape[:2]
    
    # Use original dimensions
    comparison = np.hstack([original, segmented])
    cv2.imwrite(output_path, comparison)
    print(f"  Created comparison: {output_path}")


def main():
    """Main function to process all images"""
    print("=" * 60)
    print("Apple Segmentation using SAM (Segment Anything Model)")
    print("=" * 60)
    
    # Setup SAM model
    sam, mask_generator = setup_sam_model()
    if mask_generator is None:
        return
    
    # Get list of images
    image_files = [f for f in os.listdir(ASSETS_DIR) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print(f"No images found in {ASSETS_DIR}")
        return
    
    print(f"\nFound {len(image_files)} images to process")
    print("-" * 60)
    
    # Process each image
    for image_file in sorted(image_files):
        image_path = os.path.join(ASSETS_DIR, image_file)
        base_name = os.path.splitext(image_file)[0]
        
        # Output paths
        segmented_path = os.path.join(OUTPUT_DIR, f"{base_name}_segmented.jpg")
        comparison_path = os.path.join(OUTPUT_DIR, f"{base_name}_comparison.jpg")
        
        # Segment the image
        result = segment_image(image_path, mask_generator, segmented_path)
        
        if result:
            # Save individual masks
            save_individual_masks(image_path, result, OUTPUT_DIR, base_name)
            
            # Create comparison image
            create_comparison_image(image_path, segmented_path, comparison_path)
        
        print()
    
    print("=" * 60)
    print("Processing complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
