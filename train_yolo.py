"""
Train YOLO model for Facade Element Detection
Run this after labeling data in Label Studio and exporting to YOLO format
"""

from ultralytics import YOLO

def train_yolo():
    """Train YOLOv8 on facade components dataset"""
    
    # Load a pretrained model (recommended for transfer learning)
    model = YOLO('yolov8n.pt')  # Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    
    # Train the model
    results = model.train(
        data='dataset.yaml',      # Path to dataset config
        epochs=100,               # Number of training epochs
        imgsz=640,                # Image size
        batch=16,                 # Batch size (reduce if GPU memory issues)
        patience=20,              # Early stopping patience
        save=True,                # Save checkpoints
        device=0,                 # GPU device (0, 1, 2...) or 'cpu'
        workers=8,                # Number of data loading workers
        project='runs/detect',    # Save results to project/name
        name='facade_elements',   # Experiment name
        exist_ok=True,            # Overwrite existing experiment
        pretrained=True,          # Use pretrained weights
        optimizer='auto',         # Optimizer (SGD, Adam, AdamW, auto)
        verbose=True,             # Print verbose output
        seed=42,                  # Random seed for reproducibility
        
        # Data augmentation (adjust based on your data)
        hsv_h=0.015,              # HSV-Hue augmentation
        hsv_s=0.7,                # HSV-Saturation augmentation  
        hsv_v=0.4,                # HSV-Value augmentation
        degrees=0.0,              # Rotation augmentation (degrees)
        translate=0.1,            # Translation augmentation
        scale=0.5,                # Scale augmentation
        flipud=0.0,               # Flip up-down probability
        fliplr=0.5,               # Flip left-right probability
        mosaic=1.0,               # Mosaic augmentation probability
        mixup=0.0,                # Mixup augmentation probability
    )
    
    print("\n" + "="*60)
    print("Training complete!")
    print("Best model saved to: runs/detect/facade_elements/weights/best.pt")
    print("="*60)
    
    return results


def validate_model(model_path='runs/detect/facade_elements/weights/best.pt'):
    """Validate trained model"""
    model = YOLO(model_path)
    metrics = model.val()
    print(f"\nmAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")
    return metrics


def test_inference(model_path='runs/detect/facade_elements/weights/best.pt', image_path=None):
    """Test inference on a single image"""
    model = YOLO(model_path)
    
    if image_path:
        results = model(image_path, save=True)
        print(f"Results saved to: {results[0].save_dir}")
    else:
        print("Provide an image_path to test inference")


if __name__ == "__main__":
    print("="*60)
    print("YOLO Training for Facade Element Detection")
    print("Classes: seal, tin, screw, hole, glass")
    print("="*60)
    
    print("\nMake sure you have:")
    print("1. Labeled your data in Label Studio")
    print("2. Exported to YOLO format")
    print("3. Updated dataset.yaml with correct paths")
    
    response = input("\nStart training? (y/n): ")
    if response.lower() == 'y':
        train_yolo()
