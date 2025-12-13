"""
Convert HEIC images to JPG for Label Studio
"""

import os
from PIL import Image
import pillow_heif

# Register HEIF opener
pillow_heif.register_heif_opener()

def convert_heic_to_jpg(input_dir, output_dir):
    """Convert all HEIC files in directory to JPG"""
    os.makedirs(output_dir, exist_ok=True)
    
    converted = 0
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.heic', '.heif')):
            input_path = os.path.join(input_dir, filename)
            output_filename = os.path.splitext(filename)[0] + '.jpg'
            output_path = os.path.join(output_dir, output_filename)
            
            try:
                img = Image.open(input_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(output_path, 'JPEG', quality=95)
                print(f"Converted: {filename} -> {output_filename}")
                converted += 1
            except Exception as e:
                print(f"Error converting {filename}: {e}")
        
        # Copy non-HEIC files
        elif filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            img = Image.open(input_path)
            img.save(output_path)
            print(f"Copied: {filename}")
            converted += 1
    
    print(f"\nTotal converted/copied: {converted} files")
    return converted


if __name__ == "__main__":
    # Convert positive samples
    print("Converting positive samples...")
    convert_heic_to_jpg("TRAIN/positive", "TRAIN_JPG/positive")
    
    print("\nConverting negative samples...")
    convert_heic_to_jpg("TRAIN/negative", "TRAIN_JPG/negative")
    
    print("\n" + "="*50)
    print("Done! Import images from TRAIN_JPG folder into Label Studio")
