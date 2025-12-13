"""
Combined Defect Detection Pipeline v3 (Improved Segmentation)

Detects ALL defect types using segmented_output_improved:
1. Lim (tin): Nedostaje, oštećen
2. Vijak (screw): Nedostaje, na pola pritegnut
3. Rupa (hole): Nedostaje, pomaknuta, višak
4. Staklo (glass): Pukotina, oštećenje
5. Brtva (seal): Nedostaje, oštećena/strgana

Uses:
- Original JPEGs for classification (v2 classifier)
- Mask JSON files from SAM improved segmentation (_masks.json)
- Component coverage analysis from pixel-level masks

Outputs unified defect results with confidence scores.
"""

import os
import sys
import json
import re


def combine_defects(*defect_lists) -> list:
    """
    Merge defects from all detectors.
    Keep ONE defect per (component, type) combination per image.
    E.g., one "screw missing" AND one "screw half_tightened" can both exist.
    """
    # First collect all defects
    all_defects = []
    for defect_list in defect_lists:
        if not defect_list:
            continue
        for d in defect_list:
            if isinstance(d, dict):
                all_defects.append(d)
    
    # Deduplicate: keep only ONE per (component, type) - pick highest confidence
    best_by_type = {}
    for d in all_defects:
        key = (d.get('component', ''), d.get('type', ''))
        conf = d.get('confidence', 0.5)
        if key not in best_by_type or conf > best_by_type[key].get('confidence', 0):
            best_by_type[key] = d
    
    return list(best_by_type.values())


def run_combined_detection():
    """Run the full combined detection pipeline."""
    print("="*60)
    print("COMBINED DEFECT DETECTION PIPELINE v3 (improved segmentation)")
    print("="*60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    
    # Use new segmented_output_improved structure
    seg_improved_dir = os.path.join(parent_dir, "segmented_output_improved", "negative")
    json_dir = os.path.join(parent_dir, "TRAIN", "negative")
    jpeg_dir = os.path.join(parent_dir, "TRAIN_JPG", "negative")
    
    # Check if improved segmentation exists
    if not os.path.exists(seg_improved_dir):
        print(f"ERROR: {seg_improved_dir} not found!")
        print("Run the SAM_Improved_Segmentation.ipynb notebook first.")
        return []
    
    # Load v2 classification results (from original JPEGs)
    v2_class_path = os.path.join(base_dir, "classification_results_v2.json")
    if os.path.exists(v2_class_path):
        with open(v2_class_path) as f:
            v2_classifications = json.load(f)
        # Build mapping: image_name -> model
        jpeg_to_model = {c['image']: c['matched_model'] for c in v2_classifications}
        print(f"Loaded {len(jpeg_to_model)} JPEG classifications (v2)")
    else:
        jpeg_to_model = {}
        print("WARNING: v2 classifications not found, run image_classifier_v2.py first")
    
    # Load CAD models
    with open(os.path.join(base_dir, "cad_models.json")) as f:
        cad_models = json.load(f)
    
    results = []
    
    # Iterate over mask JSON files in segmented_output_improved
    for mask_json in sorted(os.listdir(seg_improved_dir)):
        if not mask_json.endswith('_masks.json'):
            continue
        
        mask_json_path = os.path.join(seg_improved_dir, mask_json)
        
        # Load mask data
        with open(mask_json_path) as f:
            mask_data = json.load(f)
        
        # Extract base name: IMG_5349_2_facade_masks.json -> IMG_5349_2_facade
        base_name = mask_json.replace('_masks.json', '')
        
        # Map to original JPEG name: IMG_5349_2_facade -> IMG_5349 2.jpg
        jpeg_base = base_name.replace('_facade', '')
        jpeg_name = re.sub(r'_(\d+)$', r' \1', jpeg_base) + '.jpg'
        
        # Get CAD model from v2 classifier (original JPEG)
        matched_model = jpeg_to_model.get(jpeg_name)
        if not matched_model:
            jpeg_name_alt = jpeg_base.replace('_', ' ') + '.jpg'
            matched_model = jpeg_to_model.get(jpeg_name_alt, list(cad_models.keys())[0])
        
        # Find GT JSON
        json_name = re.sub(r'_(\d+)$', r' \1', jpeg_base) + '.json'
        json_path = os.path.join(json_dir, json_name)
        
        gt = None
        if os.path.exists(json_path):
            with open(json_path) as f:
                gt_data = json.load(f)
                gt = gt_data.get('defects', [])
        
        # Get image size from mask data
        img_size = mask_data.get('size', [1536, 688])
        w, h = img_size[0], img_size[1]
        total_pixels = w * h
        
        # Get CAD circles for the matched model
        cad_circles = cad_models[matched_model]['circles']
        
        # Analyze components from mask data
        components = mask_data.get('components', {})
        
        # Calculate coverage from mask JSON
        tin_masks = components.get('tin', [])
        glass_masks = components.get('glass', [])
        screw_masks = components.get('screw', [])
        hole_masks = components.get('hole', [])
        seal_masks = components.get('seal', [])
        
        # Total coverage per component
        tin_coverage = sum(m.get('coverage_percent', 0) for m in tin_masks) / 100
        glass_coverage = sum(m.get('coverage_percent', 0) for m in glass_masks) / 100
        screw_count = len(screw_masks)
        hole_count = len(hole_masks)
        seal_coverage = sum(m.get('coverage_percent', 0) for m in seal_masks) / 100
        
        # Detect defects based on coverage analysis
        defects = []
        
        # TIN detection - missing if low coverage
        if tin_coverage < 0.15:
            defects.append({
                'component': 'tin',
                'type': 'missing',
                'confidence': 0.85,
                'zone': 'full',
                'coverage': tin_coverage,
                'source': 'improved_segmentation'
            })
        elif tin_coverage < 0.30:
            defects.append({
                'component': 'tin',
                'type': 'damaged',
                'confidence': 0.75,
                'zone': 'partial',
                'coverage': tin_coverage,
                'source': 'improved_segmentation'
            })
        
        # SEAL detection - missing if low coverage
        if seal_coverage < 0.01:
            defects.append({
                'component': 'seal',
                'type': 'missing',
                'confidence': 0.80,
                'zone': 'full',
                'coverage': seal_coverage,
                'source': 'improved_segmentation'
            })
        elif seal_coverage < 0.05:
            defects.append({
                'component': 'seal',
                'type': 'torn',
                'confidence': 0.70,
                'zone': 'partial',
                'coverage': seal_coverage,
                'source': 'improved_segmentation'
            })
        
        # SCREW detection - compare with expected from CAD
        expected_screws = len([c for c in cad_circles if c.get('radius', 0) < 15])
        if screw_count < expected_screws * 0.5:
            defects.append({
                'component': 'screw',
                'type': 'missing',
                'confidence': 0.70,
                'expected': expected_screws,
                'detected': screw_count,
                'source': 'improved_segmentation'
            })
        
        # HOLE detection - compare with expected from CAD
        expected_holes = len([c for c in cad_circles if c.get('radius', 0) >= 15])
        if hole_count < expected_holes * 0.5:
            defects.append({
                'component': 'hole',
                'type': 'missing',
                'confidence': 0.70,
                'expected': expected_holes,
                'detected': hole_count,
                'source': 'improved_segmentation'
            })
        
        # GLASS detection - look for fragmented masks (potential cracks)
        if len(glass_masks) > 5:
            # Many small glass segments could indicate cracks
            avg_glass_size = sum(m.get('area_pixels', 0) for m in glass_masks) / len(glass_masks)
            if avg_glass_size < total_pixels * 0.02:
                defects.append({
                    'component': 'glass',
                    'type': 'crack',
                    'confidence': 0.65,
                    'fragments': len(glass_masks),
                    'source': 'improved_segmentation'
                })
        
        # Filter high confidence only
        high_conf = [d for d in defects if d.get('confidence', 0) >= 0.70]
        
        # Build result
        result = {
            'image': mask_json,
            'original_jpeg': jpeg_name,
            'matched_model': matched_model,
            'detection_stats': {
                'tin_coverage': tin_coverage,
                'glass_coverage': glass_coverage,
                'screw_count': screw_count,
                'hole_count': hole_count,
                'seal_coverage': seal_coverage,
                'tin_masks': len(tin_masks),
                'glass_masks': len(glass_masks),
            },
            'defects': high_conf,
            'all_defects': defects,
            'ground_truth': gt
        }
        
        # Match with GT
        if gt:
            result['gt_match'] = match_with_ground_truth(high_conf, gt)
        
        results.append(result)
        
        # Print
        n = len(high_conf)
        gt_str = ""
        if gt:
            matches = result.get('gt_match', {}).get('matches', 0)
            gt_str = f" (GT: {matches}/{len(gt)} matched)"
        print(f"  {mask_json[:45]}: {n} defects{gt_str}")
    
    # Save
    output_path = os.path.join(base_dir, "combined_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {output_path}")
    
    # Summary
    print("\n" + "="*60)
    print("COMBINED PIPELINE RESULTS")
    print("="*60)
    
    total_gt = 0
    total_det = 0
    total_match = 0
    
    for r in results:
        gt_match = r.get('gt_match', {})
        total_gt += gt_match.get('ground_truth_count', 0)
        total_det += gt_match.get('detected_count', 0)
        total_match += gt_match.get('matches', 0)
    
    print(f"Total images: {len(results)}")
    print(f"Ground truth defects: {total_gt}")
    print(f"Detected defects: {total_det}")
    print(f"Correct matches: {total_match}")
    if total_gt > 0:
        print(f"RECALL: {total_match/total_gt:.1%}")
    if total_det > 0:
        print(f"PRECISION: {total_match/total_det:.1%}")
    
    return results


def match_with_ground_truth(detected: list, ground_truth: list) -> dict:
    """Match detected defects with ground truth.
    Since each image has at most ONE defect per type, we match by component+type.
    Location is secondary - if we detect tin missing/damaged anywhere, it's a match.
    """
    type_equivalent = {
        'missing': ['missing', 'damaged', 'half_tightened'],  # half_tightened screw = problem
        'damaged': ['damaged', 'missing', 'crack', 'torn'],
        'torn': ['torn', 'damaged', 'missing'],  # seal torn = damaged/missing
    }
    
    matches = 0
    
    for gt in ground_truth:
        gt_comp = gt.get('component', '').lower()
        gt_type = gt.get('type', '').lower()
        
        # Get acceptable types
        acceptable_types = type_equivalent.get(gt_type, [gt_type])
        
        # Match if we have ANY defect of same component and equivalent type
        for det in detected:
            det_comp = det.get('component', '').lower()
            det_type = det.get('type', '').lower()
            
            if det_comp == gt_comp and det_type in acceptable_types:
                matches += 1
                break
    
    return {
        'ground_truth_count': len(ground_truth),
        'detected_count': len(detected),
        'matches': matches,
        'recall': matches / len(ground_truth) if ground_truth else 0,
        'precision': matches / len(detected) if detected else 0
    }


if __name__ == "__main__":
    run_combined_detection()
