""" 
Retail Shelf Compliance - Planogram Comparison (Enhanced Reporting)
""" 
import gradio as gr
import torch
import numpy as np
from PIL import Image
import warnings
from typing import Tuple, Dict, List
import os
import subprocess

warnings.filterwarnings('ignore')

# ============================================================================
# MODEL LOADING (Unchanged)
# ============================================================================
print("🚀 Loading models...")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📍 Device: {DEVICE}")

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from segment_anything import sam_model_registry, SamPredictor

# Grounding DINO
processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
model = AutoModelForZeroShotObjectDetection.from_pretrained(
    "IDEA-Research/grounding-dino-tiny"
).to(DEVICE)
print("✅ Grounding DINO loaded")

# SAM (Kept for compatibility, though used implicitly in full pipeline)
sam_checkpoint = "sam_vit_h_4b8939.pth"
if not os.path.exists(sam_checkpoint):
    print("⬇️ Downloading SAM weights...")
    subprocess.run([
        "wget", "-q",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    ])

sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)
print("✅ SAM loaded")

import supervision as sv
print("✅ All models ready!")

# ============================================================================
# CORE DETECTION (Unchanged Logic)
# ============================================================================
def detect_objects_robust(image: Image.Image, prompt: str, box_threshold: float, text_threshold: float) -> Dict:
    """Robust detection with fallback"""
    
    # Ensure prompt is clean for the model
    clean_prompt = prompt if prompt.endswith(".") else prompt + "."
    
    inputs = processor(images=image, text=clean_prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Try modern API first
    try:
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]]
        )[0]
        
        boxes = results["boxes"]
        scores = results["scores"]
        
        # Handle labels
        if "text_labels" in results:
            labels = results["text_labels"]
        elif "labels" in results:
            labels = [str(l) for l in results["labels"]]
        else:
            labels = ["object"] * len(boxes)
            
    except (TypeError, KeyError) as e:
        # Legacy fallback
        print(f"Using legacy mode...")
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, target_sizes=[image.size[::-1]]
        )[0]
        
        mask = results["scores"] > box_threshold
        boxes = results["boxes"][mask]
        scores = results["scores"][mask]
        
        # Try to get labels
        if "text_labels" in results:
            all_labels = results["text_labels"]
            labels = [all_labels[i] for i, m in enumerate(mask) if m]
        else:
            labels = ["object"] * len(boxes)
    
    return {
        "boxes": boxes.cpu().numpy() if torch.is_tensor(boxes) else boxes,
        "scores": scores.cpu().numpy() if torch.is_tensor(scores) else scores,
        "labels": labels,
        "count": len(boxes) if hasattr(boxes, '__len__') else 0
    }

# ============================================================================
# HELPER: LOCATION FINDER (New)
# ============================================================================
def get_location_description(box, image_w, image_h):
    """Returns human readable location (e.g., 'Top Left')"""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    
    # Vertical
    if cy < image_h / 3: vert = "Top"
    elif cy < 2 * image_h / 3: vert = "Middle"
    else: vert = "Bottom"
    
    # Horizontal
    if cx < image_w / 3: horiz = "Left"
    elif cx < 2 * image_w / 3: horiz = "Center"
    else: horiz = "Right"
    
    return f"{vert}-{horiz}"

# ============================================================================
# ENHANCED COMPARISON
# ============================================================================
def calculate_iou(box1, box2):
    """Calculate IoU between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def detailed_compare(ref_det: Dict, act_det: Dict, w: int, h: int) -> Dict:
    """Detailed comparison returning indices and locations"""
    
    ref_count = ref_det["count"]
    act_count = act_det["count"]
    
    matched_indices = set()
    used_actual_indices = set()
    
    # Matching Logic
    if ref_count > 0 and act_count > 0:
        for i, ref_box in enumerate(ref_det["boxes"]):
            best_iou = 0
            best_idx = -1
            
            for j, act_box in enumerate(act_det["boxes"]):
                if j in used_actual_indices:
                    continue
                
                iou = calculate_iou(ref_box, act_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
            
            if best_iou > 0.30:  # Threshold
                matched_indices.add(i)
                used_actual_indices.add(best_idx)

    # Compile Missing Items (In Ref, not in Act)
    missing_details = []
    for i in range(ref_count):
        if i not in matched_indices:
            lbl = ref_det["labels"][i]
            loc = get_location_description(ref_det["boxes"][i], w, h)
            missing_details.append(f"{lbl} at {loc}")

    # Compile Extra Items (In Act, not in Ref)
    extra_details = []
    for j in range(act_count):
        if j not in used_actual_indices:
            lbl = act_det["labels"][j]
            loc = get_location_description(act_det["boxes"][j], w, h)
            extra_details.append(f"{lbl} at {loc}")

    score = (len(matched_indices) / ref_count) * 100 if ref_count > 0 else 0
    
    return {
        "score": score,
        "matched": len(matched_indices),
        "missing_count": len(missing_details),
        "extra_count": len(extra_details),
        "missing_details": missing_details,
        "extra_details": extra_details
    }

# ============================================================================
# VISUALIZATION (Updated Labels)
# ============================================================================
def visualize_detections(image_np, detections):
    """Visualization with labels"""
    if detections["count"] == 0:
        return image_np
    
    # Create simpler labels for visualization
    labels = [
        f"{detections['labels'][i]}" 
        for i in range(detections["count"])
    ]

    sv_det = sv.Detections(
        xyxy=detections["boxes"],
        class_id=np.arange(detections["count"]),
        confidence=detections["scores"]
    )
    
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5)
    
    annotated = box_annotator.annotate(scene=image_np.copy(), detections=sv_det)
    annotated = label_annotator.annotate(scene=annotated, detections=sv_det, labels=labels)
    
    return annotated

# ============================================================================
# MAIN LOGIC & REPORT GENERATION
# ============================================================================
def analyze_prompt_coverage(prompt, all_detected_labels):
    """Checks which prompt words were found and which were not"""
    # Clean up prompt terms (split by . or ,)
    prompt_terms = [t.strip().lower() for t in prompt.replace(".", ",").split(",") if t.strip()]
    detected_lower = set([l.lower() for l in all_detected_labels])
    
    found = []
    not_found = []
    
    for term in prompt_terms:
        # Check if term is part of any detected label
        if any(term in d for d in detected_lower):
            found.append(term)
        else:
            not_found.append(term)
            
    return found, not_found

def planogram_comparison(
    reference_image,
    actual_image,
    prompt: str,
    box_threshold: float,
    text_threshold: float
):
    if reference_image is None or actual_image is None:
        return None, None, "❌ Please upload both images"
    
    if not prompt.strip():
        prompt = "bottle"
    
    try:
        # Convert to PIL
        ref_pil = Image.fromarray(reference_image).convert("RGB")
        act_pil = Image.fromarray(actual_image).convert("RGB")
        w, h = ref_pil.size
        
        # Detect
        print(f"🔍 Detecting with prompt: '{prompt}'")
        ref_det = detect_objects_robust(ref_pil, prompt, box_threshold, text_threshold)
        act_det = detect_objects_robust(act_pil, prompt, box_threshold, text_threshold)
        
        # Compare
        comparison = detailed_compare(ref_det, act_det, w, h)
        
        # Visualize
        ref_vis = visualize_detections(reference_image, ref_det)
        act_vis = visualize_detections(actual_image, act_det)
        
        # --- GENERATE DETAILED REPORT ---
        score = comparison["score"]
        
        # Status Header
        if score >= 90: status = "✅ EXCELLENT"
        elif score >= 75: status = "👍 GOOD"
        elif score >= 50: status = "⚠️ WARNING"
        else: status = "❌ CRITICAL"
        
        report = f"# {status}\n"
        report += f"**Compliance Score:** {score:.1f}%\n\n"
        report += "---\n"
        
        # 1. Prompt Analysis
        all_labels = ref_det['labels'] + act_det['labels']
        found_terms, not_found_terms = analyze_prompt_coverage(prompt, all_labels)
        
        report += "### 📝 Prompt Analysis\n"
        if found_terms:
            report += f"**✅ Detected Categories:** {', '.join(found_terms)}\n"
        if not_found_terms:
            report += f"**❌ Not Found in Images:** {', '.join(not_found_terms)}\n"
            report += "> *Tip: If these items are present but not detected, try lowering the 'Text Threshold'.*\n"
        report += "\n"

        # 2. General Stats
        report += "### 📊 Statistics\n"
        report += f"| Metric | Count |\n|---|---|\n"
        report += f"| Reference Items | {ref_det['count']} |\n"
        report += f"| Actual Items | {act_det['count']} |\n"
        report += f"| Matched Items | {comparison['matched']} |\n\n"
        
        # 3. Discrepancy Details
        if comparison['missing_count'] > 0:
            report += f"### 🚨 Missing Items ({comparison['missing_count']})\n"
            report += "*Items present in Planogram but missing from Shelf:*\n"
            for item in comparison['missing_details']:
                report += f"- 🔴 {item}\n"
            report += "\n"
        else:
            report += "### ✅ No Missing Items\n"
            
        if comparison['extra_count'] > 0:
            report += f"### ℹ️ Extra/Misplaced Items ({comparison['extra_count']})\n"
            report += "*Items detected on Shelf but not in Planogram:*\n"
            for item in comparison['extra_details']:
                report += f"- 🔵 {item}\n"
        
        # Warnings
        if ref_det['count'] == 0:
            report += "\n⚠️ **System Warning:** No items detected in Reference Image. Check your prompt."
        
        return ref_vis, act_vis, report
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}\n\n```\n{traceback.format_exc()}\n```"
        return reference_image, actual_image, error_msg

# ============================================================================
# GRADIO INTERFACE
# ============================================================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🛒 Retail Shelf Planogram Compliance
    Upload a reference planogram and an actual shelf image. The system will identify missing items and check layout compliance.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Configuration")
            
            reference_input = gr.Image(label="Reference (Planogram)", type="numpy")
            actual_input = gr.Image(label="Actual Shelf", type="numpy")
            
            prompt_input = gr.Textbox(
                label="Detection Prompt (Classes)",
                value="red bottle . blue bottle . green pack",
                placeholder="Separate items with dots, e.g.: coke . pepsi . fanta"
            )
            
            with gr.Accordion("⚙️ Advanced Thresholds", open=False):
                box_threshold = gr.Slider(0.1, 0.9, 0.30, 0.05, label="Box Threshold (Confidence)")
                text_threshold = gr.Slider(0.1, 0.9, 0.25, 0.05, label="Text Threshold (Labeling)")
            
            compare_btn = gr.Button("🔍 Analyze Compliance", variant="primary", size="lg")
            
            gr.Markdown("""
            **Quick Tips:**
            * Use **colors** in your prompt (e.g., 'red bottle' instead of just 'bottle').
            * Separate categories with periods `.`.
            * If detection is low, reduce thresholds to **0.20**.
            """)
        
        with gr.Column(scale=2):
            gr.Markdown("### 2. Analysis Results")
            
            with gr.Row():
                reference_output = gr.Image(label="Planogram Detections")
                actual_output = gr.Image(label="Shelf Detections")
            
            report_output = gr.Markdown(label="Compliance Report")
    
    compare_btn.click(
        fn=planogram_comparison,
        inputs=[reference_input, actual_input, prompt_input, box_threshold, text_threshold],
        outputs=[reference_output, actual_output, report_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)