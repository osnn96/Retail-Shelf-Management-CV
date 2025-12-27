---
title: Retail Shelf Compliance
emoji: 🛒
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 4.12.0
app_file: app.py
pinned: false
license: mit
---

# 🛒 Retail Shelf Compliance System

AI-powered product detection and segmentation for retail environments using **Grounding DINO** and **SAM (Segment Anything Model)**.

## 🌟 Features

- **🎯 Zero-Shot Detection**: Detect any product using natural language prompts
- **🎨 Pixel-Perfect Segmentation**: SAM-powered instance segmentation
- **⚡ Real-Time Processing**: Fast inference on uploaded images
- **🌐 Easy to Use**: Simple web interface powered by Gradio

## 🚀 How to Use

1. **Upload an image** of a retail shelf
2. **Enter a prompt** describing what to detect (e.g., "bottle. can. chocolate bar")
3. **Adjust thresholds** (optional) for fine-tuning
4. **Click "Detect & Segment"** to see results

## 💡 Example Prompts

- `bottle. can. beverage` - For drinks section
- `chocolate bar. candy. snack` - For snacks aisle
- `shampoo. soap. toiletries` - For personal care
- `blister pack. pills. medication` - For pharmacy section

## 🎯 Use Cases

- **Retail Compliance**: Verify shelf layouts and planograms
- **Inventory Management**: Automated stock counting
- **Quality Control**: Detect missing or misplaced products
- **Market Research**: Analyze product placement

## 🔧 Technology

### Models Used

- **Grounding DINO Tiny** (`IDEA-Research/grounding-dino-tiny`)
  - Open-vocabulary object detection
  - Natural language-based detection
  - Fast and accurate

- **SAM ViT-H** (Segment Anything Model)
  - State-of-the-art segmentation
  - Pixel-perfect masks
  - Universal object segmentation

### Framework

- **Gradio**: Web interface
- **PyTorch**: Deep learning framework
- **Supervision**: Computer vision utilities
- **Transformers**: Model loading and inference

## ⚙️ Configuration

### Thresholds

- **Box Threshold** (default: 0.30)
  - Controls detection confidence
  - Higher = fewer but more confident detections
  - Lower = more detections but may include false positives

- **Text Threshold** (default: 0.25)
  - Controls text matching strictness
  - Higher = stricter matching to prompt
  - Lower = more flexible matching

### Tips for Best Results

✅ Use clear, well-lit images  
✅ Specify concrete object names  
✅ Separate multiple objects with periods (.)  
✅ Try adjusting thresholds if results aren't satisfactory  

❌ Avoid overly generic prompts like "thing" or "item"  
❌ Don't use very blurry or dark images  

## 📊 Performance

- **First Run**: ~30-60 seconds (model loading + weights download)
- **Subsequent Runs**: ~3-5 seconds per image
- **Hardware**: CPU (free tier) or GPU (for faster inference)

## 🙏 Acknowledgments

- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) - Open-vocabulary object detection
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) - Universal image segmentation
- [Supervision](https://github.com/roboflow/supervision) - Computer vision utilities
- [Gradio](https://gradio.app) - Web interface framework

## 📄 License

MIT License - feel free to use this for your projects!

## 🔗 Links

- [Live on HuggingFace](https://huggingface.co/spaces/osnn96/retail-shelf-compliance)

---

**Note:** First run will be slower as it downloads the SAM model weights (~2.4GB). This is a one-time download and subsequent runs will be much faster.

Built by using Hugging Face Spaces