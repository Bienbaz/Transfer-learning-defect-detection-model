# CV Transfer Learning Project: Automated Quality Control with Transfer Learning

## Project Overview
Automated Quality Control using Transfer Learning (ResNet50) for Manufacturing Defect Detection. Implements Discovery-to-Action strategy.

## Dataset Preparation
- Organize images into `data/train/good/`, `data/train/defect/`, `data/test/good/`, `data/test/defect/`.
- Recommended datasets: NEU Steel Defect Dataset or MVTec AD (organize binary good/defect).
- Use ImageDataGenerator for preprocessing (224x224, normalization) and augmentation (rotation, flip, zoom).

## Model Architecture
- Base: ResNet50 (pre-trained on ImageNet, frozen).
- Custom Head: GlobalAveragePooling2D → Dense(128, ReLU) → Dropout → Dense(1, Sigmoid).
- Trained for 10 epochs.

## Performance
Run the notebook to generate classification report with Precision, Recall, etc.

## Factory Automation Logic
If defect probability ≥ 85%, trigger rejection via robotic arm.

## Key Insights
- **GlobalAveragePooling2D Advantage**: Reduces parameters (vs Flatten), retains spatial information, lowers overfitting risk.
- Next Steps: Fine-tune more layers, hyperparameter tuning, deployment with TensorFlow Serving or ONNX.

## Files
- `transfer_learning_quality_control.ipynb`: Full implementation with executed cells (run locally).
- Code is clean, documented with Markdown explanations.

Clone and run in Google Colab or local Jupyter with TensorFlow.