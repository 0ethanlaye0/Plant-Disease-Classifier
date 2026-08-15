# Plant Disease Classifier

A multi-class CNN image classifier that detects diseases in plant leaves across 9 fruit and vegetable categories.

## What it does

Given a photo of a plant leaf, the model predicts the disease condition (or healthy status) of the plant. Supports apple, bell pepper, cherry, corn, grape, peach, potato, strawberry, and tomato.

## How to run

**Requirements:**
```bash
pip install torch torchvision torchmetrics mlxtend Pillow
```

**Download the dataset:**
Download the [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) from Kaggle and organize images as such:

```
images/
  Apple/
    Train/
      Apple_Scab/
      Black_Rot/
      ...
    Test/
      Apple_Scab/
      Black_Rot/
      ...
  Tomato/
    Train/
      ...
    Test/
      ...
  (Bell_Pepper, Cherry, Corn, Grape, Peach, Potato, Strawberry)
```

**Run:**
```bash
python access.py
```

Follow the CLI prompts to select a fruit/vegetable, then choose to train, evaluate (with confusion matrix), or predict on a single image (with file path).

## Sample Model Results — 3 epochs

| Fruit/Vegetable | Classes | Test Size | Test Accuracy |
|----------------|---------|-----------|---------------|
| Apple | 4 | 196 | 96.43% |
| Bell Pepper | 2 | 98 | 96.09% |
| Cherry | 2 | 89 | 98.67% |
| Corn | 4 | 188 | 93.53% |
| Grape | 4 | 182 | 96.35% |
| Peach | 2 | 90 | 100% |
| Potato | 3 | 144 | 98.75% |
| Strawberry | 2 | 91 | 100% |
| Tomato | 6 | 280 | 96.18% |

## Project structure
```
├── access.py            # Main interface — select fruit, train/evaluate/predict
├── model.py             # CNN architecture 
├── helper_functions.py  # Training, testing, and prediction functions
├── image_data/          # Data loading scripts per fruit/vegetable with augmented train data.
│   ├── apple_data.py
│   ├── tomato_data.py
│   └── ...
└── models/              # Saved trained models (not included — train locally)
```

## Model architecture

3-block CNN built in PyTorch:
- 3× Conv2d + ReLU + MaxPool2d
- Flatten + Linear classifier
- Input: 128×128 RGB images
- Loss: CrossEntropyLoss
- Optimizer: Adam (lr=0.001)

## Dataset

[PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) — 54,306 images of healthy and diseased crop leaves across 38 classes.
