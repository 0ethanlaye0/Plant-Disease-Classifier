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

## Project structure
```
├── access.py            # Main interface — select fruit, train/evaluate (with confusion matrix)/predict
├── model.py             # CNN architecture 
├── helper_functions.py  # Training, testing, and prediction functions
├── image_data/          # Data loading scripts per fruit/vegetable with augmented train data.
│   ├── apple_data.py
│   ├── tomato_data.py
│   └── ...
└── models/              # Saved trained models (not included — train locally)
```

## Model architecture

- Input: 128 x 128 RGB images
- 3 Convolutional blocks + 1 classifier block
```
      Input (3 × 128 × 128)
      ↓
      Conv2d(3→64) → ReLU → MaxPool → (64 × 64 × 64) 
      ↓
      Conv2d(64→64) → ReLU → MaxPool → (64 × 32 × 32)
      ↓
      Conv2d(64→64) → ReLU → MaxPool → (64 × 16 × 16)
      ↓
      Flatten → Linear(65536 → num_classes)
      ↓
      Class probabilities
```
- Loss: CrossEntropyLoss
- Optimizer: Adam (lr=0.001)

## Sample Model Results — 3 epochs

| Fruit/Vegetable | Classes | Test Size | Test Accuracy |
|----------------|---------|-----------|---------------|
| Apple | 4 | 196 | 96.43% |
| Bell Pepper | 2 | 98 | 99.22% |
| Cherry | 2 | 89 | 98.67% |
| Corn | 4 | 188 | 93.53% |
| Grape | 4 | 182 | 96.35% |
| Peach | 2 | 90 | 100% |
| Potato | 3 | 144 | 98.75% |
| Strawberry | 2 | 91 | 100% |
| Tomato | 6 | 280 | 96.18% |

## Observations

The models achieved 93.53–100% test accuracy after about three training epochs. Performance varied by crop, reflecting differences in the number of disease classes and the visual similarity between disease categories.

The two lowest-performing models which were corn (93.53%) and tomato (96.18%), showed the most class confusion during evaluation:

**Corn:** The model occasionally confused Cercospora Leaf Spot with Northern Leaf Blight. Both diseases produce elongated, rectangular spots that run parallel to the leaf veins with similar gray coloring, which makes them visually difficult to distinguish.

**Tomato:** The model occasionally confused Bacterial Spot with Early Blight. Both diseases produce small dark spots surrounded by yellow discoloration leaves.

The remaining 7 models showed minimal confusion, likely due to more visually distinct disease classes or fewer classes per model.


## Dataset

[PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) — 54,306 images of healthy and diseased crop leaves across 38 classes.

### Limitations
- Models train on controlled laboratory-style images from the PlantVillage dataset (uniform backgrounds, consistent lighting). Therefore, photos taken in natural conditions that aren't similar to those from dataset may reduce prediction accuracy.
- For best results, photograph a single leaf against a plain background with good lighting, similar to those from the dataset.
- Each model only classifies diseases present in the PlantVillage dataset for that specific plant. Therefore, diseases not in the dataset will be misclassified as the closest matching class.
