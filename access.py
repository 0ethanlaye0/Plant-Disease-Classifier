from pathlib import Path

from model import *
from helper_functions import *


device = 'cuda' if torch.cuda.is_available() else "cpu"


fruit = input("\nWhat Fruit/Vegetable model will you like to create/access?\n('apple', 'bell pepper', 'cherry', 'corn', 'grape', 'peach', 'potato', 'strawberry', 'tomato')\n-> ")
if fruit == 'apple':
    from image_data.apple_data import *
elif fruit == 'bell pepper':
    from image_data.bell_pepper_data import *
    fruit = 'bell_pepper'
elif fruit == 'cherry':
    from image_data.cherry_data import *
elif fruit == 'corn':
    from image_data.corn_data import *
elif fruit == 'grape':
    from image_data.grape_data import *
elif fruit == 'peach':
    from image_data.peach_data import *
elif fruit == 'potato':
    from image_data.potato_data import *
elif fruit == 'strawberry':
    from image_data.strawberry_data import *
elif fruit == 'tomato':
    from image_data.tomato_data import *
else:
    print(f"No available data for {fruit}. Aborting...")
    exit()

model = Disease_Classifier(3, 64, len(class_names)).to(device)
MODEL_SAVE_PATH = Path('models') / f'{fruit}_model.pth'

loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(params = model.parameters(),
                       lr = 0.001)


model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
print(f"\nCurrent '{fruit}' model state:")
test_model(model, test_batches, loss_fn, accuracy_fn)


task = input(f"\nHow would you like to interact with '{fruit}' Disease Classifier? \n1. Predict a fruit's condition('pred') \n2. Test Model('eval')  \n3. Train Model and evaluate('train')\n-> ")



epochs = 1
if task == 'pred':
    print("\n")
    img_path = input("Enter image path: ")
    pred = predict(Path(img_path), model, class_names, torch.device(device))
    print("Evaluating image...")
    print(f"The '{fruit}' appears to be of '{pred}' condition")

elif task == 'train':
    epochs = input(f"\nEnter Train Epochs -> ")
    print(f"\nTraining and Evaluating '{fruit}' Disease Classifier Model...")
    for epoch in range(int(epochs)):
        print(f"\nEpoch: {epoch}")
        train_model(model,train_batches,loss_fn,accuracy_fn,opt)
        test_model(model,test_batches,loss_fn,accuracy_fn)


    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

else:
    print("Invalid command. Aborting...")