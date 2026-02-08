import os
import torch
from torchvision import transforms
import data_setup, engine, model_builder, utils

NUM_EPOCHS=5
BATCH_SIZE=32
HIDDEN_UNITS=10
LEARNING_RATE=0.001

train_dir="data/pizza_steak_sushi/train"
test_dir="data/pizza_steak_sushi/test"

device='mps' if torch.mps.is_available() else 'cpu'

data_transform=transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

train_dataloader, test_dataloader, class_names = data_setup.create_dataloader(train_dir=train_dir, test_dir=test_dir, transform=data_transform, batch_size=BATCH_SIZE)
model=model_builder.TinyVGG(input_shape=3, hidden_units=HIDDEN_UNITS, output_shape=len(class_names)).to(device)


loss_fn=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

from timeit import default_timer as timer 
start_time = timer()

engine.train(model=model, train_dataloader=train_dataloader,test_dataloader=test_dataloader, loss_fn=loss_fn, optimizer=optimizer, epochs=NUM_EPOCHS, device=device)

end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")
utils.save_model(model=model, target_dir="models", model_name="05_TinyVGG.pth")
