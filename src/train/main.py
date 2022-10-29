''' Main training script '''

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, RandomHorizontalFlip, ColorJitter

class ImageTextDataset(Dataset):
  pass

def train(data_path, **data_kwargs):
  # load data from tars
  train_dataset = ImageTextDataset(data_path, train=True)
  eval_dataset = ImageTextDataset(data_path, train=False)
  
  # load DataLoaders
  train_loader = DataLoader(train_dataset, **data_kwargs)
  eval_loader = DataLoader(eval_dataset, **data_kwargs)
  
  # load model
  model = Flamingo(...)
  
  # train model
  for epoch in range(num_epochs):
    for batch in train_loader:
      optim.zero_grad()
      inputs, labels = [v.to(device) for v in batch]
      preds = model(inputs)
      loss = loss_fn(preds, labels)
      loss.backward()
      optim.step()
     
    # evaluate model on zero-shot ImageNet
    with torch.no_grad():
      total_correct = 0
      
      for batch in eval_loader:
        optim.zero_grad()
        inputs, labels = [v.to(device) for v in batch]
        preds = model(inputs)
        total_correct += (preds.argmax(...) == labels).sum()
      
      print(f"acc @{epoch}: {total_correct / len(eval_dataset)}")
