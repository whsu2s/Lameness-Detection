import time
import os
import pandas as pd
from data.skeleton_dataset import SkeletonDataset
from models.hrnn import HRNN
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler

args = {
    'batch_size': 16,
    'n_epochs': 10,
    'lr': 1e-3,
}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run():
    data_dir = 'data/'
    csv_file = os.path.join(data_dir, 'data_labels.csv')
    dataset = {x: SkeletonDataset(os.path.join(data_dir, x), csv_file) for x in ['train', 'val', 'test']}
    dataloader = {x: DataLoader(dataset[x], batch_size=args['batch_size'], shuffle=True, num_workers=4)
                   for x in ['train', 'val', 'test']}

    model = HRNN(input_size=50, hidden_size=50, num_layers=1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)

    best_acc = 0.0
    train_acc_history = []
    train_loss = []
    #"""
    for epoch in range(args['n_epochs']):
        print('Epoch {}/{}'.format(epoch+1, args['n_epochs']))
        print('-' * 10)

        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for cow, sample in dataloader['train']:
            inputs = sample['seq'] #.view(args['batch_size'], 20, 50)
            inputs = inputs.float().to(device)  # requires float32
            labels = sample['label'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataset['train'])
        epoch_acc = running_corrects.double() / len(dataset['train'])

        train_loss.append(epoch_loss)
        train_acc_history.append(epoch_acc)

        print('Training Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print(epoch_loss)

    # load best model weights
    #model.load_state_dict(best_model_wts)
    #"""


if __name__ == "__main__":
    run()
