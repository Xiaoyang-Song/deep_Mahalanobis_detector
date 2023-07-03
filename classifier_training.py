from models.densenet import *
from dataset import *
from tqdm import tqdm
import os
import argparse
import time
import yaml

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_backbone_D(model, val_loader):
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()
        val_loss, val_acc, total_acc = [], [], []
        for idx, (img, label) in enumerate(tqdm(val_loader)):
            img, label = img.to(DEVICE), label.to(DEVICE)
            logits = model(img)
            loss = criterion(logits, label)
            num_correct, num_total = (torch.argmax(logits, dim=1) ==
                                      label).sum().item(), label.shape[0]
            acc = num_correct / num_total
            # print(acc)
            val_acc.append(acc)
            total_acc.append([num_correct, num_total])
            val_loss.append(loss.detach().item())
        print(np.mean(val_acc))
        print(np.mean(val_loss))
        total_acc = np.array(total_acc)
        print(np.sum(total_acc[:, 0] / np.sum(total_acc[:, 1])))

log_dir = "pre_trained/"
os.makedirs(log_dir, exist_ok=True)
# model = DenseNet3(depth=100, num_classes=8, input_channel=3).to(DEVICE)
# img_info = {'H': 28, 'W': 28, 'C': 1}
# model = DC_D(10, img_info).to(DEVICE)
# model = DenseNet3(depth=100, num_classes=8, num_channels=1).to(DEVICE)

# CIFAR10 pre training
model = DenseNet3(depth=100, num_classes=10, num_channels=3).to(DEVICE)
lr = 1e-1
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
# CIFAR10-SVHN
dset = DSET("CIFAR10-SVHN", False, 64, 128, None, None)
# MNIST
# dset = DSET("MNIST", True, 50, 256, [2, 3, 6, 8, 9], [1, 7])
# SVHN
# dset = DSET("SVHN", True, 50, 64, [0, 1, 2, 3, 4, 5, 6, 7], [8, 9])
# dset = DSET("FashionMNIST", True, 50, 64, [0, 1, 2, 3, 4, 5, 6, 7], [8, 9])
# dset = DSET("MNIST-FashionMNIST", False, 256, 256, None, None)

ind_tri_loader = dset.ind_train_loader
ind_val_loader = dset.ind_val_loader
max_epoch = 300


criterion = nn.CrossEntropyLoss()
# Simple training loop
iter_count_train = 0
iter_count_val = 0
for epoch in tqdm(range(max_epoch)):
    if epoch == 150 or epoch == 225:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr * 0.1, betas=(0.9, 0.999))
    # Training
    model.train()
    train_loss, train_acc, wass = [], [], []
    for idx, (img, labels) in enumerate(ind_tri_loader):
        img = img.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(img)
        loss = criterion(logits, labels)
        loss.backward()
        # ic(loss)
        optimizer.step()
        # Append training statistics
        acc = (torch.argmax(logits, dim=1) ==
               labels).sum().item() / labels.shape[0]
        train_acc.append(acc)
        train_loss.append(loss.detach().item())
        iter_count_train += 1

    print(f"\nEpoch  # {epoch + 1} | training loss: {np.mean(train_loss)} \
            | training acc: {np.mean(train_acc)}")
    # Evaluation
    # scheduler.step()
    model.eval()
    with torch.no_grad():
        val_loss, val_acc = [], []
        for idx, (img, labels) in enumerate(ind_val_loader):
            img, labels = img.to(DEVICE), labels.to(DEVICE)
            logits = model(img)
            loss = criterion(logits, labels)
            acc = (torch.argmax(logits, dim=1) ==
                   labels).sum().item() / labels.shape[0]
            val_acc.append(acc)
            val_loss.append(loss.detach().item())
            iter_count_val += 1

        print(f"Epoch  # {epoch + 1} | validation loss: {np.mean(val_loss)} \
            | validation acc: {np.mean(val_acc)}")
with torch.no_grad():
    # Evaluation
    torch.save(model.state_dict(),
               log_dir + f"[{dset.name}]-pretrained-classifier.pt")
    ic("Model Checkpoint Saved!")
    test_backbone_D(model, dset.ind_val_loader)
