import torch
import torch.nn as nn
from model.ClaHi_GAT import *
from dataset import *
import queue
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = PHEME_dataset()
length=len(dataset)
train_size, vali_size, test_size = int(0.8*length), int(0.1*length), length-int(0.8*length)-int(0.1*length)
train_set, vali_set, test_set = torch.utils.data.random_split(dataset, [train_size, vali_size, test_size])


train_loader = tqdm(torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0))

vali_loader = tqdm(torch.utils.data.DataLoader(vali_set, batch_size=1, shuffle=True, num_workers=0))

test_loader = tqdm(torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0))


model = ClaHi_GAT()
model.to(device)

epoch = 2
lr = 0.00005

CEloss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

correct_rate_list = []
best_correct_rate = 0
for i in range(epoch):
    train_correct = 0
    pred_list = []
    label_list = []
    for idx, (text, adj, label) in enumerate(train_loader):
        text.to(device)
        adj.to(device)
        label = label.float()
        label.to(device)

        model.train()
        pred = model(text, adj)
        pred_list.append(pred)
        label_list.append(label)
        if len(pred_list) == 16:  # batch_size
            loss = CEloss(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {i} | Batch{idx} | Train_Loss {loss}")
            pred_list = []
            label_list = []

        if pred.tolist() == label.tolist():
            train_correct += 1
    train_correct_rate = train_correct / len(train_loader)
    print(f"Train Accuracy: {train_correct_rate}")

    vali_correct = 0
    for idx, (text, adj, label) in enumerate(vali_loader):
        text.to(device)
        adj.to(device)
        label = label.float()
        label.to(device)

        model.test()
        pred = model(text, adj)

        if pred.tolist() == label.tolist():
            vali_correct += 1

    vali_correct_rate = vali_correct / len(vali_loader)
    correct_rate_list.append(vali_correct_rate)
    print(f"Test Accuracy: {vali_correct_rate}")

    if vali_correct_rate > best_correct_rate:
        torch.save(model, 'best_model.pt')

    if len(set(correct_rate_list[-5:])) == 1:
        break

model = torch.load('best_model.pt')
test_correct = 0
for idx, (text, adj, label) in enumerate(test_loader):
    text.to(device)
    adj.to(device)
    label = label.float()
    label.to(device)

    model.test()
    pred = model(text, adj)

    if pred.tolist() == label.tolist():
        test_correct += 1

test_correct_rate = test_correct / len(test_loader)
correct_rate_list.append(test_correct_rate)
print(f"Test Accuracy: {test_correct_rate}")
