import network
import dataset

import torch
import torch.optim as optim
import torch.nn.functional as F
import os
from shutil import copyfile
import numpy as np
import roc
def compute_weighted_loss(output, label):
    # return loss weighted by negative/positive quotient in training data for each image size
    lab = label.cpu().numpy().flatten()

    if lab.size == 3600:
        loss = F.binary_cross_entropy_with_logits(output[0, :, :, :], label, reduction='mean',
                                                  pos_weight=torch.tensor(2000))
    elif lab.size == 1378:
        loss = F.binary_cross_entropy_with_logits(output[0, :, :, :], label, reduction='mean',
                                                  pos_weight=torch.tensor(800))
    elif lab.size == 800:
        loss = F.binary_cross_entropy_with_logits(output[0, :, :, :], label, reduction='mean',
                                                  pos_weight=torch.tensor(400))
    elif lab.size == 338:
        loss = F.binary_cross_entropy_with_logits(output[0, :, :, :], label, reduction='mean',
                                                  pos_weight=torch.tensor(150))
    return loss

if __name__ == '__main__':
    print("Hi11!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # initialize network
    net = network.Net()
    cwd = os.getcwd()
    # load pretrained weights
    net.load_state_dict(torch.load(cwd + "/yolo_pretrained"))
    # move network on GPU if GPU is available

    net.to(device)

    # set export dir and make copy of current training script inside
    export_path = 'net_training'

    if not os.path.exists(export_path):
        os.makedirs(export_path)
    copyfile(cwd + '/train.py', export_path+'/train_script.py')

    # training parameters
    batch_size = 50
    epochs = 50
    learning_rate = 0.0001
    freeze_layers_for_epoch = 5 # freeze pretrained part od the network for number of epochs
    flip_images = False

    # load datasets
    dataset_trn = dataset.myDataset(cwd + "/ds/train",transform=network.ToTensor(flip=flip_images))
    dataset_val = dataset.myDataset(cwd + "/ds/val",transform=network.ToTensor())
    trainloader = torch.utils.data.DataLoader(dataset_trn, batch_size=1, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=2)
    print("last11")
    # loop over the dataset multiple times
    for epoch in range(epochs):
        batch = 0
        # create optimizer and set up learning rate
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.1)

        # freeze pretrained layers
        if epoch < freeze_layers_for_epoch:
            net.train_only_last(True)
        else:
            net.train_only_last(False)

        sum_loss = 0
        for i, data in enumerate(trainloader):

            batch += 1
            # load data
            input = data['image'].to(device)
            label = data['label'].to(device)

            # network forward pass
            output = net(input)

            # compute loss
            loss = compute_weighted_loss(output, label)
            # accumulate loss for one batch
            sum_loss += loss.item()

            # network backward pass
            loss.backward()

            if batch == batch_size:

                # divide gradients by batch_size
                for p in net.parameters():
                    if p.grad is not None:
                        p.grad = p.grad/batch_size
                    # normalize gradient size
                    torch.nn.utils.clip_grad_norm_(p,1)
                # upgrade gradients
                optimizer.step()
                optimizer.zero_grad()
                batch = 0
                # print loss per batch
                s = ('%g/%g' % (epoch, epochs - 1), '%g/%g' % (i, len(trainloader) - 1), '%f' % (sum_loss/batch_size))
                print(s)
                sum_loss = 0
        # save network wights after each epoch
        torch.save(net.state_dict(), export_path+'/net_epoch'+str(epoch))

        # validation dataset
        sum_val_loss = 0
        for i, data in enumerate(valloader):
            input = data['image'].to(device)
            label = data['label'].to(device)
            output = net(input)
            loss = compute_weighted_loss(output, label)
            sum_val_loss += loss.item()
        s = ('validation: %g/%g' % (epoch, epochs - 1), '%f' % (sum_val_loss/len(valloader)))
        print(s)
    print("Calling plot!")
    roc.plot(epochs)
