import numpy as np
from sklearn import metrics
import network
import torch
import dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# init network

# set up network weights directory



def plot(epochs):
    print("H1211")
    net = network.Net()
    dir = "/home.nfs/ivanomik/workspace/scripts/net_training123/"
    for j in range(0,epochs):
        name = "net_epoch"+str(j)
        # load weights
        net.load_state_dict(torch.load(dir+name))
        # set up evaluation mode of network
        net.eval()
        # set up data loader
        dataset_val = dataset.myDataset("/opt/barbie/barbie_validation_data", transform=network.ToTensor())
        valloader = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=1)
        # move to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        L = []
        O = []
        # process epoch
        print('epoch '+str(j))
        for i, data in enumerate(valloader):


            input = data['image']
            label = data['label']
            lab = label.numpy().flatten()
            output = net(input)
            out = output[0, :, :, :].detach().cpu().numpy().flatten()
            L.append(lab)
            O.append(out)

        L = np.concatenate(L)
        O = np.concatenate(O)

        # compute fpr, tpr
        fpr, tpr, thresholds = metrics.roc_curve(L, O)
        # plot roc for epoch
        plt.clf()
        plt.semilogx(fpr, tpr, color='darkorange',lw=2, label='ROC curve')
        plt.xlim([0.000001, 1.0])
        plt.ylim([0.0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig(dir+name+".png")
