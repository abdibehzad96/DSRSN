

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch
import numpy as np
import random
# test the model
import torch.nn as nn
import torch.optim as optim
import snntorch as snn
from DSRN import *
import os

class FaultData(Dataset):
    def __init__(self):
        self.data = []
        self.targets = []
        self.NoiseLevel = []
        self.max = 0
        self.min = 0
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx], self.NoiseLevel[idx]
    
    def prepare(self, df, strlen, ignore):
        inputD = torch.empty(0, strlen, dtype=torch.float32)
        targetD = torch.empty(0, dtype=torch.long)
        
        Pr_inp = torch.tensor(df[0][:-1], dtype=torch.float32).unsqueeze(0)
        Pr_lbl = torch.tensor(df[0][-1], dtype=torch.long).unsqueeze(0)
        for row in df[1:]:
                inpt = torch.tensor(row[:-1], dtype=torch.float32).unsqueeze(0)
                trgt = torch.tensor(row[-1], dtype= torch.long).unsqueeze(0)
                if trgt == Pr_lbl:
                    inptcat = torch.cat((Pr_inp, inpt), dim=1) # concatenate the input data to make it longer
                    Pr_inp = inpt
                    Pr_lbl = trgt
                    if trgt not in ignore: # remove the 10 and 11 classes
                        inputD = torch.cat((inputD, inptcat), dim=0)
                        targetD = torch.cat((targetD, Pr_lbl), dim=0)
                else:
                    Pr_inp = inpt
                    Pr_lbl = trgt
                # get the max value of the input data
        self.max = torch.max(inputD)
        self.min = torch.min(inputD)
        NoiseLevel = torch.ones_like(targetD, dtype=torch.float32)*100  
        self.data, self.targets, self.NoiseLevel = inputD, targetD, NoiseLevel
        print("Data prepared with max and min of:", self.max, self.min)

    def split(self, batch_size = 1, validation_split=0.9, shuffle=False):
        train_size = int(validation_split * len(self))
        test_size = int((1-validation_split) * len(self))
        #val_size = len(dataset) - train_size - test_size
        print("Train size:", train_size, "Test size:", test_size)

        indices = list(range(len(self)))
        random.shuffle(indices)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        #val_indices = indices[train_size + test_size:]

        # Define samplers for training and test sets
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)
        #val_sampler = SubsetRandomSampler(val_indices)
        # # Create data loaders

        train_loader = DataLoader(self, batch_size=batch_size, shuffle=shuffle, sampler=train_sampler)
        test_loader = DataLoader(self, batch_size=batch_size, shuffle=shuffle, sampler=test_sampler)
        #val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=val_sampler)
        return train_loader, test_loader
    
    def str2image(self, x, H, W):
        # zero padding if the lengh is not enough
        if x.size(1) < H*W:
            x = torch.cat((x, torch.zeros(x.size(0), H*W-x.size(1))), dim=1)
            
        if x.size(1) >= H*W:
            x = x[:,:H*W]
        x = x.reshape(x.size(0), H, W)
        x = (x - self.min)/(self.max - self.min)
        return x
    def savett(self, train, test):
        torch.save(train, os.path.join(os.getcwd(), 'processed_data', 'train.pth'))
        torch.save(test,os.path.join(os.getcwd(), 'processed_data', 'test.pth'))
        print('data saved')

    def loadtt(self):
        train = torch.load(os.path.join(os.getcwd(), 'processed_data', 'train.pth'))
        test = torch.load(os.path.join(os.getcwd(), 'processed_data', 'test.pth'))
        print('data loaded')
        return train, test

    def addnoise(self, SNR):
        noisydata = [self.data]
        n = 0
        for snr in SNR:
            signal_power = torch.mean(torch.square(self.data))
            noise_power = signal_power/(10**(snr/10))
            # create noise with the input SNR
            noise = torch.randn(self.data.size(0), self.data.size(1))
            noise = torch.mul(noise, torch.sqrt(noise_power))
            noise = torch.sub(noise, torch.mean(noise))
            noisydata.append(torch.add(self.data, noise))
            n += 1
            #append the noisy data to the original Dataloader
        noisydata = torch.stack(noisydata)
        noisydata = noisydata.clone().detach().view(-1, noisydata.size(-1))
        self.data = noisydata
        trgt_tmp = self.targets[:]
        noislvl_tmp = self.NoiseLevel[:]
        for i in range(n):
            trgt_tmp = torch.cat((trgt_tmp, self.targets), dim=0)
            noislvl_tmp = torch.cat((noislvl_tmp, torch.ones_like(self.targets, dtype=torch.float32)*SNR[i]), dim=0)
        self.targets = trgt_tmp[:]
        self.NoiseLevel = noislvl_tmp[:]



def loadcsv(csvpath):
    df = []
    print("Loading file:", csvpath)
    with open(csvpath, 'r',newline='') as file:
        for line in file:
            row = line.strip().split(',')
            rowf = [float(element) for element in row]
            #rowf = [0 if math.isnan(x) else x for x in rowf]
            df.append(rowf)
    return df, len(df[0])-1


def test_model(test, dataset, model, criterion, device, load=False):
    #load the model from the checkpoint
    imsize = 64
    if load:
        print('model loaded')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DSRN(nb_blocks=1, in_ch=1, out_ch=8, imsize= [imsize,imsize], 
                    device=device,betta = 0.95, thresh= 2).to(device)
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        #optimizer = optim.Adam(model.parameters(), lr=0.05, betas=(0.9, 0.999))
        model.load_state_dict(torch.load('model.pth'))

    test_losses = 0
    pred_results = torch.empty(0, device = device)
    true_labels = torch.empty(0, device = device)
    with torch.no_grad():
        model.eval()
        lp = 0
        for arr, labels, z in test:
            inputs = dataset.str2image(arr,imsize, imsize)
            inputs = inputs.unsqueeze(1).to(device) #add channel dim
            outputs = model(inputs)
            loss = criterion(torch.sum(outputs,dim =0), labels.to(device))
            #_, predicted = torch.max(outputs.sum(1), 0)
            #pred_results= torch.cat((pred_results,predicted), dim = 0)
            #true_labels = torch.cat((true_labels, y.to(device)), dim = 0)
            #tests_loss = criterion(outputs.sum(1), y.to(device))
            lp += inputs.size(0)
            test_losses += loss.item()
    avgloss = test_losses / lp
    return pred_results, true_labels, z, avgloss


def savelog(log, ct): # append the log to the existing log file while keeping the old logs
    # if the log file does not exist, create one
    if not os.path.exists(os.path.join(os.getcwd(),'logs')):
        os.mkdir(os.path.join(os.getcwd(),'logs'))
    with open(os.path.join(os.getcwd(),'logs', f'log-{ct}.txt'), 'a') as file:
        file.write('\n' + log)
        file.close()
