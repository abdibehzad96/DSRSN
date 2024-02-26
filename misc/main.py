from util import *
from DSRN import *
import sys
import os
cwd = os.getcwd()
print(cwd)

dataset = FaultData()
dataset.max , dataset.min = 11.6740, -10.0919
batch_size = 128
train, test = dataset.loadtt()
# csvpath = "/home/abdikhab/comp/Vib_data.csv"
# df, datalen= loadcsv(csvpath)
# dataset.prepare(df, datalen*2, ignore = [10,11,12])
# train, test = dataset.split(batch_size,0.85)
# dataset.savett(train ,test)

# Create the model
imsize = 64
Loadmodel = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
beta = [0.2,0.6,0.8,0.9,0.95]
kernel_size = [3,5,7,9]
blocks = [4,8,16,32,64]
threshold = [1]
epochs = 600
bt, ks, bl = 0.111, 7, 1
patience_limit = 100
# for loops in beta and kernel size
cwd = os.getcwd()
for th in threshold:
#     for ks in kernel_size:
#         for bl in blocks:
    iter = str(int(bt*100)) +'-'+ str(th) +'-'+ str(bl)
    print(f"running iteration {iter}")
    model = DSRN(nb_blocks=bl, in_ch=1, out_ch=8, imsize= [imsize,imsize], 
                device=device, betta = bt, thresh= th, kernelsize = ks).to(device)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0008, betas=(0.9, 0.999))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8) # decay LR by a factor of 0.1 every 10 epochs
    if Loadmodel == True:
        model.load_state_dict(torch.load('model.pth'))
        print('model loaded')
    # check if cuda is available
    if torch.cuda.is_available():
        model = model.to(device)
        criterion = criterion
        print("cuda is available")
    # Create the model
    loss_ep = []
    prev_loss = 10000
    patience = 0
    # Train the model
    model.train()
    for param in model.parameters(): #
        if param.requires_grad == False:
            print(param)
    param.requires_grad = True

    for epoch in range(epochs):
        train_loss = 0.0
        lp = 0
        patience += 1
        for arr, labels in train:
            # forward + backward + optimize
            inputs = dataset.str2image(arr,imsize, imsize)
            inputs = inputs.unsqueeze(1).to(device) #add channel dim
            outputs = model(inputs)
            loss = criterion(torch.sum(outputs,dim =0), labels.to(device))
            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print statistics
            train_loss += loss.item() * inputs.size(0)
            lp += inputs.size(0)
        loss_ep.append(train_loss /lp )
        lr_scheduler.step() #decay learning rate
        print('epoch: %d loss: %.6f' % (epoch + 1, train_loss / lp))
        if prev_loss > train_loss:
            torch.save(model.state_dict(), os.path.join(cwd,'model', f'model{iter}.pth'))
            torch.save(optimizer.state_dict(), os.path.join(cwd,'model', f'optimizer{iter}.pth'))
            torch.save(loss_ep, os.path.join(cwd,'model', f'loss{iter}.pth'))
            torch.save(outputs, os.path.join(cwd,'model', f'output.pth'))
            print('checkpoint saved')
            prev_loss = train_loss
            patience = 0
        if patience > patience_limit:
            print('early stopping')
            print("modelis saved as ", f'model{iter}.pth')
            break
