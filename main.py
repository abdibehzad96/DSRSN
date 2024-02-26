from util import *
from DSRN import *
import sys
import os
import datetime
cwd = os.getcwd()
print(cwd)

dataset = FaultData()
dataset.max , dataset.min = 11.6740, -10.0919
batch_size = 256
SNR = torch.tensor([-4,-3,-2,-1,0,1,2,3,4])
# csvpath = "/home/abdikhab/comp/Vib_data.csv"
# df, datalen= loadcsv(csvpath)
# dataset.prepare(df, datalen*2, ignore = [10,11,12])
# dataset.addnoise(SNR)
# train, test = dataset.split(batch_size,0.85)
# dataset.savett(train ,test)
train, test = dataset.loadtt()

# Create the model
imsize = 64
Loadmodel = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kernel_size = [3,5,7]
blocks = [8]
thhold = [1,] # best threshold is 0.96
bett = [.88] # best beta is 0.86
learning_rate = [1e-3,1.2e-3]
weight_dec= [0.004, 0.005]
patience_limit = 120
epochs = 400
ct = datetime.datetime.now().strftime("%m-%d-%H-%M")
Best_performance = 1000
Best_iter = []
lr, bt, th, ks,bl = 1e-3, 0.88, 1, 7, 8
# for loops in beta and kernel size
cwd = os.getcwd()
for ks in kernel_size:
    for bl in blocks:
        for th in thhold:
            for bt in bett:
                for lr in learning_rate:
                      for wdec in weight_dec:
                        iter = "Noise" + str(int(lr*1e3)) +'-'+ str(int(bt*100)) +'-'+ str(th) +'-'+ str(ks) + '-'+ str(bl) + '-'+ str(int(wdec*10000))
                        print(f"running iteration {iter}")
                        print(f"learning rate: {lr}, beta: {bt}, thres: {th}, kernel size: {ks}")
                        savelog(f"running iteration {iter}, learning rate: {lr}, beta: {bt}, thres: {th}, kernel size: {ks}",ct)
                        model = DSRN(nb_blocks=bl, in_ch=1, out_ch=8, imsize= [imsize,imsize], 
                                    device=device, betta = bt, thresh= th, kernelsize = ks).to(device)
                        # Loss and optimizer
                        criterion = nn.CrossEntropyLoss()
                        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wdec) # 0.00001*batch_size/32
                        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.2) # decay LR by a factor of 0.1 every 10 epochs
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
                        lt = 0
                        loss_test = []
                        prev_loss = 10000
                        prev_loss_test = 10000
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
                            model.train()
                            for arr, labels, z in train:
                                # forward + backward + optimize
                                inputs = dataset.str2image(arr,imsize, imsize)
                                inputs = inputs.unsqueeze(1).to(device) #add channel dim
                                optimizer.zero_grad()
                                outputs = model(inputs)
                                loss = criterion(torch.sum(outputs,dim =0), labels.to(device))
                                # zero the parameter gradients
                                loss.backward()
                                optimizer.step()
                                # print statistics
                                train_loss += loss.item()
                                lp += inputs.size(0)
                            _, _,_, lt =test_model(test,dataset, model, criterion, device, False)
                            loss_test.append(lt)
                            loss_ep.append(train_loss /lp )
                            if epoch < 250:
                                lr_scheduler.step() #decay learning rate
                            print('epoch: %d loss: %.3f e-3,  test: %.3f e-3' % (epoch + 1, 1000*loss_ep[-1], 1000*loss_test[-1]))
                            savelog('epoch: %d loss: %.3f e-3,  test: %.3f e-3' % (epoch + 1, 1000*loss_ep[-1], 1000*loss_test[-1]),ct)
                            if prev_loss > loss_ep[-1]:
                                if prev_loss_test > loss_test[-1]:
                                    torch.save(model.state_dict(), os.path.join(cwd,'model', f'model{iter}.pth'))
                                    torch.save(optimizer.state_dict(), os.path.join(cwd,'model', f'optimizer{iter}.pth'))
                                    torch.save(loss_ep, os.path.join(cwd,'model', f'loss{iter}.pth'))
                                    torch.save(outputs, os.path.join(cwd,'model', f'output.pth'))
                                    torch.save(loss_test, os.path.join(cwd,'model', f'loss_test{iter}.pth'))
                                    prev_loss_test = loss_test[-1]
                                    print('checkpoint saved')
                                    savelog('checkpoint saved',ct)
                                    patience = 0
                                    prev_loss = loss_ep[-1]

                            if Best_performance > prev_loss:
                                Best_performance = prev_loss
                                Best_iter = f"Best iteration {iter}, learning rate: {lr}, beta: {bt}, thres: {th}, kernel size: {ks}, " + (' epoch: %d, loss: %.3f e-3,  test: %.3f e-3, ct: ' % (epoch + 1, 1000*loss_ep[-1], 1000*loss_test[-1])) + ct
                                print('Hit the prev performance! Hooray ', 1000*Best_performance)
                                savelog(Best_iter, ct)
                            if patience > patience_limit:
                                print('early stopping')
                                savelog('early stopping', ct)
                                break
                        print("model is saved as ", f'model{iter}.pth')
                        savelog(f"model is saved as model{iter}.pth", ct)
                        print("best performance is ", Best_performance)
                        savelog(Best_iter, ct)
