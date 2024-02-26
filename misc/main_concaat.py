from util import *
from dsrn_concat import *
import sys
import os
cwd = os.getcwd()
print(cwd)
csvpath = R"Vib_data.csv"
df, datalen= loadcsv(csvpath)
dataset = FaultData()
dataset.prepare(df, datalen, ignore = [10,11,12])
train, test = dataset.split(128,0.85)
# Create the model
imsize = 64
Loadmodel = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DSRN(nb_blocks=32, in_ch=1, out_ch=8, imsize= imsize, device=device).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) # decay LR by a factor of 0.1 every 10 epochs
if Loadmodel == True:
    model.load_state_dict(torch.load('model.pth'))
    print('model loaded')
# check if cuda is available
if torch.cuda.is_available():
    model = model.to(device)
    criterion = criterion
    print("cuda is available")
# Create the model
epochs = 200
loss_ep = []
prev_loss = 10000
# Train the model
for epoch in range(epochs):
    train_loss = 0.0
    train_batch = iter(train)
    for inputs, labels in train_batch:
        model.train()
        # forward + backward + optimize
        inputs = dataset.str2image(inputs,imsize, imsize)
        inputs = inputs.unsqueeze(1).to(device) #add channel dim
        outputs = model(inputs)
        loss = criterion(outputs.sum(1), labels.to(device))
        # zero the parameter gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print statistics
        train_loss += loss.item()
    loss_ep.append(train_loss / len(train))
    lr_scheduler.step() #decay learning rate
    print('epoch: %d loss: %.3f' % (epoch + 1, train_loss / len(train)))
    if prev_loss > train_loss:
        torch.save(model.state_dict(), 'model.pth')
        print('checkpoint saved')
        prev_loss = train_loss
