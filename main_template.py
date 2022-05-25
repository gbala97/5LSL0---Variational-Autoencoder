# %% imports
# libraries
from numpy import dtype
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F 

# local imports
import MNIST_dataloader
from autoencoder_template import *

# %% set torches random seed
torch.random.manual_seed(0)

# %% preperations
# define parameters
data_loc = 'MNIST Dataset' #change the data location to something that works for you
batch_size = 64
no_epochs = 4
learning_rate = 3e-4

device = "cuda" if torch.cuda.is_available() else "cpu"

# get dataloader
train_loader, test_loader = MNIST_dataloader.create_dataloaders(data_loc, batch_size)

# create the autoencoder
model = AE()

# create the optimizer
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
criterion = F.mse_loss


# %% training loop
# go over all epochs

epoch_loss = []
epoch_counter = []
model.to(device = device)
for epoch in range(no_epochs):
    print(f"\nTraining Epoch {epoch}:")
    # go over all minibatches
    train_loss = []
    model.train()
    for batch_idx,(x_clean, x_noisy, label) in enumerate(tqdm(train_loader)):
        
        # fill in how to train your network using only the clean images

        x_clean = x_clean.to(device = device, dtype=torch.float32)
        output = model(x_clean)

        #print(output[0].shape)
        #print(x_clean.shape)
        loss = criterion(output, x_clean)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

    epoch_loss.append(sum(train_loss)/len(train_loss))
    epoch_counter.append(epoch)

    print(f'Loss for epoch {epoch} is {epoch_loss[epoch]*10}')

print('Training Complete of autoencoder complete')
torch.save(model.state_dict(),'model.pth')

# %% HINT
#hint: if you do not care about going over the data in mini-batches but rather want the entire dataset use:
x_clean_train = train_loader.dataset.Clean_Images
x_noisy_train = train_loader.dataset.Noisy_Images
labels_train  = train_loader.dataset.Labels

x_clean_test  = test_loader.dataset.Clean_Images
x_noisy_test  = test_loader.dataset.Noisy_Images
labels_test   = test_loader.dataset.Labels

# use these 10 examples as representations for all digits
x_clean_example = x_clean_test[0:10,:,:,:]
x_noisy_example = x_noisy_test[0:10,:,:,:]
labels_example = labels_test[0:10]


out = model(x_clean_example)
x_predicted_out = out.detach().numpy()
# show the examples in a plot
plt.figure(figsize=(12,3))
for i in range(10):
    plt.subplot(2,10,i+1)
    plt.imshow(x_clean_example[i,0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(2,10,i+11)
    plt.imshow(x_predicted_out[i,0,:,:],cmap='gray')
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.savefig("data_examples.png",dpi=300,bbox_inches='tight')
plt.show()


# %%
plt.plot(epoch_counter, epoch_loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
# %%
