import numpy as np
import scipy.io
from nn import *
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import string
import pickle

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
# batch_size = 128
#learning_rate =  1e-2
batch_size = 36
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
##########################
##### your code here #####
##########################
initialize_weights(train_x.shape[1], hidden_size, params, "layer1")
initialize_weights(hidden_size, hidden_size, params, "layer2")
initialize_weights(hidden_size, hidden_size, params, "layer3")
initialize_weights(hidden_size, train_x.shape[1], params, "output")

# should look like your previous training loops
losses = []
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   use 'm_'+name variables in initialize_weights from nn.py
        #   to keep a saved value
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        ##########################
        ##### your code here #####
        ##########################
        # forward pass
        h1 = forward(xb, params, "layer1", relu)
        h2 = forward(h1, params, "layer2", relu)
        h3 = forward(h2, params, "layer3", relu)
        output_img = forward(h3, params, "output", sigmoid)

        # loss
        loss = np.sum((output_img - xb)**2)
        total_loss += loss

        # backward
        delta1 = 2*output_img - 2*xb
        delta2 = backwards(delta1, params, "output", sigmoid_deriv)
        delta3 = backwards(delta2, params, "layer3", relu_deriv)
        delta4 = backwards(delta3, params, "layer2", relu_deriv)
        grad_xb = backwards(delta4, params, "layer1", relu_deriv)

        # apply gradient, remember to update momentum as well
        for k, v in sorted(list(params.items())):
            if "grad" in k:
                name = k.split("_")[1]
                if "W" in name:
                    Mw = params["m_" + name]
                    Mw = 0.9 * Mw - learning_rate * v
                    params[name] = params[name] + Mw
                else:
                    params[name] = params[name] - learning_rate * v

    losses.append(total_loss/train_x.shape[0])
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9
# --
# save the final network
saved_params = {k: v for k, v in params.items() if "_" not in k}
with open("q5_weights.pickle", "wb") as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
# --

# plot loss curve
plt.plot(range(len(losses)), losses)
plt.xlabel("epoch")
plt.ylabel("average loss")
plt.xlim(0, len(losses)-1)
plt.ylim(0, None)
plt.grid()
plt.show()



# Q5.3.1
# choose 5 labels (change if you want)
visualize_labels = ["A", "B", "C", "1", "2"]

# get 2 validation images from each label to visualize
visualize_x = np.zeros((2*len(visualize_labels), valid_x.shape[1]))
for i, label in enumerate(visualize_labels):
    idx = 26+int(label) if label.isnumeric() else string.ascii_lowercase.index(label.lower())
    choices = np.random.choice(np.arange(100*idx, 100*(idx+1)), 2, replace=False)
    visualize_x[2*i:2*i+2] = valid_x[choices]

# run visualize_x through your network
# name the output reconstructed_x
##########################
##### your code here #####
##########################
params = pickle.load(open("q5_weights.pickle", "rb"))
h1 = forward(visualize_x, params, "layer1", relu)
h2 = forward(h1, params, "layer2", relu)
h3 = forward(h2, params, "layer3", relu)
reconstructed_x = forward(h3, params, "output", sigmoid)

# visualize
fig = plt.figure()
plt.axis("off")
grid = ImageGrid(fig, 111, nrows_ncols=(len(visualize_labels), 4), axes_pad=0.05)
for i, ax in enumerate(grid):
    if i % 2 == 0:
        ax.imshow(visualize_x[i//2].reshape((32, 32)).T, cmap="Greys")
    else:
        ax.imshow(reconstructed_x[i//2].reshape((32, 32)).T, cmap="Greys")
    ax.set_axis_off()
plt.show()


# Q5.3.2
from skimage.metrics import peak_signal_noise_ratio
# evaluate PSNR
##########################
##### your code here #####
##########################
res = 0

params = pickle.load(open("q5_weights.pickle", "rb"))
h1 = forward(valid_x, params, "layer1", relu)
h2 = forward(h1, params, "layer2", relu)
h3 = forward(h2, params, "layer3", relu)
reconstructed_valid_x = forward(h3, params, "output", sigmoid)

n = valid_x.shape[0]
for i in range(n):
    im_true = valid_x[i]
    im_test = reconstructed_valid_x[i]
    res += peak_signal_noise_ratio(im_true, im_test)
res = res / n
print("PSNR: ", res)
