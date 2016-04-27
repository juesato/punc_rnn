import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("directory", help="directory which contains model files. Files should be formatted as *_epoch[epoch_num]_[val_loss]_[training_loss].t7")
args = parser.parse_args()

all_losses = []

for f in os.listdir(args.directory):
    if f[-3:] == ".t7":
        fname = f[:-3]
        fparts = fname.split('_')
        train_loss = float(fparts[-1])
        val_loss = float(fparts[-2])
        epoch = float(fparts[-3][5:]) # remove "epoch" in beginning
        all_losses.append((epoch, val_loss, train_loss))

print (all_losses)

import matplotlib.pyplot as plt
plt.plot([x[0] for x in all_losses], [x[1] for x in all_losses], 'ro')
plt.plot([x[0] for x in all_losses], [x[2] for x in all_losses], 'bs')
plt.show()
