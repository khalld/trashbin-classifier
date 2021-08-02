import numpy as np
import pandas as pd
from libs.TrashbinDataset import TrashbinDataset

def main():
    dst = TrashbinDataset('dataset/all_labels.csv')
    print(dst.__len__())

    means = np.zeros(3)
    stdevs = np.zeros(3)

    for data in dst:
        img = data[0]
        for i in range(3):
            img = np.asarray(img)
            means[i] += img[i, :, :].mean()
            stdevs[i] += img[i, :, :].std()

    means = np.asarray(means) / dst.__len__()
    stdevs = np.asarray(stdevs) / dst.__len__()
    print("{} : normMean = {}".format(type, means))
    print("{} : normstdevs = {}".format(type, stdevs))

main()