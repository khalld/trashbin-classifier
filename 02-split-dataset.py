
import random
import numpy as np
import pandas as pd

from libs.utils import split_train_val_test

if __name__ == "__main__":   
    random.seed(1996)
    np.random.seed(1996)

    # load all_labels.txt
    all_labels_txt = np.loadtxt('dataset/all_labels.txt', dtype=str, delimiter=',')

    print('Length of dataset:\n %d, shape %s,\nexample of couple: (filename: %s, class: %s)' %( len(all_labels_txt), all_labels_txt.shape, all_labels_txt[0][0], all_labels_txt[0][1] ) )

    images = []
    labels = []

    for i in range(len(all_labels_txt)):
        # create a str array that contains frames 
        images.append('dataset/images/' + all_labels_txt[i][0])
        # create a int array that contains frames's class
        labels.append(all_labels_txt[i][1])

    print('Tot images %d, tot labels %d' %(len(images), len(labels)))

    # instantiate dataframe
    dataset_df = pd.DataFrame({
        'image': images,
        'label': labels
    })

    training_df, validation_df, test_df = split_train_val_test(dataset=dataset_df)

    # save to csv
    training_df.to_csv('dataset/training.csv', index=None)
    validation_df.to_csv('dataset/validation.csv', index=None)
    test_df.to_csv('dataset/test.csv', index=None)
    dataset_df.to_csv('dataset/all_labels.csv', index=None)

    ids, classes = zip(*{
        0: "empty",
        1: "half",
        2: "full"
    }.items())
    ids = pd.DataFrame({'id':ids, 'class':classes}).set_index('id')
    ids.to_csv('dataset/classes.csv')