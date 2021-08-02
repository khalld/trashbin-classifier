
import random
import numpy as np
import pandas as pd

from PIL import Image
from libs.utils import split_train_val_test

def main():
    random.seed(1996)
    np.random.seed(1996)

    print("\n\n")

    all_labels_txt = np.loadtxt('dataset/all_labels.txt', dtype=str, delimiter=',')

    print('length of dst: %d, shape %s,\nex coppia: filename: %s, class: %s' %( len(all_labels_txt), all_labels_txt.shape, all_labels_txt[0][0], all_labels_txt[0][1] ) )

    images = []
    labels = []

    for i in range(len(all_labels_txt)):
        images.append('dataset/images/' + all_labels_txt[i][0]) # creo un array di label
        labels.append(all_labels_txt[i][1]) # array di classe di appartenenza

    # print(len(images), len(labels))

    dataset_df = pd.DataFrame({
        'image': images,
        'label': labels
    })

    # test corretto caricamento
    # print(dataset_df['label'][0])
    # Image.open(dataset_df['image'][0]).show()

    # stampo i primi 5 elementi del dataFrame
    # print(dataset_df.head())

    training_df, validation_df, test_df = split_train_val_test(dataset=dataset_df)

    # check distribuzione di classi
    print("training", training_df['label'].value_counts() ) 
    print("\n")
    print("validation", validation_df['label'].value_counts() ) 
    print("\n")
    print("test", test_df['label'].value_counts() )
    print("\n")


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

main()