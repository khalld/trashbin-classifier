## test per file manager
from os import path
import numpy as np

# classes rispettando l'ordine rispettivamente
# coast
# forest
# highway
# insidecity
# mountain
# opencountry
# street
# tallbuilding

# in test.txt e train.txt
# insidecity_art165.jpg, 3
# street_urb382.jpg, 6
# opencountry_land666.jpg, 5

def main():

    class_dict = {
        "empty": -1,
        "half": 0,
        "full": 1
    }

    for key in class_dict:
        print(key, '-->', class_dict[key])

    path = "static/datasets/"

    # read 
    reader = np.loadtxt(path + 'test.txt', dtype=str) # , delimiter=','
    print(reader)

    # write
    test_dict = {
        "test_file1": 0,
        "test_file2": -1,
        "test_file4": 1,
        "test_fil9": 0
    }

    writer = open(path + 'train.txt', 'w')

    for key in test_dict:
        writer.write(key + ", " + str(test_dict[key]))
        writer.write("\n")

    writer.close()

    reader2 = np.loadtxt(path + 'train.txt', dtype=str)
    print(reader2)


main()

