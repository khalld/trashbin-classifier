from os.path import join
from libs.TDContainer import TDContainer
from sklearn.model_selection import train_test_split

def get_model_name(model_name, version):
    return "%s_%s" % (model_name, version)

def import_dataset(path_dst: str, train_transform, test_transform):
    dst_train = {
        'path': join(path_dst, 'training.csv'),
        'transform': train_transform
        }

    dst_validation = {
        'path': join(path_dst, 'validation.csv'),
        'transform': test_transform
        }

    dst_test = {
        'path': join(path_dst, 'test.csv'),
        'transform': test_transform
        }

    return TDContainer(training=dst_train, validation=dst_validation, test=dst_test)

def split_train_val_test(dataset, perc=[0.6, 0.1, 0.3]):
    train, testval = train_test_split(dataset, test_size = perc[1]+perc[2])
    val, test = train_test_split(testval, test_size = perc[2]/(perc[1]+perc[2]))
    return train, val, test