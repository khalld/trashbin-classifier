from torchvision import transforms
from os.path import join
from TDContainer import TDContainer

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