from os.path import join
from numpy.core.records import array
from pandas.core.frame import DataFrame

from torchvision import transforms
from libs.TDContainer import TDContainer
from sklearn.model_selection import train_test_split
from libs.PretrainedModels import PretrainedModelsCreator

def get_model_name(model_name: str, lr: str):
    """Return a string that contain a model name and larning ratee"""
    return "%s__lr=%s" % (model_name, lr)

def import_dataset(path_dst: str, train_transform: transforms, test_transform: transforms, path_gdrive: str=''):
    dst_train = {
        'path': join(path_dst, 'training.csv'),
        'transform': train_transform
        }

    dst_validation = {
        'path': join(path_dst, 'validation.csv'),
        'transform': train_transform
        }

    dst_test = {
        'path': join(path_dst, 'test.csv'),
        'transform': test_transform
        }

    return TDContainer(training=dst_train, validation=dst_validation, test=dst_test, path_gdrive=path_gdrive)

def split_train_val_test(dataset: DataFrame, perc: array):
    train, testval = train_test_split(dataset, test_size = perc[1]+perc[2])
    val, test = train_test_split(testval, test_size = perc[2]/(perc[1]+perc[2]))
    return train, val, test

def reverse_norm(image):
    """Allow to show a normalized image"""
    
    image = image-image.min()
    return image/image.max()

def init_model(creator: PretrainedModelsCreator, model_name: str, num_classes: int = 3, feature_extract: bool=True, use_pretrained: bool = True):
    print('Initializing: %s' % (model_name))
    creator.init_model(num_classes=num_classes, model_name=model_name, feature_extract=feature_extract, use_pretrained=use_pretrained)
    # creator.get_info()
    return creator