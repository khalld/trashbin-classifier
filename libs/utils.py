from torchvision import transforms
from os.path import join
from TDContainer import TDContainer

def get_model_name(model_name, version):
    return "%s_%s" % (model_name, version)

def import_dataset(path_dst: str):
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # deviazione standard dei modelli
        ])

    test_transform = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), # crop centrale
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # deviazione standard dei modelli
        ])

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