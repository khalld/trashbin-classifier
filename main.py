import numpy as np
from os.path import join
import random
from torchvision import transforms

from libs.TrashbeanDataset import TrashbeanDataset
from libs.TDContainer import TDContainer
from libs.PretrainedModels import AverageValueMeter, PretrainedModelsCreator, CCAlexNet, CCSqueezeNet, CCVgg16

random.seed(1996)
np.random.seed(1996)

def import_dataset():
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224), # crop centrale
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    dst_train = {
        'path': join(GDRIVE_PATHS['dataset'], 'training.csv'),
        'transform': train_transform
        }

    dst_validation = {
        'path': join(GDRIVE_PATHS['dataset'], 'validation.csv'),
        'transform': test_transform
        }

    dst_test = {
        'path': join(GDRIVE_PATHS['dataset'], 'test.csv'),
        'transform': test_transform
        }

    return TDContainer(training=dst_train, validation=dst_validation, test=dst_test)

## credo vada cambiato !! perché lo devo montare dentro il git?
## valuta se conviene
GDRIVE_PATHS = {
    'main': '/content/gdrive/MyDrive/trashbean-classifier/',
    'dataset': '/content/gdrive/MyDrive/trashbean-classifier/dataset/',
    'logs': '/content/gdrive/MyDrive/trashbean-classifier/logs/',
    'models': '/content/gdrive/MyDrive/trashbean-classifier/logs/models/',
    'libs': '/content/gdrive/MyDrive/Colab Notebooks/Progetto/libs'
}

def get_model_name(model_name, version):
    return "%s_v%s" % (model_name, version)

def do_training(creator: PretrainedModelsCreator,
                dataset: TDContainer,
                output_class: int,
                model_name: str,
                batch_size: int,
                num_workers: int,
                drop_last: bool,
                lr,
                epochs,
                momentum,
                train_from_epoch,
                save_on_runtime,
                save_each_iter,
                path: dict) -> None:
    creator.initialize_dst(dataset, output_class, batch_size=batch_size, num_workers=num_workers, drop_last=drop_last)
    creator.train(model_name=model_name, lr=lr, epochs=epochs, momentum=momentum, log_dir=path['logs'], model_dir=path['models'], train_from_epoch=train_from_epoch, save_on_runtime=save_on_runtime, save_each_iter=save_each_iter)

def main():
    trashbean_dataset = import_dataset()
    print("Dataset loaded correctly..")

    print("First try launching training with Vgg16.")
    do_training(creator=CCVgg16(),
                dataset=trashbean_dataset,
                output_class=3,
                model_name=get_model_name("Vgg16", "1"),
                batch_size=32,
                num_workers=2,
                drop_last=False,
                lr=0.01,
                epochs=1,
                momentum=0.99,
                train_from_epoch=0,
                save_on_runtime=False,
                save_each_iter=25,
                path=GDRIVE_PATHS)


if __name__ == "__main__":
    main()