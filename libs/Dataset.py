from torchvision import transforms
from torchvision.transforms.transforms import RandomGrayscale, RandomHorizontalFlip, RandomInvert
from libs.utils import import_dataset
from torch.nn import ModuleList

# default mean and std needed by pretrained models from pytorch
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

path_dst = 'dataset'
path_gdrive = ''

dst = import_dataset(path_dst=path_dst, 
    train_transform=transforms.Compose([
        transforms.Resize(256),
        # using the same as the test because the trash bin is centered in the image
        transforms.CenterCrop(224), # good for inceptionv3?
        transforms.RandomApply(ModuleList([
            transforms.ColorJitter(brightness=.3, hue=.2),
        ]), p=0.3),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.2),
        transforms.RandomEqualize(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]),
    validation_transform=transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), # good for inceptionv3?
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.RandomInvert(p=0.3),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]),
    test_transform=transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), # good for inceptionv3?
        transforms.AutoAugment(transforms.AutoAugmentPolicy.SVHN),
        transforms.RandomInvert(p=0.3),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]), path_gdrive=path_gdrive)



# dst_inceptionv3 = import_dataset(path_dst=path_dst, 
#     train_transform=transforms.Compose([
#         transforms.Resize(320),
#         # using the same as the test because the trash bin is centered in the image
#         transforms.CenterCrop(299), # good for inceptionv3?
#         transforms.RandomApply(ModuleList([
#             transforms.ColorJitter(brightness=.3, hue=.2),
#         ]), p=0.3),

#         transforms.RandomApply(ModuleList([
#             transforms.Grayscale(num_output_channels=3),
#         ]), p=0.2),

#         transforms.RandomHorizontalFlip(p=0.3),
#         transforms.RandomPerspective(distortion_scale=0.3, p=0.2),
#         transforms.RandomEqualize(p=0.2),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=mean, std=std)
#     ]),
#     test_transform=transforms.Compose([
#         transforms.Resize(320), 
#         transforms.CenterCrop(299), # good for inceptionv3?
#         transforms.AutoAugment(transforms.AutoAugmentPolicy.SVHN),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=mean, std=std)
#     ]), path_gdrive=path_gdrive)