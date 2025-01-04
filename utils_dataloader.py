import os
from torchvision import transforms
import random
from PIL import ImageFilter
from torch.utils.data import DataLoader
import common.vision.datasets as datasets
from hypers import DSET2DATASET


class UConSFDATransform:
    def __init__(self, transform, transform1):
        self.transform = transform
        self.transform_s = transform1

    def __call__(self, x):
        if self.transform is None:
            return x, x
        else:
            q = self.transform(x)
            p = self.transform_s(x)
            return [q, p]


class TwoCropsTransform:
    def __init__(self, transform, transform1):
        self.transform = transform
        self.transform1 = transform
        self.transform_s = transform1

    def __call__(self, x):
        if self.transform is None:
            return x, x
        else:
            q = self.transform(x)
            k = self.transform1(x)
            p = self.transform_s(x)
            return [q, k, p]


class ThreeCropsTransform:
    def __init__(self, transform1, transform2, transform3):
        self.transformW = transform1
        self.transformS = transform2
        self.transform_base = transform3

    def __call__(self, x):
        if self.transformW is None:
            return x, x
        else:
            q = self.transformW(x)
            k = self.transformS(x)
            p = self.transform_base(x)
            return [q, k, p]

        
def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        normalize = Normalize(meanfile="./ilsvrc_2012_mean.npy")
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )


# weak aug
def image_train2(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        normalize = Normalize(meanfile="./ilsvrc_2012_mean.npy")

    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


# strong aug
def image_train_mocov2(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        normalize = Normalize(meanfile="./ilsvrc_2012_mean.npy")

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.8, 0.8, 0.5, 0.2)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomRotation(degrees=[-2, 2]),
            transforms.RandomPosterize(8, p=0.2),
            transforms.RandomEqualize(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        normalize = Normalize(meanfile="./ilsvrc_2012_mean.npy")
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


# (auto)UCon_SFDA data loader
def data_load_UCon_SFDA(args):
    # prepare data
    dset_loaders = {}
    train_bs = args.batch_size

    root_dir = os.path.join(args.root, args.dset)
    dataset = datasets.__dict__[DSET2DATASET[args.dset]]
    
    train_source_dataset = dataset(
        root=root_dir,
        task=args.target,
        r=0,
        download=False,
        list_name=args.list_name,
        transform=UConSFDATransform(
            image_train_mocov2(), image_train()),
    )
    pl_dataset = dataset(
        root=root_dir,
        task=args.target,
        r=0,
        download=False,
        list_name=args.list_name,
        transform=ThreeCropsTransform(
            image_train2(), image_train_mocov2(), image_train()
        ),
    )
    test_dataset = dataset(
        root=root_dir,
        task=args.target,
        r=0,
        download=False,
        list_name=args.list_name,
        transform=image_test(),
    )
    source_dataset = dataset(
        root=root_dir,
        task=args.source,
        r=0,
        download=False,
        list_name=args.list_name,
        transform=image_test(),
    )

    train_source_loader = DataLoader(
        train_source_dataset,
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_bs * 3,
        shuffle=False,
        num_workers=args.num_workers,
    )
    pl_loader = DataLoader(
        pl_dataset,
        batch_size=train_bs * 3,
        shuffle=False,
        num_workers=args.num_workers,
    )

    dset_loaders["target"] = train_source_loader
    dset_loaders["test"] = test_loader
    dset_loaders["pl"] = pl_loader

    source_loader = DataLoader(
        source_dataset,
        batch_size=train_bs * 3,
        shuffle=False,
        num_workers=args.num_workers,
    )
    dset_loaders["source"] = source_loader

    num_classes = train_source_dataset.num_classes
    args.nb_classes = num_classes
    args.nb_samples = len(train_source_dataset)
    print("training samples size: ", args.nb_samples)

    return dset_loaders


# original data loader
def data_load(args):
    # prepare data
    dset_loaders = {}
    train_bs = args.batch_size

    # "/scratch/gxu86/sfda_lln_extend/DATASOURCE/"
    root_dir = os.path.join(args.root, args.dset)
    dataset = datasets.__dict__[DSET2DATASET[args.dset]]
    
    
    train_source_dataset = dataset(
        root=root_dir,
        task=args.target,
        r=0,
        download=False,
        list_name=args.list_name,
        transform=image_train(),
    )
    test_dataset = dataset(
        root=root_dir,
        task=args.target,
        r=0,
        download=False,
        list_name=args.list_name,
        transform=image_test(),
    )
    source_dataset = dataset(
        root=root_dir,
        task=args.source,
        r=0,
        download=False,
        list_name=args.list_name,
        transform=image_test(),
    )
    pl_dataset = dataset(
        root=root_dir,
        task=args.target,
        r=0,
        download=False,
        list_name=args.list_name,
        transform=image_test(),
    )

    train_source_loader = DataLoader(
        train_source_dataset,
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_bs * 3,
        shuffle=False,
        num_workers=args.num_workers,
    )
    pl_loader = DataLoader(
        pl_dataset,
        batch_size=train_bs * 3,
        shuffle=False,
        num_workers=args.num_workers,
    )

    dset_loaders["target"] = train_source_loader
    dset_loaders["test"] = test_loader
    dset_loaders["pl"] = pl_loader

    source_loader = DataLoader(
        source_dataset,
        batch_size=train_bs * 3,
        shuffle=False,
        num_workers=args.num_workers,
    )
    dset_loaders["source"] = source_loader

    num_classes = train_source_dataset.num_classes
    args.nb_classes = num_classes
    args.nb_samples = len(train_source_dataset)
    print("training samples size: ", args.nb_samples)


    return dset_loaders





