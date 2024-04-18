import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
from torch.utils import data
from torch.utils.data import DataLoader

IMG_FOLDER_NAME = "A"
IMG_POST_FOLDER_NAME = "B"
LIST_FOLDER_NAME = "list"
ANNOT_FOLDER_NAME = "label"
label_suffix = ".png"


def get_img_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_FOLDER_NAME, img_name)


def get_label_path(root_dir, img_name):
    return os.path.join(
        root_dir, ANNOT_FOLDER_NAME, img_name.replace(".jpg", label_suffix)
    )


def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=np.str_)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list


def get_img_post_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_POST_FOLDER_NAME, img_name)


def pil_rescale(img, scale, order):
    assert isinstance(img, Image.Image)
    height, width = img.size
    target_size = (int(np.round(height * scale)), int(np.round(width * scale)))
    return pil_resize(img, target_size, order)


def pil_resize(img, size, order):
    assert isinstance(img, Image.Image)
    if size[0] == img.size[0] and size[1] == img.size[1]:
        return img
    if order == 3:
        resample = Image.BICUBIC
    elif order == 0:
        resample = Image.NEAREST
    return img.resize(size[::-1], resample)


def get_random_crop_box(imgsize, cropsize):
    h, w = imgsize
    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return (
        cont_top,
        cont_top + ch,
        cont_left,
        cont_left + cw,
        img_top,
        img_top + ch,
        img_left,
        img_left + cw,
    )


def pil_crop(image, box, cropsize, default_value):
    assert isinstance(image, Image.Image)
    img = np.array(image)

    if len(img.shape) == 3:
        cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype) * default_value
    else:
        cont = np.ones((cropsize, cropsize), img.dtype) * default_value
    cont[box[0] : box[1], box[2] : box[3]] = img[box[4] : box[5], box[6] : box[7]]

    return Image.fromarray(cont)


class CDDataAugmentation:

    def __init__(
        self,
        img_size,
        with_random_hflip=False,
        with_random_vflip=False,
        with_random_rot=False,
        with_random_crop=False,
        with_scale_random_crop=False,
        with_random_blur=False,
        random_color_tf=False,
    ):
        self.img_size = img_size
        if self.img_size is None:
            self.img_size_dynamic = True
        else:
            self.img_size_dynamic = False
        self.with_random_hflip = with_random_hflip
        self.with_random_vflip = with_random_vflip
        self.with_random_rot = with_random_rot
        self.with_random_crop = with_random_crop
        self.with_scale_random_crop = with_scale_random_crop
        self.with_random_blur = with_random_blur
        self.random_color_tf = random_color_tf

    def transform(self, imgs, labels, to_tensor=True):
        """
        :param imgs: [ndarray,]
        :param labels: [ndarray,]
        :return: [ndarray,],[ndarray,]
        """
        # resize image and covert to tensor
        imgs = [TF.to_pil_image(img) for img in imgs]
        if self.img_size is None:
            self.img_size = None

        if not self.img_size_dynamic:
            if imgs[0].size != (self.img_size, self.img_size):
                imgs = [
                    TF.resize(img, [self.img_size, self.img_size], interpolation=3)
                    for img in imgs
                ]
        else:
            self.img_size = imgs[0].size[0]

        labels = [TF.to_pil_image(img) for img in labels]
        if len(labels) != 0:
            if labels[0].size != (self.img_size, self.img_size):
                labels = [
                    TF.resize(img, [self.img_size, self.img_size], interpolation=0)
                    for img in labels
                ]

        random_base = 0.5
        if self.with_random_hflip and random.random() > 0.5:
            imgs = [TF.hflip(img) for img in imgs]
            labels = [TF.hflip(img) for img in labels]

        if self.with_random_vflip and random.random() > 0.5:
            imgs = [TF.vflip(img) for img in imgs]
            labels = [TF.vflip(img) for img in labels]

        if self.with_random_rot and random.random() > random_base:
            angles = [90, 180, 270]
            index = random.randint(0, 2)
            angle = angles[index]
            imgs = [TF.rotate(img, angle) for img in imgs]
            labels = [TF.rotate(img, angle) for img in labels]

        if self.with_random_crop and random.random() > 0:
            i, j, h, w = transforms.RandomResizedCrop(size=self.img_size).get_params(
                img=imgs[0], scale=(0.8, 1.2), ratio=(1, 1)
            )

            imgs = [
                TF.resized_crop(
                    img,
                    i,
                    j,
                    h,
                    w,
                    size=(self.img_size, self.img_size),
                    interpolation=Image.CUBIC,
                )
                for img in imgs
            ]

            labels = [
                TF.resized_crop(
                    img,
                    i,
                    j,
                    h,
                    w,
                    size=(self.img_size, self.img_size),
                    interpolation=Image.NEAREST,
                )
                for img in labels
            ]

        if self.with_scale_random_crop:

            scale_range = [1, 1.2]
            target_scale = scale_range[0] + random.random() * (
                scale_range[1] - scale_range[0]
            )

            imgs = [pil_rescale(img, target_scale, order=3) for img in imgs]
            labels = [pil_rescale(img, target_scale, order=0) for img in labels]

            imgsize = imgs[0].size
            box = get_random_crop_box(imgsize=imgsize, cropsize=self.img_size)
            imgs = [
                pil_crop(img, box, cropsize=self.img_size, default_value=0)
                for img in imgs
            ]
            labels = [
                pil_crop(img, box, cropsize=self.img_size, default_value=255)
                for img in labels
            ]

        if self.with_random_blur and random.random() > 0:
            radius = random.random()
            imgs = [img.filter(ImageFilter.GaussianBlur(radius=radius)) for img in imgs]

        if self.random_color_tf:
            color_jitter = transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3
            )
            imgs_tf = []
            for img in imgs:
                tf = transforms.ColorJitter(
                    color_jitter.brightness,
                    color_jitter.contrast,
                    color_jitter.saturation,
                    color_jitter.hue,
                )
                imgs_tf.append(tf(img))
            imgs = imgs_tf

        if to_tensor:
            imgs = [TF.to_tensor(img) for img in imgs]
            labels = [
                torch.from_numpy(np.array(img, np.uint8)).unsqueeze(dim=0)
                for img in labels
            ]

            imgs = [
                # TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                for img in imgs
            ]

        return imgs, labels


class ImageDataset(data.Dataset):
    """VOCdataloder"""

    def __init__(
        self, root_dir, split="train", img_size=256, is_train=True, to_tensor=True
    ):
        super(ImageDataset, self).__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split
        self.list_path = os.path.join(
            self.root_dir, LIST_FOLDER_NAME, self.split + ".txt"
        )
        self.img_name_list = load_img_name_list(self.list_path)

        self.A_size = len(self.img_name_list)
        self.to_tensor = to_tensor
        if is_train:
            self.augm = CDDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_scale_random_crop=True,
                with_random_blur=True,
                random_color_tf=True,
            )
        else:
            self.augm = CDDataAugmentation(img_size=self.img_size)

    def __getitem__(self, index):
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, self.img_name_list[index % self.A_size])
        B_path = get_img_post_path(
            self.root_dir, self.img_name_list[index % self.A_size]
        )

        img = np.asarray(Image.open(A_path).convert("RGB"))
        img_B = np.asarray(Image.open(B_path).convert("RGB"))

        [img, img_B], _ = self.augm.transform(
            [img, img_B], [], to_tensor=self.to_tensor
        )

        return {"A": img, "B": img_B, "name": name}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.A_size


class CDDataset(ImageDataset):

    def __init__(
        self,
        root_dir,
        img_size,
        split="train",
        is_train=True,
        label_transform=None,
        to_tensor=True,
    ):
        super(CDDataset, self).__init__(
            root_dir,
            img_size=img_size,
            split=split,
            is_train=is_train,
            to_tensor=to_tensor,
        )
        self.label_transform = label_transform

    def __getitem__(self, index):
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, self.img_name_list[index % self.A_size])
        B_path = get_img_post_path(
            self.root_dir, self.img_name_list[index % self.A_size]
        )
        img = np.asarray(Image.open(A_path).convert("RGB"))
        img_B = np.asarray(Image.open(B_path).convert("RGB"))
        L_path = get_label_path(self.root_dir, self.img_name_list[index % self.A_size])

        label = np.array(Image.open(L_path), dtype=np.uint8)

        if self.label_transform == "norm":
            label = label // 255

        [img, img_B], [label] = self.augm.transform(
            [img, img_B], [label], to_tensor=self.to_tensor
        )

        return {"pixel_valuesA": img, "pixel_valuesB": img_B, "labels": label.squeeze()}
        # return {"name": name, "A": img, "B": img_B, "L": label.squeeze()}


def give_dataloader(config):
    root_dir = config.dataset.root_dir
    label_transform = "norm"
    split = "train"
    split_val = config.dataset.split_val
    training_set = CDDataset(
        root_dir=root_dir,
        split=split,
        img_size=config.dataset.image_size,
        is_train=True,
        label_transform=label_transform,
    )
    val_set = CDDataset(
        root_dir=root_dir,
        split=split_val,
        img_size=config.dataset.image_size,
        is_train=False,
        label_transform=label_transform,
    )

    datasets = {"train": training_set, "val": val_set}
    dataloaders = {
        x: DataLoader(
            datasets[x],
            batch_size=config.dataset.batch_size,
            shuffle=True,
            num_workers=config.dataset.num_workers,
        )
        for x in ["train", "val"]
    }

    return dataloaders
