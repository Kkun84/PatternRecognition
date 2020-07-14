import itertools
from os import mkdir
from PIL import Image
from sklearn import metrics
import numpy as np
import os
import hydra
from hydra.utils import to_absolute_path
from logging import getLogger
import pathlib
import torch
import torchvision
import random


logger = getLogger(__name__)


def get_data_path():
    train_list = [(pathlib.Path(f"./Images/TrainingSamples/{c:1d}-{i:04d}.png"), c) for c, i in itertools.product(range(10), range(200))]
    test_list = [(pathlib.Path(f"./Images/TestSamples/{c:1d}-{i:04d}.png"), c) for c, i in itertools.product(range(10), range(1000))]
    logger.info(f"{len(train_list)=}")
    logger.info(f"{len(test_list)=}")
    return train_list, test_list


def get_image_list(data_list):
    image_list = [Image.open(to_absolute_path(p)) for p, _ in data_list]
    logger.info(f"{len(image_list)=}")
    return image_list


def get_class_list(data_list):
    class_list = list(zip(*data_list))[1]
    logger.info(f"{len(class_list)=}")
    return class_list


def augment(image_list, class_list, transforms, aug_n):
    image_list = sum([image_list] + [[transforms(i) for i in image_list] for _ in range(aug_n)], [])
    class_list = class_list * (aug_n + 1)
    logger.info(f"{len(image_list)=}")
    logger.info(f"{len(class_list)=}")
    return image_list, class_list


@hydra.main(config_path='../config/config.yaml')
def main(config):
    print(config)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    logger.info('train_list, test_list')
    train_list, test_list = get_data_path()
    logger.info('train_images, test_images')
    train_images, test_images = [get_image_list(i) for i in [train_list, test_list]]
    logger.info('train_class, test_class')
    train_class, test_class = [get_class_list(i) for i in [train_list, test_list]]

    transforms = torchvision.transforms.Compose([
        hydra.utils.instantiate(i) for i in config.transforms
    ] if config.transforms else [])

    logger.info(f"{transforms=}")
    train_images, train_class = augment(train_images, train_class, transforms, aug_n=config.aug_n)
    train_images, test_images = [np.stack([np.array(i).reshape(-1) for i in image_list]) for image_list in [train_images, test_images]]

    classifier = hydra.utils.instantiate(config.classifier)
    classifier.fit(train_images, train_class)
    predict = classifier.predict(test_images)

    confusion_matrix = metrics.confusion_matrix(test_class, predict)
    confusion_matrix = np.vstack([confusion_matrix, confusion_matrix.sum(0, keepdims=True)])
    confusion_matrix = np.hstack([confusion_matrix, confusion_matrix.sum(1, keepdims=True)])
    logger.info(f"confusion_matrix=\n{confusion_matrix}")
    ac_score = metrics.accuracy_score(test_class, predict)
    logger.info(f"{ac_score=}")
    classification_report = metrics.classification_report(test_class, predict, digits=4)
    logger.info(f"classification_report=\n{classification_report}")

    for (path, class_), p_class in zip(test_list, predict):
        directory = pathlib.Path('./result') / f"{class_}->{p_class}"
        directory.mkdir(exist_ok=True, parents=True)
        os.symlink(pathlib.Path('/home/keisuke/Documents/pr') / path, directory / str(path)[-10:])

    return


if __name__ == "__main__":
    main()
