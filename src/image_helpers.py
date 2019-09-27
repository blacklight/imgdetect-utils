import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

from .file_helpers import normalize_path
from .utils import partition
from .workers import workers_pool


def expand_images(path):
    img_extensions = {'jpg', 'jpeg', 'png', 'bmp'}
    path = normalize_path(path)
    return [os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f)) and
            f.lower().split('.')[-1] in img_extensions]


def prepare_image(img_file, resize=None, color_convert=None, normalize=True):
    img_file = normalize_path(img_file)
    img = cv2.imread(img_file)

    if resize:
        img = cv2.resize(img, resize)

    if color_convert:
        if isinstance(color_convert, str):
            color_convert = getattr(cv2, color_convert)

        img = cv2.cvtColor(img, color_convert)

    img = np.asarray(img)
    if normalize:
        img = img / 255.0

    return img


def scan_images_dir(images_dir):
    dataset = []
    classes = set()

    for label in os.listdir(images_dir):
        if not os.path.isdir(os.path.join(images_dir, label)):
            continue

        image_files = expand_images(os.path.join(images_dir, label))
        if not image_files:
            continue

        classes.add(label)
        dataset.extend([
            {
                'label': label,
                'file': img,
            }
            for img in image_files
        ])

    classes = sorted([c for c in classes])
    np.random.shuffle(dataset)
    return dataset, classes


def _dataset_file_processor(dataset_file, dataset, classes, resize=None, color_convert=None, normalize=True):
    data = []
    labels = []

    for img in dataset:
        labels.append(classes.index(img['label']))
        img = prepare_image(img['file'], resize=resize, color_convert=color_convert, normalize=normalize)
        data.append(img)

    if data:
        print('Storing dataset vectors to {}'.format(dataset_file))
        np.savez_compressed(dataset_file, data=np.asarray(data),
                            labels=np.asarray(labels, dtype=np.uint8),
                            classes=np.asarray(classes))
        return dataset_file

    print('No directories with valid images found')
    return None


def create_dataset_files(images_dir, datasets_dir, split_size=100, num_threads=1, processor=_dataset_file_processor,
                         resize=None, color_convert=None, normalize=True):
    """
    Create dataset files from images as numpy compressed files

    :param images_dir: Directory where the labelled source images are contained. Each of its subdirectories identifies
        a label name, and image files are stored in these subdirectories.
    :param datasets_dir: Directory where the output dataset files will be stored
    :param split_size: Maximum number of images to be stored per file (default: 100)
    :param num_threads: Specifies the maximum number of threads to be used to process the source images
        (default: 1)
    :param processor: Worker callback for processing a chunk of raw items to a dataset file
        (default: _dataset_file_img_worker)
    :type processor: callable(dataset_file, dataset)
    :param resize: If set, images will be resized to the specified dimensions before being stored to the dataset file
    :type resize: tuple or list of two elements
    :param color_convert: If set, the specified color conversion will be applied to the images before storing them
        in the dataset file
    :type color_convert: cv2 color conversion constant or string that identifies such constant (e.g.
        either cv2.COLOR_BGR2GRAY or 'COLOR_BGR2GRAY')
    :param normalize: If set, then dataset values will be normalized between 0 and 1 (default: True)
    :return: generator - This function yields dataset file names as they are created
    """

    images_dir = normalize_path(images_dir)
    dataset, classes = scan_images_dir(images_dir)
    n_dataset_files = int(len(dataset) / split_size) + (1 if len(dataset) % split_size else 0)
    dataset_file_format = \
        os.path.join(normalize_path(datasets_dir), 'dataset{:0') + str(len(str(n_dataset_files))) + '}.npz'

    print('Processing {} images to {} dataset files. Format: {}'.format(
        len(dataset), n_dataset_files, dataset_file_format))

    workers = workers_pool(num_threads)

    for file_idx, chunk in enumerate(partition(dataset, split_size)):
        dataset_file = dataset_file_format.format(file_idx)
        worker = workers[file_idx % len(workers)]
        worker.queue_task(processor, dataset_file, chunk, classes, resize=resize,
                          color_convert=color_convert, normalize=normalize)

    dataset_files = []

    for worker in workers:
        worker.schedule_stop()
        worker_files = [ret for ret in worker.wait_all_tasks()]
        dataset_files.extend(worker_files)

    return dataset_files


def plot_image_histogram(img):
    plt.figure()
    plt.imshow(img)
    plt.colorbar()
    plt.grid(False)
    plt.show()


def plot_images_grid(images, labels, classes, rows=5, cols=5):
    plt.figure(figsize=(10, 10))
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        # noinspection PyUnresolvedReferences
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(classes[labels[i]])

    plt.show()
