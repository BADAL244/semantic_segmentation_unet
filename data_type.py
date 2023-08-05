import copy
import os
import sys
import time
from os import listdir
from os.path import isfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.models.segmentation import FCN_ResNet101_Weights, DeepLabV3_ResNet101_Weights, \
    DeepLabV3_MobileNet_V3_Large_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torchvision.models as models
import segmentation_models_pytorch as smp


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("\nSegmentation project running on", device)

# training parameters
train = True
in_size = (480, 480)
b_size = 1

lr = 1e-4
nb_epoch = 1

# model selection
model_choice = "dlab_large"
ft = True
appendix = "_ft" if ft else ""

if model_choice not in ["dlab", "dlab_large", "fcn"]:
    print("Error (wrong choice) : choose between dlab, dlab_large, or fcn")
    sys.exit(1)

import segmentation_models_pytorch as smp

class SegmentationModel:
    def __init__(self, encoder='resnet101', encoder_weights='imagenet', activation='softmax', num_classes=2):
        self.encoder = encoder
        self.encoder_weights = encoder_weights
        self.activation = activation
        self.num_classes = num_classes
        self.model = smp.FPN(
            encoder_name=self.encoder,
            encoder_weights=self.encoder_weights,
            classes=self.num_classes,
            activation=self.activation
        )

    def get_model(self):
        return self.model


class CocoDataset(Dataset):
    def __init__(self, root, subset, transform=None, sup=False):
        print(f"\nLoading {subset} dataset")

        self.imgs_dir = os.path.join(root + f"/{subset}2017/")

        ann_file = os.path.join("/home/badal/semantic_segmentation/data/annotations_trainval2017/annotations/", f"instances_{subset}2017.json")
        self.coco = COCO(ann_file)

        self.sup = sup
        self.classes = self.coco.loadCats(self.coco.getCatIds())

        self.class_names = [cat['name'] for cat in self.classes]
        self.superclasses = list(set([cat['supercategory'] for cat in self.classes]))
        # print(len(self.superclasses))

        self.target_classes = self.superclasses if self.sup else self.classes

        self.target_classes_nb = len(self.target_classes) + 1

        self.img_ids = self.coco.getImgIds()

        self.transform = transform

    def assign_class(self, normal_class, attrname):
        for c in self.classes:
            if c['id'] == normal_class:
                return c[attrname]

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        anns = self.coco.loadAnns(self.coco.getAnnIds(img_id))
        img_obj = self.coco.loadImgs(img_id)[0]

        img = Image.open(os.path.join(self.imgs_dir, img_obj['file_name'])).convert('RGB')

        mask = np.zeros(img.size[::-1], dtype=np.uint8)

        for ann in anns:
            class_name = self.assign_class(ann['category_id'], 'name')
            pixel_value = self.class_names.index(class_name) + 1
            mask = np.maximum(self.coco.annToMask(ann) * pixel_value, mask)

        if self.sup:
            for cl in self.classes:
                idx = mask == cl['id']
                class_index = self.assign_class(cl['id'], 'supercategory')
                mask[idx] = self.superclasses.index(class_index) + 1

            idx = mask >= self.target_classes_nb
            mask[idx] = 0

        mask = Image.fromarray(mask)
        # print(mask.shape)

        if self.transform is not None:
            img = self.transform(img)
            img = T.ToTensor()(img)
            img = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

            mask = self.transform(mask)
            mask = T.PILToTensor()(mask)

        return img, mask.long()

    def __len__(self):
        return len(self.img_ids)
    
class CocoTestDataset(Dataset):
    def __init__(self, root, subset, transform=None):
        print(f"\nLoading {subset} dataset")

        self.imgs_dir = os.path.join(root + "/test2017/")
        self.img_names = [f for f in listdir(self.imgs_dir) if isfile(os.path.join(self.imgs_dir, f))]
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.imgs_dir, self.img_names[idx])).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.img_names)
    

def get_data(input_size, batch_size=64, sup=False):
    data_transforms = {
        'train': T.Compose([
            T.Resize(input_size, interpolation=F.InterpolationMode.BILINEAR),
            T.CenterCrop(input_size)
        ]),
        'val': T.Compose([
            T.Resize(input_size, interpolation=F.InterpolationMode.BILINEAR),
            T.CenterCrop(input_size),
        ]),
        'test': T.Compose([
            T.Resize(input_size, interpolation=F.InterpolationMode.BILINEAR),
            T.CenterCrop(input_size),
            T.ToTensor()
        ]),
    }

    coco_train = CocoDataset(root="/home/badal/semantic_segmentation/data/images/", subset="train", transform=data_transforms["train"], sup=True)
    sub_train = torch.utils.data.Subset(coco_train, range(0, 200))
    train_dl = DataLoader(sub_train, batch_size=batch_size, shuffle=True)

    coco_val = CocoDataset(root="/home/badal/semantic_segmentation/data/images/", subset="val", transform=data_transforms["val"], sup=True)
    sub_val = torch.utils.data.Subset(coco_val, range(0, 50))
    val_dl = DataLoader(sub_val, batch_size=batch_size, shuffle=True)

    coco_test = CocoTestDataset(root="/home/badal/semantic_segmentation/data/images/", subset="test", transform=data_transforms["test"])
    sub_test = torch.utils.data.Subset(coco_test, range(0, 150))
    test_dl = DataLoader(sub_test, batch_size=None, shuffle=True)

    cats = ['unlabeled'] + coco_train.target_classes
    # print(cats)

    return train_dl, val_dl, test_dl, cats


def decode_segmap(image, colormap, nc):
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = colormap[l][0]
        g[idx] = colormap[l][1]
        b[idx] = colormap[l][2]

    rgb = np.stack([r, g, b], axis=2)

    return rgb


def image_overlay(image, segmented_image):
    alpha = 1  # transparency for the original image
    beta = 0.75  # transparency for the segmentation map
    gamma = 0  # scalar added to each sum

    image = np.array(image)

    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def detect_classes(img, cats, nb_class):
    detected = []
    for lp in range(0, nb_class):
        idx = img == lp

        if idx.any():
            detected.append(lp)

    return [cats[cnb] for cnb in detected]


def segment_map(output, img, colormap, cats, nb_class):
    om = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()

    cnames = detect_classes(om, cats, nb_class)

    segmented_image = decode_segmap(om, colormap, nb_class)

    # Resize to original image size
    segmented_image = cv2.resize(segmented_image, om.shape, cv2.INTER_CUBIC)

    np_img = np.array(img * 255, dtype=np.uint8)

    overlayed_image = image_overlay(np_img, segmented_image)

    return segmented_image, overlayed_image, cnames


import re
import torch.nn as nn

import torch


def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def iou(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    intersection = torch.sum(gt * pr)
    union = torch.sum(gt) + torch.sum(pr) - intersection + eps
    return (intersection + eps) / union


jaccard = iou


def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate F-score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta ** 2) * tp + eps) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

    return score


def accuracy(pr, gt, threshold=0.5, ignore_channels=None):
    """Calculate accuracy score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt == pr, dtype=pr.dtype)
    score = tp / gt.view(-1).shape[0]
    return score


def precision(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate precision score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp

    score = (tp + eps) / (tp + fp + eps)

    return score


def recall(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate Recall between ground truth and prediction
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: recall score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fn = torch.sum(gt) - tp

    score = (tp + eps) / (tp + fn + eps)

    return score


class BaseObject(nn.Module):
    def __init__(self, name=None):
        super().__init__()
        self._name = name

    @property
    def __name__(self):
        if self._name is None:
            name = self.__class__.__name__
            s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
            return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
        else:
            return self._name


class Metric(BaseObject):
    pass


class Loss(BaseObject):
    def __add__(self, other):
        if isinstance(other, Loss):
            return SumOfLosses(self, other)
        else:
            raise ValueError("Loss should be inherited from `Loss` class")

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, value):
        if isinstance(value, (int, float)):
            return MultipliedLoss(self, value)
        else:
            raise ValueError("Loss should be inherited from `BaseLoss` class")

    def __rmul__(self, other):
        return self.__mul__(other)


class SumOfLosses(Loss):
    def __init__(self, l1, l2):
        name = "{} + {}".format(l1.__name__, l2.__name__)
        super().__init__(name=name)
        self.l1 = l1
        self.l2 = l2

    def __call__(self, *inputs):
        return self.l1.forward(*inputs) + self.l2.forward(*inputs)


class MultipliedLoss(Loss):
    def __init__(self, loss, multiplier):

        # resolve name
        if len(loss.__name__.split("+")) > 1:
            name = "{} * ({})".format(multiplier, loss.__name__)
        else:
            name = "{} * {}".format(multiplier, loss.__name__)
        super().__init__(name=name)
        self.loss = loss
        self.multiplier = multiplier

    def __call__(self, *inputs):
        return self.multiplier * self.loss.forward(*inputs)
    
class Activation(nn.Module):

    def __init__(self, name, **params):

        super().__init__()

        if name is None or name == 'identity':
            self.activation = nn.Identity(**params)
        elif name == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif name == 'softmax2d':
            self.activation = nn.Softmax(dim=1, **params)
        elif name == 'softmax':
            self.activation = nn.Softmax(**params)
        elif name == 'logsoftmax':
            self.activation = nn.LogSoftmax(**params)
        elif name == 'argmax':
            self.activation = ArgMax(**params)
        elif name == 'argmax2d':
            self.activation = ArgMax(dim=1, **params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError('Activation should be callable/sigmoid/softmax/logsoftmax/None; got {}'.format(name))

    def forward(self, x):
        return self.activation(x)

import torch.nn as nn

class JaccardLoss(Loss):
    def __init__(self, eps=1.0, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - jaccard(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class DiceLoss(Loss):
    def __init__(self, eps=1.0, beta=1.0, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - f_score(
            y_pr,
            y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )
    
class IoU(Metric):
    __name__ = "iou_score"

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return iou(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Fscore(Metric):
    def __init__(self, beta=1, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return f_score(
            y_pr,
            y_gt,
            eps=self.eps,
            beta=self.beta,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Accuracy(Metric):
    def __init__(self, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return accuracy(
            y_pr,
            y_gt,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Recall(Metric):
    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return recall(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Precision(Metric):
    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return precision(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )

import numpy as np


class Meter(object):
    """Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    """

    def reset(self):
        """Reset the meter to default settings."""
        pass

    def add(self, value):
        """Log a new value to the meter
        Args:
            value: Next result to include.
        """
        pass

    def value(self):
        """Get the value of the meter in the current state."""
        pass


class AverageValueMeter(Meter):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan

import sys
import torch
from tqdm import tqdm as tqdm

class Epoch:
    def __init__(self, model, loss, metrics, stage_name, device="cpu", verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(
            dataloader,
            desc=self.stage_name,
            file=sys.stdout,
            disable=not (self.verbose),
        ) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):
    def __init__(self, model, loss, metrics, optimizer, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="train",
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        return loss, prediction


class ValidEpoch(Epoch):
    def __init__(self, model, loss, metrics, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="valid",
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        return loss, prediction

def main():
    train_ds, val_ds, test_ds, cats = get_data(input_size=in_size, batch_size=b_size, sup=False)
    nb_classes = len(cats)
    model = SegmentationModel(num_classes=nb_classes)
    model_ = model.get_model().to(device)
    # print(model_)
    #Define Optimization algorithm with Learning rate
    optimizer = torch.optim.Adam([ 
    dict(params=model_.parameters(), lr=0.0001),
    ])
    # from segmentation_models_pytorch.losses import DiceLoss
    # #Define Loss Function
    loss = DiceLoss()
    metrics = [
    IoU(threshold=0.5),
    Accuracy(threshold=0.5),
    Fscore(threshold=0.5),
    Recall(threshold=0.5),
    Precision(threshold=0.5),
    ]   

    train_epoch = TrainEpoch(
    model_, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=device,
    verbose=True,
    )

    valid_epoch = ValidEpoch(
    model_, 
    loss=loss, 
    metrics=metrics, 
    device=device,
    verbose=True,
    )

    max_score = 0

    for i in range(0, 5):
        
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_ds)
        valid_logs = valid_epoch.run(val_ds)
        
        # Save the model with best iou score
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model_, "/home/badal/semantic_segmentation/pytorch-segmentation/train_model/UnetPlus.pth")
            print('Model saved!')
            
        if i == 2:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')


if __name__ == '__main__':
    main()
    






