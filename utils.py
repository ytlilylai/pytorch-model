from os.path import join as PJ
import argparse
from torchvision import transforms
from sklearn.metrics import accuracy_score, recall_score
from tensorboardX import SummaryWriter


def parser_args(config_dict):
    parser = argparse.ArgumentParser(description='multilabel classification with transformer')
    args = parser.parse_args()

    for key, value in config_dict.items():
        setattr(args, key, value)

    return args


def data_transform(new_size, train=True):
    transform_list = [transforms.Resize((new_size, new_size)),
                      transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform_list = [transforms.RandomHorizontalFlip(),
                      transforms.RandomVerticalFlip(),
                      transforms.RandomRotation(90)] + transform_list if train else transform_list
    transform = transforms.Compose(transform_list)
    return transform


def cal_acc(results, weighted):

    if weighted:
        # Weighted by instance
        acc = recall_score(results['label'], results['predict'], average="weighted")

    elif weighted == 0.33:
        # Weighted with 0.33
        acc = recall_score(results['label'], results['predict'], average=None)
        acc = (acc[0]+acc[1]+acc[2]) / 3

    return acc


class MyRecoder():
    def __init__(self, save_path, configs):
        self.file = open(PJ(save_path, "log.txt"), "w")
        self.writer = SummaryWriter(save_path)
        self.rec_cogfigs(configs)

    def rec_configs(self, configs):
        self.file.write("==== Configs ====")
        for i in configs:
            self.file.write(": ".join([i, configs[i]]))

    def write(self, label, data, scale):
        self.file.write([label, data, scale])
        self.writer.add_scalar(label, data, scale)

    def close(self):
        self.file.close()
        self.writer.close()