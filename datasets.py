from os.path import join as PJ
from PIL import Image
from torch.utils.data import Dataset

class MyDataset(Dataset):
    """
        File format: .txt.
        First row: column discription
        Per line(data): image_path,label
        Image REAL path: data_root/image_path
    """
    def __init__(self, data_root, data_file, transform=None):
        self.data_root = data_root
        self.data_file = data_file
        self.transform = transform

        with open(PJ(self.data_root, self.data_file)) as f:
            data = f.readlines()
        self.data = [line.strip().split() for line in data]
        self.data = self.data[1:]    # remove discription row

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        self.image_path, self.class_name = self.data[index]

        label = int(self.class_name)

        image = Image.open(PJ(self.data_root, self.image_path)).convert('RGB')
        image = self.transform(image) if self.transform else image

        return label, image