import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

class ImageTextDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        self.data = pd.read_csv(csv_file)
        print(f"数据集的行数：{len(self.data)}")
        self.img_dir = img_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        caption = self.data.iloc[idx, 1]
        return image, caption