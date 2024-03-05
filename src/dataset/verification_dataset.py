import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class VerificationDataset(Dataset):
    def __init__(self, filename, transform=None, max_rank=None):
        self.df = pd.read_csv(filename)

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]) # TODO: Check these transforms again
        else:
            self.transform = transform

        if max_rank is not None:
            self.df = self.df[self.df["rank"] <= max_rank]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        synthetic_image = self.transform(Image.open(row["synthetic_image"]).convert('RGB'))
        real_image = self.transform(Image.open(row["real_image"]).convert('RGB'))
        data = {
            'distance': row['distance'],
            'rank': row['rank'],
            'synthetic_image_path': row['synthetic_image'],
            'real_image_path': row['real_image']
        }
        return synthetic_image, real_image, data