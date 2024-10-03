from PIL import Image
import os
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
from torch import nn
from pathlib import Path
from torchvision import transforms as T

class SimpleDataset2D(data.Dataset):
    def __init__(
        self,
        path_root,
        item_pointers=[],
        crawler_ext="tif",  # other options are ['jpg', 'jpeg', 'png', 'tiff'],
        transform=None,
        image_resize=None,
        augment_horizontal_flip=False,
        augment_vertical_flip=False,
        image_crop=None,
    ):
        super().__init__()
        self.path_root = Path(path_root)
        self.crawler_ext = crawler_ext
        if len(item_pointers):
            self.item_pointers = item_pointers
        else:
            self.item_pointers = self.run_item_crawler(self.path_root, self.crawler_ext)

        if transform is None:
            self.transform = T.Compose(
                [
                    (
                        T.Resize(image_resize)
                        if image_resize is not None
                        else nn.Identity()
                    ),
                    (
                        T.RandomHorizontalFlip()
                        if augment_horizontal_flip
                        else nn.Identity()
                    ),
                    T.RandomVerticalFlip() if augment_vertical_flip else nn.Identity(),
                    (
                        T.CenterCrop(image_crop)
                        if image_crop is not None
                        else nn.Identity()
                    ),
                    T.ToTensor(),
                    T.Normalize(
                        mean=0.5, std=0.5
                    ),  # WARNING: mean and std are not the target values but rather the values to subtract and divide by: [0, 1] -> [0-0.5, 1-0.5]/0.5 -> [-1, 1]
                ]
            )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.item_pointers)

    def __getitem__(self, index):
        rel_path_item = self.item_pointers[index]
        path_item = self.path_root / rel_path_item
        # img = Image.open(path_item)
        img = self.load_item(path_item)
        return {"uid": rel_path_item.stem, "source": self.transform(img)}

    def load_item(self, path_item):
        return Image.open(path_item).convert("RGB")

    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        return [
            path.relative_to(path_root)
            for path in Path(path_root).rglob(f"*.{extension}")
        ]

    def get_weights(self):
        """Return list of class-weights for WeightedSampling"""
        return None
    
class ChexpertSmall_Dataset(SimpleDataset2D):
    attr_all_names = [
        "No Finding",
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "Support Devices",
    ]
    attr_names = ["Cardiomegaly"]

    def __init__(self, mode="train", data_filter=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_root = os.path.expanduser(self.path_root)
        assert mode in ["train", "valid", "test"]
        self.mode = mode

        # if mode is test; root is path to csv file (in test mode), construct dataset from this csv;
        # if mode is train/valid; root is path to data folder with `train`/`valid` csv file to construct dataset.
        test_df, train_df = self._maybe_process(data_filter)

        if mode == "test":
            self.data = test_df
        else:
            valid_df = train_df.sample(frac=0.2, random_state=0)
            train_df = train_df.drop(valid_df.index)
            self.data = valid_df if mode == "valid" else train_df

        # store index of the selected attributes in the columns of the data for faster indexing
        self.attr_idxs = [self.data.columns.tolist().index(a) for a in self.attr_names]

    def __getitem__(self, idx):
        # 1. select and load image
        img_path = self.data.iloc[idx, 0]  # 'Path' column is 0
        full_img_path = os.path.join(self.path_root, img_path)
        img = Image.open(os.path.join(full_img_path).convert("RGB"))
        if self.transform is not None:
            img = self.transform(img)

        # 2. select attributes as targets
        attr = self.data.iloc[idx, self.attr_idxs].values.astype(np.float32)
        attr = torch.from_numpy(attr).item()

        # 3. save index for extracting the patient_id in prediction/eval results as 'CheXpert-v1.0-small/valid/patient64541/study1'
        #    performed using the extract_patient_ids function
        idx = self.data.index[
            idx
        ]  # idx is based on len(self.data); if we are taking a subset of the data, idx will be relative to len(subset);
        # self.data.index(idx) pulls the index in the original dataframe and not the subset

        return {"source": img, "target": int(attr), "path": str(full_img_path), "uid": idx}

    def __len__(self):
        return len(self.data)

    def _maybe_process(self, data_filter):
        # Dataset labels are: blank for unmentioned, 0 for negative, -1 for uncertain, and 1 for positive.
        # Process by:
        #    1. fill NAs (blanks for unmentioned) as 0 (negatives)
        #    2. fill -1 as 1 (U-Ones method described in paper)  # TODO -- setup options for uncertain labels
        #    3. apply attr filters as a dictionary {data_attribute: value_to_keep} e.g. {'Frontal/Lateral': 'Frontal'}

        test_df = pd.read_csv(
            os.path.join(self.path_root, "CheXpert-v1.0-small", "valid.csv"),
            keep_default_na=True,
        )
        train_df = self._load_and_preprocess_training_data(
            os.path.join(self.path_root, "CheXpert-v1.0-small", "train.csv"),
            data_filter,
        )

        return test_df, train_df

    def _load_and_preprocess_training_data(self, csv_path, data_filter):
        train_df = pd.read_csv(csv_path, keep_default_na=True)

        # 1. fill NAs (blanks for unmentioned) as 0 (negatives)
        # attr columns ['No Finding', ..., 'Support Devices']; note AP/PA remains with NAs for Lateral pictures
        train_df[self.attr_names] = train_df[self.attr_names].fillna(0)
        train_df.dropna(subset=self.attr_names)

        # 2. fill -1 as 1 (U-Ones method described in paper)  # TODO -- setup options for uncertain labels
        train_df[self.attr_names] = train_df[self.attr_names].replace(-1, 1)

        if data_filter is not None:
            # 3. apply attr filters
            # only keep data matching the attribute e.g. df['Frontal/Lateral']=='Frontal'
            for k, v in data_filter.items():
                train_df = train_df[train_df[k] == v]

        return train_df
    
    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        """Overwrite to speed up as paths are determined by .csv file anyway"""
        return []
