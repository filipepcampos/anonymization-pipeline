import torch.utils.data as data
from torch import nn
from pathlib import Path
from torchvision import transforms as T
import pandas as pd
import math


from PIL import Image


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


class MIMIC_CXR_Dataset(SimpleDataset2D):
    def __init__(self, split_path, split="train", *args, **kwargs):
        super().__init__(*args, **kwargs)

        metadata_labels = pd.read_csv(self.path_root / "mimic-cxr-2.0.0-metadata.csv")
        metadata_labels = metadata_labels.loc[metadata_labels["ViewPosition"] == "PA"]

        chexpert_labels = pd.read_csv(
            self.path_root / "mimic-cxr-2.0.0-chexpert.csv",
            index_col=["subject_id", "study_id"],
        )
        splits = pd.read_csv(split_path)
        labels = metadata_labels.merge(chexpert_labels, on="study_id", how="left")
        labels = labels.dropna(subset=["subject_id"])

        labels = labels.merge(
            splits, on="dicom_id", suffixes=("", "_right"), how="left"
        )

        labels = labels[labels["split"] == split]

        labels["Cardiomegaly"] = labels["Cardiomegaly"].map(
            lambda x: 2 if x < 0 or math.isnan(x) else x
        )
        labels = labels.set_index("dicom_id")

        def get_path(row):
            dicom_id = str(row.name)
            subject = "p" + str(int(row["subject_id"]))
            study = "s" + str(int(row["study_id"]))
            image_file = dicom_id + ".jpg"
            return self.path_root / "files" / subject[:3] / subject / study / image_file

        labels["Path"] = labels.apply(get_path, axis=1)
        labels = labels[labels["Cardiomegaly"] != 2]

        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        dicom_id = self.labels.index[index]
        row = self.labels.loc[dicom_id]

        img = self.load_item(row["Path"])
        return {
            "uid": dicom_id,
            "source": self.transform(img),
            "target": int(row["Cardiomegaly"]),
            "path": str(row["Path"]),
        }

    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        """Overwrite to speed up as paths are determined by .csv file anyway"""
        return []

    def get_weights(self):
        n_samples = len(self)
        weight_per_class = 1 / self.labels["Cardiomegaly"].value_counts(normalize=True)
        weights = [0] * n_samples
        for index in range(n_samples):
            target = self.labels.loc[self.labels.index[index], "Cardiomegaly"]
            weights[index] = weight_per_class[target]
        return weights
