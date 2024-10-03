import torch
from torchvision import transforms
from models.retrieval_model import get_retrieval_model
from dataset.mimic import MIMIC_CXR_Dataset
from dataset.brax import BRAX_Dataset
from dataset.chexpert import ChexpertSmall_Dataset
import yaml
import os

config_name = os.environ.get("CONFIG_NAME", "config.yaml")

with open(config_name, "r") as f:
    config = yaml.safe_load(f)

dataset_name = config["dataset_name"]
assert dataset_name in ["mimic", "brax", "chexpert"], "Invalid dataset name"

if dataset_name == "mimic":
    dataset = MIMIC_CXR_Dataset(
        image_resize=256,
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
        augment_horizontal_flip=False,
        augment_vertical_flip=False,
        path_root=config["real_dataset_path"],
        split_path=config["real_dataset_splits_path"],
    )
elif dataset_name == "brax":
    dataset = BRAX_Dataset(
         image_resize=(256, 256),
         augment_horizontal_flip=False,
         augment_vertical_flip=False,
         mode="train",
         data_filter = {
             "ViewPosition": "PA",
         },
         path_root = config["real_dataset_path"],
    )
elif dataset_name == "chexpert":
    dataset = ChexpertSmall_Dataset(
        image_resize=(256, 256),
        augment_horizontal_flip=False,
        augment_vertical_flip=False,
        mode="train",
        data_filter = {
            "Frontal/Lateral": "Frontal",
            "AP/PA": "PA",
        },
        path_root = config["real_dataset_path"],
    )

print("Loaded dataset")

retrieval_model = get_retrieval_model(config["retrieval_model"])
retrieval_model.eval()

print("Loaded retrieval model")

dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)

print("Creating lookup table")
lookup_table = {}

for i, batch in enumerate(dataloader):
    img = batch["source"]
    embedding = retrieval_model(img)

    # fill in the lookup table
    for i in range(len(batch["path"])):
        lookup_table[batch["path"][i]] = embedding[i].detach().cpu().numpy()

print(f"Created lookup table for {len(lookup_table.keys())} images")

torch.save(lookup_table, config["lookup_table_output"])