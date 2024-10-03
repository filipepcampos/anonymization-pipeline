import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
import pytorch_lightning as pl
import numpy as np
import time
from models.retrieval_model import get_retrieval_model
from sklearn.neighbors import KDTree
import yaml
import os

# make training deterministic
pl.seed_everything(3)

config_name = os.environ.get("CONFIG_NAME", "config.yaml")

with open(config_name, "r") as f:
    config = yaml.safe_load(f)

print("Loading dataset")

dataset = ImageFolder(
    config["anonymous_dataset_path"],
    transform=transforms.Compose(
        [
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
)

print("Loading lookup table")
lookup_table = torch.load(config["lookup_table_output"])

print("Loading model")
retrieval_model = get_retrieval_model(config["retrieval_model"])
retrieval_model.eval()

path_array = np.array(list(lookup_table.keys()))
embeddings_array = np.array(list(lookup_table.values()))
tree = KDTree(embeddings_array)

N_IMAGES = 3  # Number of real images which should be retrieved for each synthetic image

print("Computing nearest images")

BATCH_SIZE = 128
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

start = time.time()
with open(config["nearest_images_output"], "w") as f:
    f.write("synthetic_image,real_image,distance,rank\n")
    for batch_idx, (img, label) in enumerate(dataloader):
        embeddings = retrieval_model(img)
        embeddings = embeddings.detach().cpu().numpy()
        distances, nearest_idx = tree.query(embeddings, k=N_IMAGES)

        for b in range(img.size(0)):
            for j in range(N_IMAGES):
                path = path_array[nearest_idx[b, j]]
                f.write(
                    f"{dataset.imgs[BATCH_SIZE*batch_idx + b][0]},{path},{distances[b, j]},{j}\n"
                )
end = time.time()
print(f"Elapsed time: {end - start}s")
