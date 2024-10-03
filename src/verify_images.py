import torch
from models.verification_model import SiameseNetwork
from dataset.verification_dataset import VerificationDataset
import yaml
import os

config_name = os.environ.get("CONFIG_NAME", "config.yaml")

with open(config_name, "r") as f:
    config = yaml.safe_load(f)

dataset = VerificationDataset(config["nearest_images_output"], max_rank=0)

verification_model = SiameseNetwork()
state_dict = torch.load(config["verification_model"])
verification_model.load_state_dict(state_dict)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)


with open(config["verification_output"], "w") as f:
    f.write("synthetic_image,real_image,score\n")
    for synthetic_image, real_image, data in dataloader:
        output = verification_model(synthetic_image, real_image)
        output = torch.sigmoid(output).detach().cpu().numpy()

        for i in range(len(output)):
            f.write(
                f'{data["synthetic_image_path"][i]},{data["real_image_path"][i]},{output[i].item()}\n'
            )
