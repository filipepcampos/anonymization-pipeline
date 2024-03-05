import torch
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_metric_learning import losses
from models.retrieval_model import get_retrieval_model
from dataset.mimic import MIMIC_CXR_Dataset

# make training deterministic
pl.seed_everything(3)

anonymous_dataset_path = "/nas-ctm01/homes/fpcampos/dev/diffusion/medfusion/dataset/synthetic_250"


dataset = MIMIC_CXR_Dataset(
    image_resize=256,
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # TODO: I should check these transforms again 
    ]),
    augment_horizontal_flip=False,
    augment_vertical_flip=False,
    path_root = '/nas-ctm01/datasets/public/MEDICAL/MIMIC-CXR',
    split_path = '/nas-ctm01/homes/fpcampos/dev/diffusion/medfusion/data/mimic-cxr-2.0.0-split.csv'
)

retrieval_model = get_retrieval_model()
retrieval_model.eval()

dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)

lookup_table = {}

for i, batch in enumerate(dataloader):
    img = batch["source"]
    embedding = retrieval_model(img)
    
    # fill in the lookup table
    for i in range(len(batch["path"])):
        lookup_table[batch["path"][i]] = embedding[i].detach().cpu().numpy()

print(len(lookup_table.keys()))
torch.save(lookup_table, "outputs/lookup_table.pth")
