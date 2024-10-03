import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
import yaml
import os

config_name = os.environ.get("CONFIG_NAME", "config.yaml")

with open(config_name, "r") as f:
    config = yaml.safe_load(f)

df = pd.read_csv(config["verification_output"])

anonymous_images = {}

anonymous_pairs = []
non_anonymous_pairs = []
barely_anonymous_pairs = []

hist = {
    0: 0,
    0.1: 0,
    0.2: 0,
    0.3: 0,
    0.4: 0,
    0.5: 0,
    0.6: 0,
    0.7: 0,
    0.8: 0,
    0.9: 0,
    1: 0,
}

for image in df.synthetic_image:
    anonymous_images[image] = True

# Obtain some examples near different decision boundaries
for row in df.itertuples():
    if row.score > 0.5:
        anonymous_images[row.synthetic_image] = False
        non_anonymous_pairs.append((row.real_image, row.synthetic_image))
    if row.score > 0.4 and row.score < 0.5:
        barely_anonymous_pairs.append((row.real_image, row.synthetic_image))
    if row.score < 0.1:
        anonymous_pairs.append((row.real_image, row.synthetic_image))
    hist[int(row.score * 10) / 10] += 1

print(hist)

print("There are ", len(anonymous_images), " images in the dataset")
print("Anonymous images: ", len([k for k, v in anonymous_images.items() if v]))
print("Non-anonymous images: ", len([k for k, v in anonymous_images.items() if not v]))
print("Barely anonymous images: ", len(barely_anonymous_pairs))

os.makedirs(config["filtered_dataset_output"], exist_ok=True)
os.makedirs(f"{config['filtered_dataset_output']}/Cardiomegaly", exist_ok=True)
os.makedirs(f"{config['filtered_dataset_output']}/No_Cardiomegaly", exist_ok=True)

for k, v in anonymous_images.items():
    split = k.split("/")
    if v:
        os.symlink(k, f"{config['filtered_dataset_output']}/{split[-2]}/{split[-1]}")


def show_random_samples(img_list, output_filename):
    fig, ax = plt.subplots(2, 5, figsize=(20, 10))
    ax[0, 0].set_ylabel("Real")
    ax[1, 0].set_ylabel("Fake")
    i = 0
    for real, fake in img_list[:5]:
        ax[0, i].imshow(cv2.imread(real))
        ax[1, i].imshow(cv2.imread(fake))
        i += 1
    plt.tight_layout()

    plt.savefig(f"outputs/{output_filename}")

    i = 0
    for real, fake in img_list[:5]:
        # Clear the figure
        plt.clf()
        #plt.imshow(cv2.imread(real))
        # Plt show image without axis
        plt.imshow(cv2.imread(real))
        plt.axis("off")
        

        plt.savefig(f"outputs/{output_filename.split('.')[0]}_{i}_real.png")
        plt.clf()
        plt.imshow(cv2.imread(fake))
        plt.axis("off")
        plt.savefig(f"outputs/{output_filename.split('.')[0]}_{i}_fake.png")
        i += 1


show_random_samples(non_anonymous_pairs, "filtered_dataset_non_anonymous_samples.png")
show_random_samples(
    barely_anonymous_pairs,
    "filtered_dataset_barely_anonymous_samples.png",
)
show_random_samples(anonymous_pairs, "filtered_dataset_anonymous_samples.png")

