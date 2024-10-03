# Anonymization Pipeline

This repository implements a pipeline which aims to replicate the procedure used by Packhäuser _et al._ [\[1\]](#1). This pipeline aims to anonymize a synthetic dataset which has been previously generated by employing both a patient-retrieval model and a patient-verification model.

## Requirements

The experiments were run using Python 3.8.5 and CUDA 11.4, the remaining dependencies can be installed using `pip install -r requirements.txt` inside a virtual environment of your choice.

## Steps

| Step | Script | Output                | Description                                                                                                                                                                                    |
| ---- | ---- | --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    |  `compute_training_embeddings.py`  | Lookup Table          | Using the **retrieval** model, computes embeddings for real images in the training set and stores them in a lookup table.                                                                  |
| 2    | `search_nearest_image.py` | CSV File              | Utilizes KD-Tree for efficient lookup, computes embeddings for synthetic images using the **retrieval** model, and finds closest matches in the training data based on embedding distances. |
| 3    | `verify_images.py` | CSV File              | Compares pairs of images obtained in the previous step using the **verification** model and stores the predictions.                                                                          |
| 4    | `filter_dataset.py` | Filtered Dataset      | Removes non-anonymous images from the synthetic dataset based on a specified threshold.                                                                                                       |

## How to run

You first need to pre-train both the verification and retrieval models. Adjust the data paths in the `config.yaml` file and simply execute all the steps in sequential order.

## License

This project is licensed under an GPL-3.0 License, see [LICENSE](LICENSE) for more details.

## References

<a id="1">\[1\]</a>
K. Packhäuser, L. Folle, F. Thamm and A. Maier, "Generation of Anonymous Chest Radiographs Using Latent Diffusion Models for Training Thoracic Abnormality Classification Systems," 2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI), Cartagena, Colombia, 2023, pp. 1-5, doi: 10.1109/ISBI53787.2023.10230346.
