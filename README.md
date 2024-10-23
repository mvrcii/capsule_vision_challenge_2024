# Capsule Vision Challenge 2024

## Quickstart
1. Create a new virtual environment with Python 3.11 (preferably with venv): `python3.11 -m venv env`
2. Install required packages with `pip install -r requirements.txt`
3. Download the Capsule Vision Challenge 2024 dataset from the [official website](https://capsule.vision/challenge/2024) and store it in the `data/` directory **in the repository root**
4. Download the pre-trained model weights via link or command and store them in the `pretrained_models/` directory **in the repository root**
5. Run `python train.py --config configs/submission/run_eva02_base_patch14_224.ee24_ft_ce24.yaml` to start fine-tuning the pre-trained EndoExtend24 model

## Model Weights
You can find the weights for both the pre-trained model and the fine-tuned downstream task model below. For an easy download on a bash cluster such as a slurm master, you can use the `gdown` command as shown below.

| **Type**    | **Dataset**        | **Checkpoint** | **Command**                                                          |
|-------------|----------------|----------------|----------------------------------------------------------------------------|
| Pre-trained | EndoExtend24    | [eva02_base_patch14_224.pt_ee24](https://drive.google.com/file/d/1Ok58RCRvKdq1_VcFn35FQOHyznvq8JFr/view?usp=sharing)             | `gdown 1Ok58RCRvKdq1_VcFn35FQOHyznvq8JFr` |
| Fine-tuned  | CE24            | [eva02_base_patch14_224.ee24_ft_ce24](https://drive.google.com/file/d/123TjuBw-34bKXBu7njzKjbcObNXsnuEY/view?usp=sharing)             | `gdown 123TjuBw-34bKXBu7njzKjbcObNXsnuEY` |
