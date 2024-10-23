<div style="text-align: center; margin-bottom: 2rem; font-family: 'Times New Roman', Times, serif; font-size: 17px">
   <h1>
      Domain-Adaptive Pre-training of Self-Supervised Foundation Models for Medical Image Classification in Gastrointestinal Endoscopy
   </h1>
</div>


-------------------


<div style="text-align: center; font-family: 'Times New Roman', Times, serif; font-size: 20px">

<b>Authors:</b> Marcel Roth, Micha V. Nowak, Dr. Adrian Krenzer, Prof. Dr. Frank Puppe

</div>

<div style="display: flex; justify-content: center; font-family: 'Courier New', Courier, monospace; font-size: 16px; max-width: 800px; margin: auto;">
  <pre>
@article{roth2024domain,
  title={Domain-Adaptive Pre-training of Self-Supervised Foundation Models for Medical Image Classification in Gastrointestinal Endoscopy},
  author={Marcel Roth, Micha V. Nowak, Dr. Adrian Krenzer, Prof. Dr. Frank Puppe},
  journal={arXiv preprint arXiv:XXXXXXX},
  year={2024}
}
  </pre>
</div>


<p style="text-align: justify; max-width: 800px; margin: auto; font-family: 'Times New Roman', Times, serif; font-size: 24px; line-height: 30px">
Video capsule endoscopy (VCE) has revolutionized gastrointestinal endoscopy (GIE) diagnostics by providing a non-invasive way to capture high-resolution images of the GI tract, facilitating early detection of various diseases. Additionally, it significantly improves accessibility, as it can be deployed in areas with limited medical infrastructure and enables remote diagnostics.
However, a key limitation of this technology is the massive volume of data generated - often exceeding 1 million frames over the course of a 6-8 hour procedure - making manual analysis infeasible and necessitating the development of automated solutions. Current medical image analysis models face challenges due to the variability in image quality, the requirement for expert annotations, and the lack of large, high-quality labeled datasets. To address these challenges, we introduce EndoExtend24, a novel, large-scale gastrointestinal endoscopy dataset comprising over 226,000 labeled images. This dataset is constructed by merging and re-stratifying the train/test splits of ten existing public and private datasets, ensuring no patient data overlap between splits. The dataset includes dynamic class mappings that accommodate variations in labeling granularity across the original datasets, supporting up to 123 distinct pathological findings. 
<br><br>
<p style="text-align: justify; max-width: 800px; margin: auto; font-family: 'Times New Roman', Times, serif; font-size: 24px; line-height: 30px">
We further propose leveraging the EVA-02 model, a vision transformer pretrained on ImageNet-22k with masked image modeling (MIM) using EVA-CLIP as a teacher model, for domain adaptation in GIE diagnostics. The EVA-02 model incorporates advanced architectural features like SwiGLU activations, Rotary Position Embeddings (ROPE), and additional Layer Normalization (LN) in its MLP layers. Pretraining the EVA-02 model on the EndoExtend24 dataset allows it to learn domain-specific features, before being fine-tuned on the Capsule Endoscopy 2024 (CE24) Challenge dataset. Our experimental results demonstrate the effectiveness of this approach, achieving an AUC Macro score of <b>0.993</b> and a balanced accuracy of <b>89.3%</b> on the challenge validation set.
</p>

---

<div style="text-align: center; margin-bottom: 2rem; font-family: 'Times New Roman', Times, serif;">
   <h1>EndoExtend24</h1>
   <img src="assets/endoextend24.png" alt="EndoExtend Datasets" width="80%">
</div>



## Quickstart

1. Create a Virtual Environment with Python 3.11 via ```python3.11 -m venv venv``` and activate
   it ```source venv/bin/activate```
2. Install pytorch 2.4.0+cu121
   via ```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121```
3. Install required packages with `pip install -r requirements.txt`
4. Download the Capsule Vision Challenge
   2024 [dataset](https://github.com/misahub2023/Capsule-Vision-2024-Challenge?tab=readme-ov-file) from the official
   GitHub repository, store it in `data/` **in the repository root** and make sure to have the following structure:
   ```
    data/
    ├── capsulevision/
    │   ├── training/
    │   │   ├── Angioectasia/
    │   │   │   ├── KID/
    │   │   │   │   ├── angioectasia-P0-1.jpg
    │   │   │   │   ├── ...
    │   │   │   ├── KVASIR/
    │   │   │   │   ├── 04a78ef00c5245e0_888.jpg
    │   │   │   │   ├── ...
    │   │   │   ├── SEE-AI/
    │   │   │   │   ├── image00279.jpg
    │   │   │   │   ├── ...   
    │   ├── validation/
    │   │   ├── ...
    │   ├── testing/
    │   │   ├── Images
    │   │   │   ├── 00Z0Xo99wp.jpg
    │   │   │   ├── ...
   ```
5. Download the pre-trained EndoExtend24 model weights via link/command and store them in `pretrained_models/` **in the
   repository root**
6. Make sure that your **repository root** is structured as follows:

```
.
├── configs/
├── data/
├── datasets/
│   ├── ce24/
│   │   ├── class_mapping.json
│   │   ├── train_val.csv
│   │   ├── test.csv
├── pretrained_models/
│   ├── eva02_base_patch14_224.pt_ee24.ckpt
│   ├── eva02_base_patch14_224.ee24_ft_ce24.ckpt   
├── src/
├── README.md
├── requirements.txt
├── infer.py
├── train.py
├── ...
```

7. Run `python train.py --config configs/submission/run_eva02_base_patch14_224.ee24_ft_ce24.yaml` to start fine-tuning
   the pre-trained EndoExtend24 model.

## Inference on the Test Set

## Model Weights

You can find the weights for both the pre-trained model and the fine-tuned downstream task model below. For an easy
download on a bash cluster such as a slurm master, you can use the `gdown` command as shown below.

| **Type**    | **Dataset**  | **Checkpoint**                                                                                                            | **Command**                               |
|-------------|--------------|---------------------------------------------------------------------------------------------------------------------------|-------------------------------------------|
| Pre-trained | EndoExtend24 | [eva02_base_patch14_224.pt_ee24](https://drive.google.com/file/d/1Ok58RCRvKdq1_VcFn35FQOHyznvq8JFr/view?usp=sharing)      | `gdown 1Ok58RCRvKdq1_VcFn35FQOHyznvq8JFr` |
| Fine-tuned  | CE24         | [eva02_base_patch14_224.ee24_ft_ce24](https://drive.google.com/file/d/123TjuBw-34bKXBu7njzKjbcObNXsnuEY/view?usp=sharing) | `gdown 123TjuBw-34bKXBu7njzKjbcObNXsnuEY` |

## Introduction
