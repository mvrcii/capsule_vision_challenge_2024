import argparse
import os
import warnings

import numpy as np
import pandas as pd
import torch
import wandb

from src.data.dataset import ImageDataset
from src.models.timm.timm_model import TimmModel
from src.utils.class_mapping import load_class_mapping
from src.utils.transform_utils import load_transforms

warnings.filterwarnings("ignore", ".*A new version of Albumentations is*")


def save_predictions_to_excel(image_paths, y_pred, output_path):
    class_columns = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 'Lymphangiectasia', 'Normal',
                     'Polyp', 'Ulcer', 'Worms']
    y_pred_classes = np.argmax(y_pred, axis=1)
    predicted_class_names = [class_columns[i] for i in y_pred_classes]
    df_prob = pd.DataFrame(y_pred, columns=class_columns)
    df_prob.insert(0, 'image_path', image_paths)
    df_class = pd.DataFrame({'image_path': image_paths, 'predicted_class': predicted_class_names})
    df_merged = pd.merge(df_prob, df_class, on='image_path')
    df_merged.to_excel(output_path, index=False)


def prepare_model(ckpt_path, config, class_to_idx):
    """Load the model from the checkpoint."""
    if not ckpt_path:
        raise ValueError("Checkpoint path not provided")
    print(f"Loading model from checkpoint: {ckpt_path}")
    model = TimmModel.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        config=config,
        class_to_idx=class_to_idx
    )
    if not model:
        raise ValueError("Failed to load model from checkpoint")
    model.eval()
    return model


def load_data(dataset_csv_path, dataset_path):
    test_path = os.path.join(dataset_csv_path, 'test.csv')
    X_test = pd.read_csv(test_path)
    X_test['frame_path'] = X_test['frame_path'].apply(lambda x: os.path.join(dataset_path, x))
    return X_test


def get_image_batch(dataset, start_idx, batch_size):
    max_idx = len(dataset)
    end_idx = min(start_idx + batch_size, max_idx)
    batch_images = []

    for i in range(start_idx, end_idx):
        # Assuming dataset returns only image when class_to_index is not provided
        image = dataset[i]
        batch_images.append(image)

    # Stack the list of images into tensors
    batch_images = torch.stack(batch_images)

    return batch_images


def main(args):
    api = wandb.Api()
    run = api.run(f'{args.entity}/{args.wandb_project}/{args.run_id}')
    config = argparse.Namespace(**run.config)

    if args.dataset_path:
        config.dataset_path = args.dataset_path

    if args.dataset_csv_path:
        config.dataset_csv_path = args.dataset_csv_path

    class_mapping = load_class_mapping(os.path.join(config.dataset_csv_path, 'class_mapping.json'))
    idx_to_class = {v: k for k, v in class_mapping.items()}

    _, val_transforms = load_transforms(img_size=config.img_size, transform_path=config.transform_path)

    test_df = load_data(config.dataset_csv_path, config.dataset_path)

    # Initialize Dataset
    test_dataset = ImageDataset(test_df, transform=val_transforms)

    # Load Model
    model = prepare_model(ckpt_path=args.ckpt_path, config=config, class_to_idx=class_mapping)
    if torch.cuda.is_available():
        model = model.cuda()

    batch_size = 8
    num_batches = (len(test_dataset) // batch_size) + 1

    preds = []
    for batch_num in range(1, num_batches + 1):
        current_start_idx = (batch_num - 1) * batch_size

        images = get_image_batch(test_dataset, start_idx=current_start_idx, batch_size=batch_size)

        if torch.cuda.is_available():
            images = images.cuda()

        # Forward pass
        with torch.no_grad():
            logits = model.forward(images)

        batch_preds = torch.softmax(logits, dim=1).detach().cpu().numpy()
        preds.append(batch_preds)

    preds = np.vstack(preds)
    preds = preds.reshape(-1, 10)

    os.makedirs('submission', exist_ok=True)
    img_paths = test_df['proposed_name'].values

    save_predictions_to_excel(image_paths=img_paths, y_pred=preds, output_path='submission/prediction.xlsx')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict on images and create a submission xlsx file.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--dataset_csv_path", type=str,
                        help="Path to the dataset csv files. Allows for overriding if required. (Default is the path from the checkpoint config)")

    parser.add_argument("--run_id", type=str, required=True, help="Run ID to fetch")
    parser.add_argument("--wandb_project", default="SEER", type=str, help="Wandb project name")
    parser.add_argument("--entity", default="mvrcii_", type=str, help="Wandb entity name")

    arguments = parser.parse_args()

    main(arguments)
