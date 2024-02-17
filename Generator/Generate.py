from IPython.display import set_matplotlib_formats
import matplotlib
import seaborn as sns

## PyTorch
import torch
import pytorch_lightning as pl
import gc
from Generator.TrainGenerator import trainEnergyModel
from Generator.GenCallBack import GenerateCallback
from Generator.SettingsEM import (
    CHECKPOINT_PATH,
    LEARNING_RATE,
)
from DataHandlers.CustomDataset import CustomMNISTDataset
from torch.utils.data import DataLoader

sns.reset_orig()
matplotlib.rcParams["lines.linewidth"] = 2.0
set_matplotlib_formats("svg", "pdf")  # For export


def generate(train_loader: DataLoader, test_loader: DataLoader, device):
    # Path to the folder where the pretrained models are saved

    final_images = []
    final_labels = []

    for targetLabel in range(10):
        images_target_class = []
        for originalImage, originalLabel in train_loader.dataset:
            if originalLabel == targetLabel:
                images_target_class.append(originalImage)

        N_TO_BE_GENERATED = len(images_target_class)
        generated_dataset = CustomMNISTDataset(
            images_target_class, [targetLabel] * N_TO_BE_GENERATED
        )

        original_loader_target_class = DataLoader(
            generated_dataset,
            batch_size=128,
            shuffle=True,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

        del generated_dataset, images_target_class
        gc.collect()

        model = trainEnergyModel(
            original_loader_target_class,
            test_loader,
            device,
            CHECKPOINT_PATH,
            img_shape=(1, 28, 28),
            batch_size=train_loader.batch_size,
            lr=LEARNING_RATE,
            beta1=0.0,
        )

        model.to(device)
        pl.seed_everything(43)

        for turn in range(2): # 'N_TO_BE_GENERATED' is around 6000. On our system we cannot generate 6000 images at once due to memory constraint. So, we break them into two batches of 3000 images and run this process two times.
            if N_TO_BE_GENERATED % 2 == 0 or turn == 0:
                n_generated_images_at_this_turn = N_TO_BE_GENERATED // 2
            else:
                n_generated_images_at_this_turn = N_TO_BE_GENERATED // 2 + 1
            callback = GenerateCallback(
                batch_size=n_generated_images_at_this_turn, vis_steps=8, num_steps=256
            )
            imgs_per_step = callback.generate_imgs(model)
            imgs_per_step = imgs_per_step.cpu()

            for i in range(n_generated_images_at_this_turn):
                step_size = callback.num_steps // callback.vis_steps
                imgs_to_plot = imgs_per_step[step_size - 1 :: step_size, i]
                imgs_to_plot = torch.cat([imgs_per_step[0:1, i], imgs_to_plot], dim=0)
                finalImage = imgs_to_plot[8, :, :, :]
                final_images.append(finalImage)  # Remove the batch dimension
                final_labels.append(targetLabel)

    # Convert the list of tensors into a single tensor
    generated_dataset = CustomMNISTDataset(final_images, final_labels)
    return generated_dataset
