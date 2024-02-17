import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from Generator.EnergyModel import DeepEnergyModel
from Generator.GenCallBack import GenerateCallback
from Generator.SampCallBack import SamplerCallback
from Generator.OutCallBack import OutlierCallback
from Generator.SettingsEM import MAX_EPOCHS


def trainEnergyModel(train_loader, test_loader, device, CHECKPOINT_PATH, **kwargs):
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "MNIST"),
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=MAX_EPOCHS,
        gradient_clip_val=0.1,
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True, mode="min", monitor="val_contrastive_divergence"
            ),
            GenerateCallback(every_n_epochs=5),
            SamplerCallback(every_n_epochs=5),
            OutlierCallback(),
            LearningRateMonitor("epoch"),
        ],
    )
    # Check whether pretrained model exists. If yes, load it and skip training
    # pretrained_filename = os.path.join(CHECKPOINT_PATH, "MNIST.ckpt")
    # if os.path.isfile(pretrained_filename):
    #     print("Found pretrained model, loading...")
    #     model = DeepEnergyModel.load_from_checkpoint(pretrained_filename)
    # else:
    pl.seed_everything(42)
    model = DeepEnergyModel(**kwargs)
    trainer.fit(model, train_loader, test_loader)
    model = DeepEnergyModel.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )
    # No testing as we are more interested in other properties
    return model
