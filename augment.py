from DataHandlers.DataLoading import loadOriginalTrainSet, loadTestSet
from Generator.SettingsEM import GENERATED_FILE_PATH

## PyTorch
import torch
import pytorch_lightning as pl
from Generator.Generate import generate
from ShowImages import showImages


def main():
    # Setting the seed
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print("Device:", device)

    train_loader = loadOriginalTrainSet()
    test_loader = loadTestSet()

    generated_dataset = generate(train_loader, test_loader, device)
    torch.save(generated_dataset, GENERATED_FILE_PATH)
    print('Data saved successfully!')
    showImages(generated_dataset, "Generated Images")


if __name__ == "__main__":
    main()
