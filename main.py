import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import lightning as L

# --- 改善点1: ハイパーパラメータを定数として定義 (マジックナンバーの排除) ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
ENCODER_DIM = 64
DECODER_DIM = 3

class LitAutoEncoder(L.LightningModule):
    # --- 改善点2: 型ヒント (Type Hints) の追加 ---
    def __init__(self, encoder_dim: int = ENCODER_DIM, decoder_dim: int = DECODER_DIM) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, decoder_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(decoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, 28 * 28)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        return optimizer

def main() -> None:
    # データの準備
    dataset = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    # モデルの作成
    autoencoder = LitAutoEncoder()

    # 学習の設定 (動作確認用なので1エポックだけ回す)
    trainer = L.Trainer(max_epochs=1, accelerator="auto", devices=1)
    trainer.fit(model=autoencoder, train_dataloaders=train_loader)

if __name__ == "__main__":
    main()
