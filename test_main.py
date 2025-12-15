import torch
import pytest
from main import LitAutoEncoder

def test_model_structure() -> None:
    """モデルの構造と出力サイズが正しいか確認するテスト"""
    model = LitAutoEncoder()
    batch_size = 4
    # 28x28の画像データをフラットにした入力 (batch, 784)
    dummy_input = torch.randn(batch_size, 28 * 28)

    # エンコーダーを通す
    z = model.encoder(dummy_input)
    # デコーダーを通す
    reconstructed = model.decoder(z)

    # 入力と出力のサイズが同じであることを確認 (AutoEncoderの基本動作)
    assert reconstructed.shape == dummy_input.shape

def test_optimizer_setup() -> None:
    """オプティマイザが正しく設定されているか確認するテスト"""
    model = LitAutoEncoder()
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam)
