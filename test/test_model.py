import torch


from vlm.model.vision_transformer import ViT
from vlm.config import VLMConfig


def vit_model_test():
    config = VLMConfig()
    config.vit_cls_flag = True  # Set to True for classification tasks

    model = ViT.from_pretrained(config)
    model = ViT(config)

    model.eval()

    # Dummy image tensor: (batch_size=1, channels=3, height=224, width=224)
    dummy_image = torch.randn(8, 3, config.vit_img_size, config.vit_img_size)

    with torch.no_grad():
        output = model(dummy_image)

    assert output.shape == (8, config.vit_hidden_dim), "Output shape mismatch"

    print("ViT model test passed. Output shape:", output.shape)


if __name__ == "__main__":
    vit_model_test()
