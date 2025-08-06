## Visual Language Model

This repository contains the code to implement a visual language model that can process and understand both text and images. The model is designed to handle multimodal inputs, allowing it to generate text based on visual content and vice versa.


To start training just run the following command to install the requirements:
This command will automatically create a virtual environment and install the necessary dependencies.

```bash 
uv run python 
``` 
>[!NOTE]
> Make sure you have `uv` installed. If not, you can install it using 
> ```bash
> wget -qO- https://astral.sh/uv/install.sh | sh
> export PATH="$HOME/.local/bin:$PATH"
> uv --version
> ```


### Visual Encoder

using the traditional Vision Transformer (ViT) architecture, the visual encoder processes images to extract meaningful features. The ViT model is pre-trained on a large dataset, enabling it to understand various visual concepts.


### Language Models

Language Model are LLaMA like architectures with:
- RoPE
- Grouped Query Attention
- KV Cache