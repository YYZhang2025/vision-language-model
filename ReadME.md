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

For details description of the projects, please visited this [link](https://yuyang.info/posts/Projects/VLM/vlm.html). 


### Visual Encoder
For the image encoder, we use a Vision Transformer (ViT) architecture. The model is designed to process images by dividing them into patches, which are then linearly embedded and passed through a series of transformer blocks. 

![](assets/vit.gif)

