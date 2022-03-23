This directory provides a notebook and ImageNet-1k class mapping file to run evaluation on the ImageNet-1k `validation` split using the [ViT models from TF-Hub](https://tfhub.dev/sayakpaul/collections/vision_transformer/1). One should use this same setup to evaluate the [MLP-Mixer models from TF-Hub](https://tfhub.dev/sayakpaul/collections/mlp-mixer/1). The notebook assumes the following files are present in your working directory:

* The `validation` split directory of ImageNet-1k.
* The class mapping files (`.json`).
