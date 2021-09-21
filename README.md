# ViT-jax2tf

This repository hosts code for converting the original Vision Transformer models [1] (JAX) to
TensorFlow. Find the model collection on TensorFlow Hub: https://tfhub.dev/sayakpaul/collections/vision_transformer/1.

The original models were fine-tuned on the ImageNet-1k dataset [2]. For more details
on the training protocols, please follow [3]. The authors of [3] open-sourced about
**50k different variants of Vision Transformer models** in JAX. Using the 
[`conversion.ipynb`](https://colab.research.google.com/github/sayakpaul/ViT-jax2tf/blob/main/conversion.ipynb)
notebook, one should be able to take a model from the pool of models and convert that
to TensorFlow and use that with TensorFlow Hub and Keras.

The original model classes and weights [4] using the `jax2tf` tool [5].

## Vision Transformers on TensorFlow Hub

Eight best performing ImageNet-1k models have also been made available on TensorFlow 
Hub that can be used either for off-the-shelf image classification or transfer learning.
Please follow the [`model-selector.ipynb`](https://colab.research.google.com/github/sayakpaul/ViT-jax2tf/blob/main/model-selector.ipynb)
notebook to understand how these models were chosen.

The table below provides a performance summary:

| **Model** | **Top-1 Accuracy (%)** |
|:---:|:---:|
| B/8 | 85.948 |
| L/16 | 85.716 |
| B/16 | 84.018 |
| R50-L/32 | 83.784 |
| R26-S/32 (light aug) | 80.944 |
| R26-S/32 (medium aug) | 80.462 |
| S/16 | 80.462 |
| B/32 | 79.436 |

Note that the top-1 accuracy is reported on ImageNet-1k validation set.

### Image classifiers

* [ViT-S16](https://tfhub.dev/sayakpaul/vit_s16_classification/1)
* [ViT-B8](https://tfhub.dev/sayakpaul/vit_b8_classification/1)
* [ViT-B16](https://tfhub.dev/sayakpaul/vit_b16_classification/1)
* [ViT-B32](https://tfhub.dev/sayakpaul/vit_b32_classification/1)
* [ViT-L16](https://tfhub.dev/sayakpaul/vit_l16_classification/1)
* [ViT-R26-S32 (light augmentation)](https://tfhub.dev/sayakpaul/vit_r26_s32_lightaug_classification/1)
* [ViT-R26-S32 (medium augmentation)](https://tfhub.dev/sayakpaul/vit_r26_s32_medaug_classification/1)
* [ViT-R50-L32](https://tfhub.dev/sayakpaul/vit_r50_l32_classification/1)

### Feature extractors

* [ViT-S16](https://tfhub.dev/sayakpaul/vit_s16_fe/1)
* [ViT-B8](https://tfhub.dev/sayakpaul/vit_b8_fe/1)
* [ViT-B16](https://tfhub.dev/sayakpaul/vit_b16_fe/1)
* [ViT-B32](https://tfhub.dev/sayakpaul/vit_b32_fe/1)
* [ViT-L16](https://tfhub.dev/sayakpaul/vit_l16_fe/1)
* [ViT-R26-S32 (light augmentation)](https://tfhub.dev/sayakpaul/vit_r26_s32_lightaug_fe/1)
* [ViT-R26-S32 (medium augmentation)](https://tfhub.dev/sayakpaul/vit_r26_s32_medaug_fe/1)
* [ViT-R50-L32](https://tfhub.dev/sayakpaul/vit_r50_l32_fe/1)

## Other notebooks

* [`classification.ipynb`](https://colab.research.google.com/github/sayakpaul/ViT-jax2tf/blob/main/classification.ipynb): Shows how to load a Vision Transformer model from TensorFlow Hub
  and run image classification.
* [`fine_tune.ipynb`](https://colab.research.google.com/github/sayakpaul/ViT-jax2tf/blob/main/fine_tune.ipynb): Shows how to
  fine-tune a Vision Transformer model from TensorFlow Hub on the `tf_flowers` dataset.

## References

[1] [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale by Dosovitskiy et al.](https://arxiv.org/abs/2010.11929)

[2] [ImageNet-1k](https://www.image-net.org/challenges/LSVRC/2012/index.php)

[3] [How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers by Steiner et al.](https://arxiv.org/abs/2106.10270)

[4] [Vision Transformer GitHub](https://github.com/google-research/vision_transformer)

[5] [jax2tf tool](https://github.com/google/jax/tree/main/jax/experimental/jax2tf/)

## Acknowledgements

Thanks to the authors of Vision Transformers for their efforts put into open-sourcing
the models.
