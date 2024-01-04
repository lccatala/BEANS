# Knowledge Distillation
Experiment with Knowledge Distillation from a Vision Transformer into a MobileNet.

The teacher model is a [Vision Transformer](https://huggingface.co/google/vit-base-patch16-224-in21k) pre-trained on ImageNet and fine-tuned on the [Beans Dataset](https://huggingface.co/datasets/beans) from Hugging Face.
The student model is a randomly initialized MobileNetV2.

In the training set, it obtained a loss of `59.2%` and an accuracy of `72.7%`, a `12%` higher than training the same net from scratch with the same hyperparameters.
