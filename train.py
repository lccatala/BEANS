from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoImageProcessor, TrainingArguments, Trainer
from transformers import AutoModelForImageClassification, MobileNetV2Config, MobileNetV2ForImageClassification
from transformers import DefaultDataCollator
import torch
import torch.nn as nn
import torch.nn.functional as F
import evaluate
import numpy as np

class ImageDistillTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, temperature=None, lambda_param=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.student = kwargs['model']
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        self.teacher.to(device)
        self.teacher.eval()
        self.temperature = temperature # Controls importance of each soft target
        self.lambda_param = lambda_param # Controls importance of the distillation loss

        # Given two data points P and Q, KL Divergence explains 
        # how much extra information we need to represent P using Q.
        # Returns 0 when P and Q are identical.
        self.loss_function = nn.KLDivLoss(reduction='batchmean')

    def compute_loss(self, student, inputs, return_outputs=False):
        student_output = self.student(**inputs)

        with torch.no_grad():
            teacher_output = self.teacher(**inputs)

        # Compute soft targets for teacher and student
        soft_teacher = F.softmax(teacher_output.logits / self.temperature, dim=-1)
        soft_student = F.softmax(student_output.logits / self.temperature, dim=-1)

        # Distillation loss
        distillation_loss = self.loss_function(soft_student, soft_teacher) * (self.temperature ** 2)

        # True label loss
        student_target_loss = student_output.loss

        # Final loss
        loss = (1. + self.lambda_param) * student_target_loss + self.lambda_param * distillation_loss

        return (loss, student_output) if return_outputs else loss




if __name__ == '__main__':
    print('Loading dataset...')
    dataset = load_dataset("beans")

    teacher_processor = AutoImageProcessor.from_pretrained('merve/beans-vit-224')

    def process(examples):
        processed_inputs = teacher_processor(examples['image'])
        return processed_inputs

    print('Processing dataset...')
    processed_datasets = dataset.map(process, batched=True)
    # print('Logging in...')

    # login()

    training_args = TrainingArguments(
        output_dir='checkpoints',
        num_train_epochs=30,
        fp16=False,
        logging_dir='logs_wandb',
        logging_strategy='epoch',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        report_to=['tensorboard'],
        push_to_hub=False,
        hub_strategy='every_save',
        hub_model_id='distillation'
    )

    num_labels = len(processed_datasets['train'].features['labels'].names)

    # Initialise models
    teacher_model = AutoModelForImageClassification.from_pretrained(
        'merve/beans-vit-224',
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )

    # Train MobileNetV2 from scratch
    student_config = MobileNetV2Config()
    student_config.num_labels = num_labels
    student_model = MobileNetV2ForImageClassification(student_config)


    accuracy = evaluate.load('accuracy')
    def compute_metrics(eval_prod):
        predictions, labels = eval_prod
        acc = accuracy.compute(references=labels, predictions=np.argmax(predictions, axis=1))
        return {'accuracy': acc['accuracy']}

    data_collator = DefaultDataCollator()
    trainer = ImageDistillTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=processed_datasets["train"],
        eval_dataset=processed_datasets["validation"],
        data_collator=data_collator,
        # tokenizer=teacher_extractor,
        compute_metrics=compute_metrics,
        temperature=5,
        lambda_param=0.5
    )

    trainer.train()
    print(trainer.evaluate(processed_datasets["test"]))
