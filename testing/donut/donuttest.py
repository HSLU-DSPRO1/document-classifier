import os
import torch
import numpy as np
from torch import nn
from datasets import load_dataset
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    TrainingArguments,
    Trainer,
)
import evaluate

# --------------------------------------------------------------------
# Environment setup
# --------------------------------------------------------------------
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"

# Enable performance optimizations
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


# --------------------------------------------------------------------
# Custom classification model built on Donut encoder
# --------------------------------------------------------------------
class DonutForClassification(nn.Module):
    def __init__(self, encoder, num_labels):
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, pixel_values, labels=None):
        outputs = self.encoder(pixel_values=pixel_values)
        pooled = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}


# --------------------------------------------------------------------
# Main execution
# --------------------------------------------------------------------
if __name__ == '__main__':
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    # --------------------------------------------------------------------
    # Load the preprocessed dataset - THIS IS FAST!
    # --------------------------------------------------------------------
    print("\nLoading RVL-CDIP dataset (preprocessed version)...")
    dataset = load_dataset("hf-tuner/rvl-cdip-document-classification")

    print(f"Dataset structure: {dataset}")
    print(f"Features: {dataset['train'].features}")
    print(f"First example keys: {dataset['train'][0].keys()}")

    # Get label information
    labels = dataset["train"].features["label"].names
    num_labels = len(labels)
    print(f"\nDetected {num_labels} classes: {labels}")

    # Split training data to create validation set
    print("\nCreating train/validation split...")
    train_val_split = dataset["train"].train_test_split(test_size=0.1, seed=42)
    dataset["train"] = train_val_split["train"]
    dataset["validation"] = train_val_split["test"]
    print(f"Train size: {len(dataset['train'])}")
    print(f"Validation size: {len(dataset['validation'])}")
    print(f"Test size: {len(dataset['test'])}")

    # --------------------------------------------------------------------
    # Load Donut model
    # --------------------------------------------------------------------
    base_model = "naver-clova-ix/donut-base"
    processor = DonutProcessor.from_pretrained(base_model, use_fast=True)

    donut_model = VisionEncoderDecoderModel.from_pretrained(base_model)
    encoder = donut_model.encoder

    model = DonutForClassification(encoder, num_labels)

    # Move model to GPU
    if torch.cuda.is_available():
        model = model.cuda()


    # --------------------------------------------------------------------
    # Preprocessing function
    # --------------------------------------------------------------------
    def preprocess(batch):
        # Process images
        images = []
        for img in batch["image"]:
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)

        encodings = processor(images=images, return_tensors="pt")
        encodings["labels"] = batch["label"]
        return encodings


    # Preprocess dataset
    print("\nPreprocessing dataset...")
    dataset = dataset.map(
        preprocess,
        batched=True,
        batch_size=100,
        remove_columns=dataset["train"].column_names,
        desc="Preprocessing"
    )

    dataset.set_format(type="torch")
    print("✅ Preprocessing complete!")

    # --------------------------------------------------------------------
    # Metrics
    # --------------------------------------------------------------------
    accuracy = evaluate.load("accuracy")


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=preds, references=labels)


    # --------------------------------------------------------------------
    # Training arguments - Optimized for RTX 5090
    # --------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir="./donut-classifier-results",
        overwrite_output_dir=True,
        num_train_epochs=10,

        # Batch sizes
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,  # effective batch = 64

        learning_rate=3e-5,
        weight_decay=0.05,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,

        # Eval & logging
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,

        # Performance
        fp16=True,
        tf32=True,
        optim="adamw_torch_fused",
        gradient_checkpointing=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        max_grad_norm=1.0,

        # Misc
        report_to="none",
        seed=42,
        save_total_limit=3,
    )

    # --------------------------------------------------------------------
    # Trainer
    # --------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
    )

    # --------------------------------------------------------------------
    # Train
    # --------------------------------------------------------------------
    print("\n🚀 Starting training...")
    print("💡 Monitor GPU usage - should be 80-95%!")
    trainer.train()

    # --------------------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------------------
    print("\nEvaluating on test set...")
    metrics = trainer.evaluate(dataset["test"])
    print("✅ Test metrics:", metrics)

    # --------------------------------------------------------------------
    # Save model
    # --------------------------------------------------------------------
    save_dir = "./donut-classifier-model"
    trainer.save_model(save_dir)
    processor.save_pretrained(save_dir)
    print(f"✅ Model saved to {save_dir}")