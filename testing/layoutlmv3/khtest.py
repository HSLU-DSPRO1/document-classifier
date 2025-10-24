import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    LayoutLMv3ForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate
from PIL import Image
# --------------------------------------------------------------------
# Environment setup
# --------------------------------------------------------------------
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ Running on CPU. Training will be slow.")

# --------------------------------------------------------------------
# Load dataset
# --------------------------------------------------------------------
print("Loading RVL-CDIP dataset...")
dataset = load_dataset("chainyo/rvl-cdip")

labels = dataset["train"].features["label"].names
print(f"Detected {len(labels)} classes: {labels}")

# --------------------------------------------------------------------
# Load model and processor
# --------------------------------------------------------------------
model_name = "microsoft/layoutlmv3-base"
processor = AutoProcessor.from_pretrained(model_name)

model = LayoutLMv3ForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
)

# --------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------

print(dataset["train"].features)

def preprocess(batch):
    # Convert grayscale or palette images to RGB
    images = []
    for img in batch["image"]:
        if img.mode != "RGB":
            img = img.convert("RGB")
        images.append(img)

    encodings = processor(images=images, return_tensors="pt")
    encodings["labels"] = batch["label"]
    return encodings

dataset = dataset.map(preprocess, batched=True)
# --------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

# --------------------------------------------------------------------
# Training configuration
# --------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./rvlcdip-results",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=torch.cuda.is_available(),  # only use fp16 if CUDA available
    report_to="none",  # disable W&B / Hub logging
)

# --------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    compute_metrics=compute_metrics,
)


# --------------------------------------------------------------------
# Train
# --------------------------------------------------------------------
trainer.train()

# --------------------------------------------------------------------
# Evaluate
# --------------------------------------------------------------------
metrics = trainer.evaluate(dataset["test"])
print("Test metrics:", metrics)

# --------------------------------------------------------------------
# Save final model + processor
# --------------------------------------------------------------------
save_dir = "./layoutlmv3-rvlcdip-model"
trainer.save_model(save_dir)
processor.save_pretrained(save_dir)
print(f"✅ Model saved to {save_dir}")