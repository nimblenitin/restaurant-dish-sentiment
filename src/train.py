"""
mT5 Fine-tuning for Restaurant Food Recommendation
Train a model to analyze reviews and recommend top/bottom dishes
"""

import os
import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import evaluate
from preprocess import prepare_data, MODEL_CHECKPOINT


# Training configuration
BATCH_SIZE = 8
NUM_TRAIN_EPOCHS = 10
LEARNING_RATE = 5.6e-5
WEIGHT_DECAY = 0.01
SAVE_TOTAL_LIMIT = 3
MODEL_NAME = "mt5-finetuned-restaurant-food"


def compute_metrics(eval_pred):
    """
    Compute ROUGE metrics for evaluation
    
    Args:
        eval_pred: Evaluation predictions
        
    Returns:
        Dictionary of metrics
    """
    rouge_score = evaluate.load("rouge")
    
    predictions, labels = eval_pred
    
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    
    # Compute ROUGE
    result = rouge_score.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    
    # Extract scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    return {k: round(v, 4) for k, v in result.items()}


def setup_training(data_path: str = "data/training_data.json"):
    """
    Setup training pipeline
    
    Args:
        data_path: Path to training data
        
    Returns:
        trainer: Seq2SeqTrainer instance
        tokenized_datasets: Tokenized datasets
    """
    global tokenizer
    
    print("\n" + "="*60)
    print("SETTING UP TRAINING")
    print("="*60)
    
    # Prepare data
    tokenized_datasets, tokenizer = prepare_data(data_path)
    
    # Load model
    print(f"\nLoading model from {MODEL_CHECKPOINT}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
    print(f"Model loaded! Parameters: {model.num_parameters():,}")
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # Training arguments
    output_dir = f"models/{MODEL_NAME}"
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        save_total_limit=SAVE_TOTAL_LIMIT,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        predict_with_generate=True,
        logging_steps=100,
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        greater_is_better=True,
        fp16=False,
        report_to="none",
    )
    
    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print("\nTraining setup complete!")
    print(f"Model: {MODEL_CHECKPOINT}")
    print(f"Training examples: {len(tokenized_datasets['train'])}")
    print(f"Validation examples: {len(tokenized_datasets['validation'])}")
    print(f"Epochs: {NUM_TRAIN_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    
    return trainer, tokenized_datasets


def train(trainer):
    """
    Run training
    
    Args:
        trainer: Seq2SeqTrainer instance
    """
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    # Train
    train_result = trainer.train()
    
    # Save model
    trainer.save_model()
    
    # Print results
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Training loss: {train_result.metrics['train_loss']:.4f}")
    print(f"Training time: {train_result.metrics['train_runtime']:.2f}s")
    print(f"Model saved to: {trainer.args.output_dir}")
    
    return train_result


def evaluate(trainer):
    """
    Evaluate the trained model
    
    Args:
        trainer: Seq2SeqTrainer instance
    """
    print("\n" + "="*60)
    print("EVALUATING MODEL")
    print("="*60)
    
    # Evaluate
    eval_results = trainer.evaluate()
    
    # Print results
    print("\nEvaluation Results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    
    return eval_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train mT5 for restaurant food recommendations")
    parser.add_argument("--data", default="data/training_data.json",
                       help="Path to training data JSON")
    parser.add_argument("--epochs", type=int, default=NUM_TRAIN_EPOCHS,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                       help="Learning rate")
    parser.add_argument("--evaluate-only", action="store_true",
                       help="Only evaluate, don't train")
    
    args = parser.parse_args()
    
    # Update globals if specified
    NUM_TRAIN_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    
    # Setup training
    trainer, tokenized_datasets = setup_training(args.data)
    
    if not args.evaluate_only:
        # Train
        train_result = train(trainer)
        
        # Evaluate
        eval_results = evaluate(trainer)
    else:
        # Just evaluate
        eval_results = evaluate(trainer)
    
    print("\nDone!")
