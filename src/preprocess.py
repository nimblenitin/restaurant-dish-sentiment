"""
Data Preprocessing for Restaurant Food Recommendation Model
Tokenizes and prepares data for mT5 fine-tuning
"""

import json
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from typing import Dict, List


# Model configuration
MODEL_CHECKPOINT = "google/mt5-small"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128


def load_training_data(file_path: str = "data/training_data.json") -> List[Dict]:
    """
    Load training data from JSON file
    
    Args:
        file_path: Path to training data JSON
        
    Returns:
        List of training examples
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} training examples")
    return data


def create_dataset(data: List[Dict], train_split: float = 0.8) -> DatasetDict:
    """
    Create HuggingFace DatasetDict from training data
    
    Args:
        data: List of training examples
        train_split: Fraction of data for training
        
    Returns:
        DatasetDict with train and validation splits
    """
    # Shuffle the data
    import random
    random.seed(42)
    random.shuffle(data)
    
    # Split into train and validation
    split_idx = int(len(data) * train_split)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    # Create datasets
    train_dataset = Dataset.from_dict({
        "input": [ex["input"] for ex in train_data],
        "output": [ex["output"] for ex in train_data]
    })
    
    val_dataset = Dataset.from_dict({
        "input": [ex["input"] for ex in val_data],
        "output": [ex["output"] for ex in val_data]
    })
    
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })
    
    print(f"Created dataset with {len(train_dataset)} train and {len(val_dataset)} validation examples")
    
    return dataset_dict


def preprocess_function(examples: Dict, tokenizer) -> Dict:
    """
    Tokenize and preprocess examples for mT5
    
    Args:
        examples: Batch of examples with 'input' and 'output'
        tokenizer: mT5 tokenizer
        
    Returns:
        Tokenized examples
    """
    # Tokenize inputs
    model_inputs = tokenizer(
        examples["input"],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length"
    )
    
    # Tokenize targets
    labels = tokenizer(
        text_target=examples["output"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length"
    )
    
    # Set labels
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs


def tokenize_dataset(dataset_dict: DatasetDict, tokenizer) -> DatasetDict:
    """
    Apply tokenization to the entire dataset
    
    Args:
        dataset_dict: DatasetDict with train and validation splits
        tokenizer: mT5 tokenizer
        
    Returns:
        Tokenized DatasetDict
    """
    print("\nTokenizing dataset...")
    
    tokenized_datasets = dataset_dict.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=["input", "output"],
        desc="Tokenizing"
    )
    
    print(f"Tokenization complete!")
    print(f"Train dataset columns: {tokenized_datasets['train'].column_names}")
    print(f"Train dataset size: {len(tokenized_datasets['train'])}")
    
    return tokenized_datasets


def prepare_data(file_path: str = "data/training_data.json"):
    """
    Complete data preparation pipeline
    
    Args:
        file_path: Path to training data JSON
        
    Returns:
        Tokenized DatasetDict and tokenizer
    """
    print("\n" + "="*60)
    print("DATA PREPROCESSING")
    print("="*60)
    
    # Load training data
    data = load_training_data(file_path)
    
    # Create dataset splits
    dataset_dict = create_dataset(data)
    
    # Load tokenizer
    print(f"\nLoading tokenizer from {MODEL_CHECKPOINT}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    print(f"Tokenizer loaded! Vocabulary size: {tokenizer.vocab_size}")
    
    # Tokenize dataset
    tokenized_datasets = tokenize_dataset(dataset_dict, tokenizer)
    
    return tokenized_datasets, tokenizer


def save_tokenized_data(tokenized_datasets: DatasetDict, output_dir: str = "data/tokenized"):
    """
    Save tokenized datasets to disk
    
    Args:
        tokenized_datasets: Tokenized DatasetDict
        output_dir: Directory to save datasets
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    tokenized_datasets["train"].save_to_disk(f"{output_dir}/train")
    tokenized_datasets["validation"].save_to_disk(f"{output_dir}/validation")
    
    print(f"Saved tokenized datasets to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess data for mT5 training")
    parser.add_argument("--input", default="data/training_data.json",
                       help="Input JSON file with training data")
    parser.add_argument("--output", default="data/tokenized",
                       help="Output directory for tokenized datasets")
    parser.add_argument("--save", action="store_true",
                       help="Save tokenized datasets to disk")
    
    args = parser.parse_args()
    
    # Prepare data
    tokenized_datasets, tokenizer = prepare_data(args.input)
    
    # Save if requested
    if args.save:
        save_tokenized_data(tokenized_datasets, args.output)
    
    # Show sample
    print("\n" + "="*60)
    print("SAMPLE TOKENIZED EXAMPLE")
    print("="*60)
    
    sample = tokenized_datasets["train"][0]
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Attention mask shape: {sample['attention_mask'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    
    # Decode to verify
    print(f"\nDecoded input: {tokenizer.decode(sample['input_ids'], skip_special_tokens=True)[:200]}...")
    print(f"Decoded output: {tokenizer.decode([l for l in sample['labels'] if l != -100], skip_special_tokens=True)}")
