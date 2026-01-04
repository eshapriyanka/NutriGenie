import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import os

# 1. Load and Preprocess Data
def prepare_dataset(file_path):
    print("Loading dataset...")
    # FIX: Added on_bad_lines='skip' to ignore rows with formatting errors
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip')
    except Exception as e:
        print(f"Standard load failed, trying python engine: {e}")
        df = pd.read_csv(file_path, sep=',', on_bad_lines='skip', engine='python')

    print(f"Successfully loaded {len(df)} recipes.")
    
    # We will use 'TranslatedIngredients' and 'TranslatedInstructions' for English
    # Drop rows with missing values
    df = df.dropna(subset=['TranslatedIngredients', 'TranslatedInstructions'])
    
    # Format: "Ingredients: [List] Instructions: [Steps]"
    df['text'] = "Ingredients: " + df['TranslatedIngredients'] + " \nInstructions: " + df['TranslatedInstructions'] + "<|endoftext|>"
    
    # Save to a text file for GPT-2 training
    with open("train_recipes.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(df['text'].tolist()))
    print("Dataset prepared: train_recipes.txt")

# 2. Train the Model
def train_gpt2():
    model_name = "gpt2"
    output_dir = "./model"
    
    print("Downloading GPT-2 tokenizer and model...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    print("Processing dataset for training...")
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path="train_recipes.txt",
        block_size=128
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3, 
        per_device_train_batch_size=4, 
        save_steps=500, 
        save_total_limit=2,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    print("Starting training on GPU...")
    trainer.train()
    
    print("Saving model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model saved to ./model folder!")

if __name__ == "__main__":
    prepare_dataset("Indian_Food_Recipes.csv")
    train_gpt2()