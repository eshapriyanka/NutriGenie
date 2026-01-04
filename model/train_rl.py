# -------------------------------------------------------------------------
# NUTRIGENIE: Reinforcement Learning (RL) Module
# Implements "Proximal Policy Optimization" (PPO) as per the Diabetasty Paper
# -------------------------------------------------------------------------

# 1. INSTALL LIBRARIES (Run this in a separate cell first if needed)
# !pip install trl transformers torch pandas accelerate

import torch
import pandas as pd
import numpy as np
from transformers import GPT2Tokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model

# --- CONFIGURATION ---
MODEL_PATH = "./model"  # Path to your fine-tuned GPT-2 model
CSV_PATH = "gi_values.csv"
OUTPUT_DIR = "./model_rl"

# --- 1. LOAD THE KNOWLEDGE BASE (GI Values) ---
def load_gi_database(csv_path):
    print(f"Loading GI Database from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        # Create a dictionary: { "food name": gi_value }
        # Convert names to lowercase for easier matching
        gi_dict = pd.Series(df['Glycemic Index'].values, index=df['Food Name'].str.lower()).to_dict()
        print(f"Loaded {len(gi_dict)} ingredients into memory.")
        return gi_dict
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return {}

GI_DATABASE = load_gi_database(CSV_PATH)

# --- 2. THE REWARD FUNCTION (The Judge) ---
# Logic from Paper: Reward = (Threshold - Avg_GI) - Missing_Penalty
def calculate_reward(generated_text, target_ingredients):
    generated_text = generated_text.lower()
    total_gi = 0
    ingredient_count = 0
    
    # A. Calculate Average GI of the recipe
    words = generated_text.replace(",", " ").replace(".", " ").split()
    
    # We check phrases in the text against our DB
    # (Simple word matching - can be improved with n-grams but this is fast)
    for word in words:
        if word in GI_DATABASE:
            total_gi += GI_DATABASE[word]
            ingredient_count += 1
            
    # Default GI is 55 (Medium) if no ingredients are recognized
    avg_gi = total_gi / ingredient_count if ingredient_count > 0 else 55.0
    
    # B. Missing Ingredient Penalty
    # Paper: "Penalty is associated with every ingredient missing"
    targets = [t.strip().lower() for t in target_ingredients.split(",")]
    found_count = 0
    for target in targets:
        # Check if the target ingredient appears in the generated text
        if target in generated_text:
            found_count += 1
            
    # Huge penalty if it ignores your input (Hallucination check)
    missing_penalty = (len(targets) - found_count) * 2.0 
    
    # C. Calculate Final Score (PPO Reward)
    # The Paper uses a dynamic threshold based on DiSsCo, but for training 
    # we teach it to ALWAYS aim for Low GI (<55).
    threshold_gi = 55.0
    
    # Reward is Positive if Avg_GI < 55, Negative if Avg_GI > 55
    gi_score = (threshold_gi - avg_gi)
    
    # Total Reward
    final_reward = gi_score - missing_penalty
    
    # Normalize between -1 and 1 for stability
    return max(min(final_reward / 50.0, 1.0), -1.0)

# --- 3. TRAINING LOOP ---
def train_rl():
    print("Initializing Model & PPO Trainer...")
    
    # Load Model with a "Value Head" (Required for RL)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(MODEL_PATH)
    
    # Create a "Reference Model" (A copy of the original to keep it speaking English)
    ref_model = create_reference_model(model)
    
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    
    # PPO Configuration
    config = PPOConfig(
        model_name="nutrigenie_rl",
        learning_rate=1.41e-5,
        batch_size=4,
        mini_batch_size=4,
        gradient_accumulation_steps=1,
    )
    
    ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)
    
    # Create Synthetic Prompts for Training
    # (In a real scenario, we would use the dataset again, but this is faster)
    # We combine high GI and low GI inputs to teach it the difference
    sample_prompts = [
        "Spinach, Paneer, Tomato", "Aloo, Gobhi, Peas", "Rice, Dal, Ghee", 
        "Chicken, Onion, Garlic", "Oats, Milk, Apple", "Maida, Sugar, Butter",
        "Egg, Toast, Avocado", "Fish, Lemon, Asparagus", "Pasta, Cream, Cheese"
    ] * 20 # 180 training steps
    
    print("Starting PPO Training Loop...")
    
    for i, prompt_text in enumerate(sample_prompts):
        
        # 1. Encode Input
        query_txt = f"Ingredients: {prompt_text} \nInstructions:"
        query_tensors = tokenizer.encode(query_txt, return_tensors="pt").to(ppo_trainer.accelerator.device)
        
        # 2. Generate Recipe (Action)
        response_tensor = ppo_trainer.generate(
            [query_tensors[0]], 
            return_prompt=False,
            max_new_tokens=80, # Keep it short for speed
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
        response_txt = tokenizer.decode(response_tensor[0])
        
        # 3. Calculate Reward (Feedback)
        reward_value = calculate_reward(response_txt, prompt_text)
        rewards = [torch.tensor(reward_value).to(ppo_trainer.accelerator.device)]
        
        # 4. Update Model Weights (The Learning Step)
        stats = ppo_trainer.step([query_tensors[0]], [response_tensor[0]], rewards)
        
        # Print progress every 10 steps
        if i % 10 == 0:
            print(f"Step {i}: Prompt='{prompt_text}' | Reward={reward_value:.3f}")

    print(f"Training Complete! Saving RL-Tuned model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("DONE. Zip and download this folder.")

if __name__ == "__main__":
    train_rl()
