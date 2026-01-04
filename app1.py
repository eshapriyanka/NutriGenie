import streamlit as st
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# --- CONFIGURATION ---
MODEL_PATH = "./model"
GI_FILE_PATH = "gi_values.csv"

# --- 1. ROBUST RESOURCE LOADER ---
@st.cache_resource
def load_resources():
    # A. Load AI Model
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
        model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
    except:
        return None, None, {}

    # B. Load GI Database
    gi_map = {}
    try:
        gi_df = pd.read_csv(GI_FILE_PATH, on_bad_lines='skip')
        gi_df = gi_df.dropna(subset=['Food Name', 'Glycemic Index'])
        for index, row in gi_df.iterrows():
            food_name = str(row['Food Name']).lower().strip()
            gi_val = float(row['Glycemic Index'])
            gi_map[food_name] = gi_val
    except:
        pass

    # OVERRIDES (Safety Net)
    manual_fixes = {
        # HIGH GI (BAD)
        "sugar": 100, "jaggery": 85, "honey": 60, "corn syrup": 115,
        "rice": 73, "white rice": 73, "cooked rice": 73,
        "potato": 82, "potatoes": 82, "aloo": 82,
        "maida": 75, "refined flour": 75, "white bread": 75, "bread": 75,
        "pasta": 70, "noodles": 70,
        
        # LOW GI (GOOD/SAFE)
        "paneer": 30, "cottage cheese": 30,
        "spinach": 15, "palak": 15,
        "tomato": 30, "tomatoes": 30,
        "dal": 30, "lentils": 30, "moong": 30,
        "chicken": 0, "fish": 0, "egg": 0, "eggs": 0,
        "ginger": 0, "garlic": 0, "turmeric": 0, "spices": 0, "onion": 15,
        "ghee": 0, "oil": 0, "butter": 0, "curd": 30, "yogurt": 30,
        "cauliflower": 15, "gobhi": 15, "cabbage": 15, "broccoli": 15,
        "water": 0, "lemon": 20, "lime": 20
    }
    gi_map.update(manual_fixes)
    
    return tokenizer, model, gi_map

# --- 2. CALCULATE DISSCO ---
def calculate_dissco(age, hba1c, bmi, gender, heart_disease):
    gf = 1.4 if gender == "Female" else 1.0 
    hdf = 1.0
    if heart_disease == "Hypertension":
        hdf += 0.5
    elif heart_disease == "Stent or Bypass":
        hdf += 1.0 
    score = ((age / 10) + hba1c + (bmi / 10)) * gf * hdf
    return min(score / 50.0, 1.0) 

# --- 3. CLEAN TEXT ---
def clean_recipe_text(text):
    # Cut off at serving suggestions to avoid hallucinations
    cutoff_phrases = [
        "Serve with", "Serve hot", "Serve the", "Serve it", 
        "Enjoy with", "Best served", "Pair with", "Garnish with"
    ]
    for phrase in cutoff_phrases:
        if phrase in text:
            text = text.split(phrase)[0]
    return text.strip().strip(",").strip("-")

# --- 4. SAFETY LOGIC ---
def analyze_recipe_health(ingredients_input, recipe_text, dissco_score, gi_map):
    if dissco_score > 0.6:
        max_allowed_gi = 55
    elif dissco_score > 0.3:
        max_allowed_gi = 65
    else:
        max_allowed_gi = 75
        
    input_risks = []
    output_risks = []
    
    # Check User Input (Strict)
    user_words = ingredients_input.lower().replace(",", " ").split()
    for word in user_words:
        if word in gi_map and gi_map[word] > max_allowed_gi:
            input_risks.append((word, gi_map[word]))
                
    # Check Generated Text (Lenient)
    gen_words = recipe_text.lower().replace(",", " ").replace(".", " ").split()
    for word in gen_words:
        if word in gi_map and gi_map[word] > max_allowed_gi:
            if (word, gi_map[word]) not in input_risks:
                output_risks.append((word, gi_map[word]))
    
    return len(input_risks) > 0, input_risks, output_risks, max_allowed_gi

# --- 5. GENERATE RECIPE (FIXED) ---
def generate_recipe(ingredients, dissco_score, tokenizer, model):
    # FIXED PROMPT: No more "Heat oil" forcing.
    # We match the training format exactly: "Ingredients: ... \nInstructions:"
    prompt = f"Ingredients: {ingredients}\nInstructions:"
    
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    outputs = model.generate(
        inputs, 
        max_length=1000,        # Give it enough space to finish
        min_length=50,         # Force it to write something substantial
        num_return_sequences=1, 
        temperature=0.8,       # Higher creative freedom to connect ingredients
        top_k=50, 
        top_p=0.95,
        repetition_penalty=1.2,
        do_sample=True, 
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- USER INTERFACE ---
st.set_page_config(page_title="Nutrigenie", page_icon="🥗", layout="wide")
tokenizer, model, gi_map = load_resources()

st.title("🥗 Nutrigenie")
st.markdown("Generates recipes and checks safety against medical data.")

# Sidebar
with st.sidebar:
    st.header("Patient Profile")
    age = st.number_input("Age", 20, 100, 45)
    weight = st.number_input("Weight (kg)", 30, 150, 70)
    height = st.number_input("Height (cm)", 100, 250, 170)
    gender = st.selectbox("Gender", ["Male", "Female"])
    hba1c = st.number_input("HbA1c Level (%)", 4.0, 15.0, 6.5)
    heart = st.selectbox("Heart Condition", ["None", "Hypertension", "Stent or Bypass"])

    height_m = height / 100
    bmi = weight / (height_m ** 2)
    dissco_score = calculate_dissco(age, hba1c, bmi, gender, heart)
    
    st.divider()
    st.metric("DiSsCo Score", f"{dissco_score:.2f}")
    if dissco_score > 0.6:
        st.error("Profile: High Risk")
    elif dissco_score > 0.3:
        st.warning("Profile: Moderate Risk")
    else:
        st.success("Profile: Low Risk")

# Main
col1, col2 = st.columns([2, 1])

with col1:
    ingredients = st.text_area("Ingredients", "Spinach, Paneer")
    
    if st.button("Generate"):
        if model is None:
            st.error("Model error.")
        else:
            with st.spinner("Processing..."):
                # 1. Generate
                raw_text = generate_recipe(ingredients, dissco_score, tokenizer, model)
                
                # 2. Clean Text
                cleaned_text = clean_recipe_text(raw_text)
                
                # 3. Analyze Safety
                is_blocked, input_risks, output_risks, limit = analyze_recipe_health(ingredients, cleaned_text, dissco_score, gi_map)

                # 4. Display Logic
                # Separate Ingredients and Instructions for cleaner look
                try:
                    parts = cleaned_text.split("Instructions:")
                    final_ingredients = parts[0].replace("Ingredients:", "").strip()
                    final_instructions = parts[1].strip()
                except:
                    final_ingredients = ingredients
                    final_instructions = cleaned_text

                st.markdown("### 🍲 Result")
                
                if is_blocked:
                    st.error("⚠️ RECIPE BLOCKED: Unsafe Ingredients")
                    st.write(f"Your Safety Limit: **GI {limit}**")
                    for food, gi in input_risks:
                        st.write(f"- ❌ **{food.upper()}** (GI: {int(gi)})")
                    st.info("Please remove these ingredients and try again.")
                    
                elif len(output_risks) > 0:
                    st.warning("⚠️ Warning: Recipe generated, but check details.")
                    st.write("The AI added these high-GI items:")
                    for food, gi in output_risks:
                        st.write(f"- ⚠️ **{food.upper()}** (GI: {int(gi)})")
                    st.success("✅ Main Ingredients are Safe!")
                    st.markdown("---")
                    st.markdown(f"**Ingredients:** {final_ingredients}")
                    st.markdown(f"**Instructions:** {final_instructions}")
                    
                else:
                    st.success("✅ SAFE TO EAT")
                    st.write(f"All ingredients are within your limit of GI {limit}.")
                    st.markdown("---")
                    st.markdown(f"**Ingredients:** {final_ingredients}")
                    st.markdown(f"**Instructions:** {final_instructions}")

# with col2:
#     st.info("System Logic")
#     st.write(f"Database Size: {len(gi_map)} items")
#     if "honey" in gi_map:
#         st.write(f"Honey GI: {gi_map['honey']}")
#     if "chicken" in gi_map:
#         st.write(f"Chicken GI: {gi_map['chicken']}")