import streamlit as st
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# --- PAGE CONFIG ---
st.set_page_config(page_title="NutriGenie", page_icon="🥗", layout="wide")

# --- CUSTOM CSS FOR MODERN LANDING PAGE ---
st.markdown("""
    <style>
    .stApp { background-color: white; }
    
    /* Navigation Bar */
    .nav-bar {
        display: flex; justify-content: space-between; align-items: center;
        padding: 15px 5%; background-color: #2E7D32; position: sticky; top: 0; z-index: 1000;
        width: 100%;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        
    }
    .nav-links a {
        text-decoration: none; color: white; font-weight: bold; margin-left: 30px; font-size: 1rem;
    }

    /* Hero Section */
    .hero-title {
        font-size: 4rem; font-weight: 800; color: #1B263B; line-height: 1.1; margin-top: 20px;
    }
    .hero-highlight { color: #4CAF50; }
    
    /* App Section Background */
    # .app-section {
    #     background-color: #f1f8e9; padding: 60px 5%; border-radius: 50px 50px 0 0; margin-top: 50px;
    # }
    .app-section {
        background-color: #f1f8e9;
        width: 100%;              /* decrease box size */
        margin: 20px auto;       /* center the box horizontally */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);

        display: flex;           /* enable flexbox */
        justify-content: center; /* center horizontally */
        align-items: center;     /* center vertically */
        text-align: center;      /* center text inside */
    }


    /* White Cards Styling */
    div[data-testid="stVerticalBlock"] > div:has(div.white-card) {
        background-color: white !important;
        padding: 35px !important;
        border-radius: 25px !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08) !important;
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #2E7D32 0%, #4CAF50 100%);
        color: white; border-radius: 12px; border: none; font-weight: bold; height: 3.2em; transition: 0.3s;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(76, 175, 80, 0.3); }

    /* Hide default Streamlit padding */
    .block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. RESOURCE LOADER ---
@st.cache_resource
def load_resources():
    MODEL_PATH = "./model"
    GI_FILE_PATH = "gi_values.csv"
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
        model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
    except: return None, None, {}

    gi_map = {}
    try:
        gi_df = pd.read_csv(GI_FILE_PATH, on_bad_lines='skip')
        gi_df = gi_df.dropna(subset=['Food Name', 'Glycemic Index'])
        for _, row in gi_df.iterrows():
            gi_map[str(row['Food Name']).lower().strip()] = float(row['Glycemic Index'])
    except: pass

    # Safety overrides
    gi_map.update({"sugar": 100, "rice": 73, "potato": 82, "paneer": 30, "spinach": 15, "chicken": 0, "honey": 60})
    return tokenizer, model, gi_map

tokenizer, model, gi_map = load_resources()

# --- 2. LOGIC FUNCTIONS ---
def calculate_dissco(age, hba1c, bmi, gender, heart_disease):
    gf = 1.4 if gender == "Female" else 1.0 
    hdf = 1.0
    if heart_disease == "Hypertension": hdf += 0.5
    elif heart_disease == "Stent or Bypass": hdf += 1.0 
    score = ((age / 10) + hba1c + (bmi / 10)) * gf * hdf
    return min(score / 50.0, 1.0) 

def generate_recipe(ingredients, tokenizer, model):
    prompt = f"Ingredients: {ingredients}\nInstructions:"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=500, temperature=0.8, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def analyze_recipe_health(ingredients_input, recipe_text, dissco_score, gi_map):
    limit = 55 if dissco_score > 0.6 else (65 if dissco_score > 0.3 else 75)
    input_risks = [(w, gi_map[w]) for w in ingredients_input.lower().replace(","," ").split() if w in gi_map and gi_map[w] > limit]
    gen_words = recipe_text.lower().replace(","," ").replace("."," ").split()
    output_risks = [(w, gi_map[w]) for w in gen_words if w in gi_map and gi_map[w] > limit and (w, gi_map[w]) not in input_risks]
    return len(input_risks) > 0, input_risks, output_risks, limit

# --- HEADER / NAV ---
# You can replace the URL below with your actual logo file path or a different icon URL
LOGO_URL = "https://www.shutterstock.com/image-vector/healthy-food-logo-design-combination-260nw-2398835953.jpg"

st.markdown(f"""
    <div class="nav-bar">
        <div style="display: flex; align-items: center;">
            <img src="{LOGO_URL}" width="55" style="margin-right: 18px; filter: drop-shadow(2px 2px 4px rgba(0,0,0,0.2));">
            <h2 style="color: white; margin:0; letter-spacing: -1px; font-size: 1.8rem;">NutriGenie</h2>
        </div>
        <div class="nav-links">
            <a href="#home">HOME</a>
            <a href="#features">FEATURES</a>
        </div>
    </div><div id="home"></div>""", unsafe_allow_html=True)
# st.markdown("""
#     <div class="nav-bar">
#         <h2 style="color: white; margin:0; letter-spacing: -1px;">NutriGenie</h2>
#         <div class="nav-links">
#             <a href="#home">HOME</a>
#             <a href="#features">FEATURES</a>
#         </div>
#     </div><div id="home"></div>""", unsafe_allow_html=True)

# --- HERO SECTION ---
col_h1, col_h2 = st.columns([1.2, 1])
with col_h1:
    st.markdown("""
        <div style="padding: 40px 0 0 8%;">
            <h1 class="hero-title">DON'T LET YOUR <br><span class="hero-highlight">DIABETES</span><br>STOP YOU FROM EATING <span class="hero-highlight">TASTY</span></h1>
            <p style="font-size: 1.2rem; color: #555; margin: 25px 0; max-width: 500px;">
                Using AI, we customize each recipe to fit your specific health condition, dietary goals, and taste.
            </p>
        </div>""", unsafe_allow_html=True)
    
    # "Get Started" scrolling button
    if st.button("GET STARTED"):
        st.markdown('<script>window.parent.document.querySelector("section.main").scrollTo(0, 1000);</script>', unsafe_allow_html=True)

with col_h2:
    st.image("https://img.freepik.com/free-vector/healthy-people-carrying-different-icons_53876-66139.jpg", width=480)

# --- APP SECTION ---
#st.markdown('<div id="features" class="app-section">', unsafe_allow_html=True)
st.markdown(
    """
    <div id="features" class="app-section">
        <h2>Diabetes-Friendly Indian Meal Planning</h2>
    </div>
    """,
    unsafe_allow_html=True
)

col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
   #  st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.markdown(
    """
    <div class="white-card">
        <h3>Patient Data</h3>
    </div>
    """,
    unsafe_allow_html=True
)
    # st.markdown("## 🧬 Patient Data")
    
    age = st.slider("Age", 1, 100, 45)
    gender = st.selectbox("Gender 🚻", ["Male", "Female"])
    hba1c = st.number_input("HbA1c Level (%) 🧪", 4.0, 15.0, 6.5)
    heart = st.selectbox("Heart Condition ❤️", ["None", "Hypertension", "Stent or Bypass"])
    
    cl1, cl2 = st.columns(2)
    weight = cl1.number_input("Weight (kg)", 30, 150, 70)
    height = cl2.number_input("Height (cm)", 100, 250, 170)
    
    bmi = weight / ((height/100)**2)
    dissco_score = calculate_dissco(age, hba1c, bmi, gender, heart)
    
    # Score stays here permanently
    st.markdown("---")
    st.metric("Your DiSsCo Score", f"{dissco_score:.2f}")
    if dissco_score > 0.6: st.error("High Risk Profile")
    elif dissco_score > 0.3: st.warning("Moderate Risk Profile")
    else: st.success("Safe Profile")
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    # st.markdown('<div class="white-card">', unsafe_allow_html=True)
    # st.markdown("<h2 style='text-align: center;'>Ingredients & Recipe</h2>", unsafe_allow_html=True)
    st.markdown(
    """
    <div class="white-card">
        <h3>Ingredients & Recipe</h3>
    </div>
    """,
    unsafe_allow_html=True
)
    
    ing_input = st.text_area("What's in your kitchen?", placeholder="e.g. Spinach, Paneer, Garlic", height=100)
    
    if st.button("Generate Safe Recipe ✨"):
        if not ing_input:
            st.warning("Please enter ingredients first!")
        else:
            with st.spinner("Genie is analyzing health safety..."):
                raw_recipe = generate_recipe(ing_input, tokenizer, model)
                is_blocked, in_risks, out_risks, limit = analyze_recipe_health(ing_input, raw_recipe, dissco_score, gi_map)
                
                if is_blocked:
                    st.error(f"### 🚫 Recipe Blocked (Limit: GI {limit})")
                    for food, gi in in_risks: st.write(f"❌ **{food.upper()}** (GI: {int(gi)}) is too high!")
                else:
                    if out_risks: st.warning(f"⚠️ AI added high-GI items: {', '.join([f[0] for f in out_risks])}")
                    else: st.success("✅ Perfectly safe for your health profile!")
                    
                    # Display Instructions
                    try:
                        final_recipe = raw_recipe.split("Instructions:")[1]
                    except:
                        final_recipe = raw_recipe
                    
                    st.markdown("#### 📖 Instructions")
                    st.write(final_recipe)
                    st.caption(f"Safety Threshold: GI {limit}")
    else:
        # Default view when not generated
        st.info("Input ingredients and click the button to see your personalized healthy recipe here.")
        st.image("https://img.freepik.com/free-vector/chef-concept-illustration_114360-2241.jpg", width=250)
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True) # End of App Section