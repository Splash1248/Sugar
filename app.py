# filename: app.py
import os
import google.generativeai as genai
import gradio as gr
import matplotlib.pyplot as plt
import re
from dotenv import load_dotenv # Import dotenv to load environment variables from .env file

# Load environment variables from a .env file if it exists
load_dotenv()

# ====== API KEYS ======
# Get the API key from environment variables
# Set the GOOGLE_API_KEY environment variable in your system or in a .env file
gemini_key = os.getenv('GOOGLE_API_KEY')

# ====== ENHANCED CACHE ======
CACHE = {
    "coke": "12oz can: 39g sugar (9.75 cubes). Try Zevia soda!",
    "pepsi": "12oz can: 41g sugar (10.25 cubes). Try Pepsi Zero Sugar!",
    "banana": "Medium banana: 14g sugar (3.5 cubes). Pair with peanut butter.",
    "frappuccino": "Grande caramel frappuccino: 54g sugar (13.5 cubes!). Try iced coffee with sugar-free syrup.",
    "yogurt": "Flavored yogurt (6oz): ~19g sugar (4.75 cubes). Choose plain Greek yogurt + berries.",
    "muffin": "Blueberry muffin (bakery style): ~30-40g sugar (7.5-10 cubes). Try a whole-wheat muffin or oatmeal.",
    "waffle": "Plain waffle (frozen, 4-inch): ~3-5g sugar (0.75-1.25 cubes). Top with fruit instead of syrup.",
    "orange juice": "Orange juice (8oz): ~21g sugar (5.25 cubes). Try water with a splash of lemon.", # Added orange juice
    "default": "Could not find specific sugar info. Please check spelling or try a different item." # Updated default
}

# ====== SUGAR VISUALIZATION ======
def show_sugar(grams):
    try:
        grams = float(grams)
    except (ValueError, TypeError):
        grams = 0
    cubes = grams / 4
    # Prevent matplotlib GUI from popping up if running headlessly
    plt.switch_backend('Agg')
    plt.figure(figsize=(6,2))
    plt.barh([f'{grams}g Sugar'], [cubes], color=['#FF6B6B'])
    plt.xlabel('Sugar Cubes (1 cube = 4g)')
    max_tick = int(cubes) + 2 if cubes > 0 else 5
    plt.xticks(range(0, max_tick))
    plt.title('Approximate Sugar Content')
    plt.grid(axis='x', linestyle='--')
    plt.tight_layout()
    return plt

# ====== RESPONSE PARSING FUNCTION ======
def parse_sugar_grams(text_response):
    if not isinstance(text_response, str): # Add check if response is not string
        return 0
    match = re.search(r'\b(\d+\.?\d*|\.\d+)\s?g(?:\s?sugar)?\b', text_response, re.IGNORECASE)
    if match:
        try: return float(match.group(1))
        except ValueError: return 0
    match = re.search(r'\b(\d+\.?\d*|\.\d+)\s?grams?\b', text_response, re.IGNORECASE)
    if match:
        try: return float(match.group(1))
        except ValueError: return 0
    return 0

# ====== GEMINI AI FUNCTION ======
def sugar_advisor(query):
    query = query.strip().lower()

    # 1. Check enhanced cache
    for food_key, response_text in CACHE.items():
        if re.search(r'\b' + re.escape(food_key) + r'\b', query):
            print(f"Cache hit for: {food_key}")
            return response_text

    # 2. Check if API key exists before trying
    if not gemini_key:
        print("API Key not found. Please set the GOOGLE_API_KEY environment variable.")
        return "API Key is missing. Cannot contact the AI service."

    # 3. Try Gemini API call
    try:
        print("Attempting Gemini API call...")
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')

        prompt = f"""You are a helpful assistant providing concise nutritional information about sugar content. Respond EXACTLY in this format:
"[Food Name] (approx. serving size): [X]g sugar ([Y] cubes). Try [healthy alternative suggestion]."
Calculate Y as X divided by 4. Be factual and brief. If unsure, state that clearly instead of guessing. Use common serving sizes if not specified.

Query: Provide sugar content and a healthy alternative for: {query}"""

        response = model.generate_content(prompt)

        print("Gemini API call successful.")
        if response.parts:
             generated_text = response.text
        else:
             print(f"Gemini response blocked or empty. Reason: {response.prompt_feedback.block_reason}")
             generated_text = f"Response blocked due to: {response.prompt_feedback.block_reason}. Please rephrase your query."

        return generated_text

    except Exception as e:
        print(f"Gemini API Error: {e}")
        # Provide more specific feedback if possible
        error_message = f"An error occurred while contacting the AI service: {e}"
        if "API key not valid" in str(e):
             error_message = "Gemini API Error: The API key provided is not valid. Please check your key."
        elif "quota" in str(e).lower():
             error_message = "Gemini API Error: You have exceeded your API quota. Please check your usage limits."
        return error_message # Return the error to the user interface

    # This part is now less likely to be reached directly unless key was missing initially
    print("Returning default message.")
    return CACHE["default"]

# ====== COMBINED FUNCTION FOR GRADIO ======
def get_sugar_info_and_visualize(query):
    text_result = sugar_advisor(query)
    grams = parse_sugar_grams(text_result)
    plot_result = show_sugar(grams)
    return text_result, plot_result

# ====== GRADIO UI ======
# Define the interface components
with gr.Blocks(theme=gr.themes.Soft(), title="SugarWise AI - Gemini") as demo:
    gr.Markdown("## 🍬 SugarWise AI - Gemini Edition")

    with gr.Row():
        inp = gr.Textbox(label="Ask about any food/drink",
                         placeholder="e.g. 'sugar in orange juice'")
        btn = gr.Button("Analyze", variant="primary")

    with gr.Row():
        out = gr.Textbox(label="Sugar Facts", interactive=False)
        fig = gr.Plot(label="Sugar Cubes Visualization")

    btn.click(
        fn=get_sugar_info_and_visualize,
        inputs=inp,
        outputs=[out, fig]
    )

    gr.Examples(
         examples=[
            ["sugar in a can of pepsi"],
            ["how much sugar in 1 banana"],
            ["starbucks caramel frappuccino sugar"],
            ["healthy alternative to candy"],
            ["sugar in blueberry muffin"],
            ["sugar in coke zero"],
            ["sugar in 1 plain waffle"],
            ["sugar in orange juice"]
        ],
        inputs=inp
    )

# ====== MAIN EXECUTION BLOCK ======
# This ensures the demo launches only when the script is run directly
if __name__ == "__main__":
    if not gemini_key:
        print("-----------------------------------------------------")
        print("WARNING: GOOGLE_API_KEY environment variable not set.")
        print("The application will only use cached values.")
        print("Please set the key (e.g., in a .env file) and restart.")
        print("-----------------------------------------------------")
        # Optionally, you could disable the button or show a message in the UI
        # demo.load(lambda: gr.update(interactive=False), None, btn) # Example: Disable button if key missing

    print("Launching Gradio Interface...")
    # share=False is default and usually preferred for local execution
    # You can add server_name="0.0.0.0" to make it accessible on your local network
    demo.launch()