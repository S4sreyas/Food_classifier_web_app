from flask import Flask, render_template, request
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import requests
import cv2
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load the trained model
my_model = tf.keras.models.load_model('model_trained.h5', compile=False)

# List of food categories
food_list = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare", "beet_salad", "beignets",
    "bibimbap", "bread_pudding", "breakfast_burrito", "bruschetta", "caesar_salad", "cannoli", "caprese_salad",
    "carrot_cake", "ceviche", "cheese_plate", "cheesecake", "chicken_curry", "chicken_quesadilla", "chicken_wings",
    "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder", "club_sandwich", "crab_cakes", "creme_brulee",
    "croque_madame", "cup_cakes", "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict", "escargots",
    "falafel", "filet_mignon", "fish_and_chips", "foie_gras", "french_fries", "french_onion_soup", "french_toast",
    "fried_calamari", "fried_rice", "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad",
    "grilled_cheese_sandwich",
    "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup", "hot_dog", "huevos_rancheros", "hummus",
    "ice_cream", "lasagna", "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
    "mussels", "nachos", "omelette", "onion_rings", "oysters", "pad_thai", "paella", "pancakes", "panna_cotta",
    "peking_duck", "pho", "pizza", "pork_chop", "poutine", "prime_rib", "pulled_pork_sandwich", "ramen", "ravioli",
    "red_velvet_cake", "risotto", "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
    "spaghetti_bolognese",
    "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake", "sushi", "tacos", "takoyaki", "tiramisu",
    "tuna_tartare", "waffles"
]

# Function to get calorie information from Edamam API
def get_calories(food_name):
    # Replace 'YOUR_APP_ID' and 'YOUR_APP_KEY' with your actual Edamam API credentials
    app_id = '10d339e2'
    app_key = '1aa6bf1a7864e7ee64a68822bec8f9a0'

    # Remove underscores from the food name
    food_name = food_name.replace('_', ' ')

    url = f"https://api.edamam.com/api/food-database/v2/parser?app_id={app_id}&app_key={app_key}&ingr={food_name}"
    response = requests.get(url)
    data = response.json()
    if 'parsed' in data and data['parsed']:
        calories = data['parsed'][0]['food']['nutrients']['ENERC_KCAL']
        return calories
    else:
        return None

# Function to predict classes of new images
def predict_class(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.

    pred = my_model.predict(img_array)
    pred_index = np.argmax(pred)
    food_name = food_list[pred_index]
    calories = get_calories(food_name)

    return food_name, calories

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/upload', methods=['POST'])
def predict_upload():
    # Check if a file was uploaded
    if 'image' not in request.files:
        return "No file uploaded", 400

    file = request.files['image']

    # Check if the file is empty
    if file.filename == '':
        return "Empty file uploaded", 400

    # Check if the file is an allowed format
    if file:
        filename = 'uploaded_image.jpg'
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        img = cv2.imread(file_path)
        food_name, calories = predict_class(img)
        return f"{food_name}, Calories: {calories}"

@app.route('/predict/camera', methods=['POST'])
def predict_camera():
    img = request.files['image'].read()
    img = Image.open(BytesIO(img))
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    food_name, calories = predict_class(img)
    return f"{food_name}, Calories: {calories}"

if __name__ == '__main__':
    app.run(debug=True)
