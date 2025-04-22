# views.py
import json
import os

import requests
import io
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet50 import preprocess_input
import firebase_admin
from firebase_admin import credentials, firestore



# Initialize Firebase
firebase_config = os.getenv("FIREBASE_CREDENTIALS")

if not firebase_config:
    raise ValueError("Firebase credentials not found in environment variables!")

cred = credentials.Certificate(json.loads(firebase_config))
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load model once
model = ResNet50(weights='imagenet', include_top=False)


@csrf_exempt
def recommend(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=400)

    # Process uploaded outfit image
    if 'image' not in request.FILES:
        return JsonResponse({'error': 'No image uploaded'}, status=400)

    try:
        # Extract outfit features
        outfit_img = Image.open(request.FILES['image'])
        outfit_features = extract_features(outfit_img)
    except Exception as e:
        return JsonResponse({'error': f'Image processing failed: {str(e)}'}, status=400)

    recommendations = []

    # Get all shoes from Firestore
    shoes_ref = db.collection('shoes').stream()

    for shoe in shoes_ref:
        shoe_data = shoe.to_dict()
        try:
            # Process shoe image
            shoe_response = requests.get(shoe_data['image_url'], timeout=10)
            shoe_img = Image.open(io.BytesIO(shoe_response.content))
            shoe_features = extract_features(shoe_img)

            # Calculate similarity
            similarity = np.dot(outfit_features.flatten(), shoe_features.flatten())
            recommendations.append((shoe_data, similarity))
        except Exception as e:
            print(f"Skipping shoe {shoe.id}: {str(e)}")
            continue

    # Sort and get top 5
    recommendations.sort(key=lambda x: x[1], reverse=True)
    top_shoes = [shoe[0] for shoe in recommendations[:5]]

    # Return essential data
    return JsonResponse({
        'recommendations': [{
            'name': shoe.get('name'),
            'image_url': shoe.get('image_url'),
            'price': shoe.get('price'),
            'description': shoe.get('description')
        } for shoe in top_shoes]
    })


def extract_features(img):
    img = img.resize((224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()