import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import urllib.request

# STEP 2: Load Pretrained model
model = MobileNetV2(weights='imagenet')

# STEP 3: Download Image
image_url = "https://picsum.photos/300/300?random=12" # Changed URL to ensure valid image and random selection
image_path = "test_image.jpg"
urllib.request.urlretrieve(image_url, image_path)

# STEP 4: Preprocess the Image
img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
processed_img = preprocess_input(img_array)

# STEP 5: Predict
predictions = model.predict(processed_img)
decoded_results = decode_predictions(predictions, top=5)[0]

# STEP 6: Show Image & Predictions
plt.imshow(img)
plt.axis("off")
plt.title("Input Image")
plt.show()

print("Top Predictions:\n")
for pred in decoded_results:
    class_name = pred[1]
    probability = pred[2]
    print(f"{class_name}: {probability*100:.2f}%")