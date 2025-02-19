import librosa
import json
import numpy as np
import tensorflow as tf
import wikipedia  # To fetch species details from Wikipedia

def deep_learning_prediction(audio_file):
    """
    Deep learning model prediction function. This should process the audio file and return the result along with species details.
    """
    # Load the Prediction JSON File to Predict Target_Label
    with open('bird_detection_interface/prediction_models/deep_learning/prediction.json', mode='r') as f:
        prediction_dict = json.load(f)

    # Extract the Audio_Signal and Sample_Rate from Input Audio
    audio, sample_rate = librosa.load(audio_file)

    # Extract the MFCC Features and Aggregate
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_features = np.mean(mfccs_features, axis=1)

    # Reshape MFCC features to match the expected input shape for Conv1D both batch & feature dimension
    mfccs_features = np.expand_dims(mfccs_features, axis=0)
    mfccs_features = np.expand_dims(mfccs_features, axis=2)

    # Convert into Tensors
    mfccs_tensors = tf.convert_to_tensor(mfccs_features, dtype=tf.float32)

    # Load the Model and Prediction
    model = tf.keras.models.load_model('bird_detection_interface/prediction_models/deep_learning/model.h5')
    prediction = model.predict(mfccs_tensors)

    # Find the Maximum Probability Value
    target_label = np.argmax(prediction)

    # Find the Target_Label Name using Prediction_dict
    predicted_class = prediction_dict[str(target_label)]
    confidence = round(np.max(prediction)*100, 2)

    # Fetch details about the species from Wikipedia
    species_info, image_url = fetch_species_details(predicted_class)

    # Return the predicted class and species information
    return {
        "predicted_class": predicted_class,
        "species_info": species_info,
        "species_image_url": image_url
    }

def fetch_species_details(species_name):
    """
    Fetch details about the species from Wikipedia using wikipedia-api.
    """
    # Initialize Wikipedia API with user-agent specified
    user_agent = "bird-detection-system/1.0 (kamathsamarth@gmail.com)"  # Change this to your contact email or app name
    result = wikipedia.summary(species_name)
    page = wikipedia.page(species_name)
    print(page.images)
    image_url = page.images[2]

    return result, image_url

