import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, jsonify
import tempfile # For handling temporary audio files
import wave # To check WAV file headers if necessary (though librosa handles most)

app = Flask(__name__)

# --- Configuration (These should match your training setup) ---
max_pad_len = 174  # The fixed length you padded/truncated MFCCs to during training
n_mfcc = 40        # The number of MFCC coefficients extracted

# Define the class labels in the same order as your model's output
# Based on your classification report, the order is Batak, Javanese, Sundanese, Umum.
class_labels = ['Batak', 'Javanese', 'Sundanese', 'Umum']

# Create a mapping from index to class name for easy interpretation of predictions
idx_to_class = {i: label for i, label in enumerate(class_labels)}

# --- Load the trained model globally when the app starts ---
best_cnn_lstm_model = None
try:
    # Ensure this path is correct relative to app.py
    model_path = 'best_cnn_lstm_model.keras'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    best_cnn_lstm_model = load_model(model_path)
    print(f"Model loaded successfully from {model_path}.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load the model. Please check the path and file integrity: {e}")
    # It's better to let the app fail early if the model is essential.
    # In a real-world scenario, you might want to serve a "maintenance" page.
    best_cnn_lstm_model = None # Ensure it's None if loading fails

# --- Prediction Function (from previous turn, slightly adapted) ---
def predict_accent_from_audio(audio_file_path, model, max_pad_len, n_mfcc, idx_to_class):
    """
    Predicts the accent from a given audio file using a trained CNN-LSTM model.

    Args:
        audio_file_path (str): The full path to the WAV audio file.
        model (tf.keras.Model): The loaded and trained Keras CNN-LSTM model.
        max_pad_len (int): The maximum padding length used for MFCC features during training.
        n_mfcc (int): The number of MFCC coefficients.
        idx_to_class (dict): A dictionary mapping numerical class indices to accent names.

    Returns:
        tuple: (predicted_accent_label_str, confidence_float) or (error_message_str, None).
    """
    if model is None:
        return "Model not loaded. Please check server logs.", None

    try:
        # 1. Load the audio file
        # Using sr=None to preserve original sampling rate, librosa handles various formats.
        y, sr = librosa.load(audio_file_path, sr=None)

        # 2. Extract MFCC features
        mfcc_features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        # 3. Pad or truncate MFCC features to the expected length
        if mfcc_features.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc_features.shape[1]
            mfcc_features = np.pad(mfcc_features, pad_width=((0, 0), (0, pad_width)), mode='constant')
        elif mfcc_features.shape[1] > max_pad_len:
            mfcc_features = mfcc_features[:, :max_pad_len]
        
        # Ensure the number of MFCCs is consistent (n_mfcc rows)
        if mfcc_features.shape[0] != n_mfcc:
            return f"Error: MFCC features shape mismatch. Expected {n_mfcc} rows, got {mfcc_features.shape[0]}. Check librosa settings or n_mfcc.", None

        # 4. Reshape for model input: (batch_size, timesteps, features, channels)
        # Your model's input_shape is (max_pad_len, n_mfcc, 1).
        # The MFCC features extracted are typically (n_mfcc, timesteps).
        # So we transpose to (timesteps, n_mfcc) and add the batch and channel dimensions.
        processed_features = mfcc_features.transpose(1, 0)
        processed_features = processed_features[np.newaxis, ..., np.newaxis] # Adds batch_size and channel dimension

        # 5. Make a prediction
        predictions = model.predict(processed_features, verbose=0)
        
        # 6. Interpret the prediction
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        predicted_accent = idx_to_class[predicted_class_idx]
        confidence = float(np.max(predictions)) # Convert to float for JSON serialization

        return predicted_accent, confidence

    except Exception as e:
        print(f"Error during audio processing or prediction: {e}")
        return f"An error occurred during prediction: {e}", None

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_accent', methods=['POST'])
def predict_accent_endpoint():
    if 'audio_data' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio_data']
    if not audio_file.filename:
        return jsonify({"error": "No selected file"}), 400

    # Create a temporary file to save the audio
    # Use a specific suffix to help librosa identify the format (e.g., .webm, .wav, .ogg)
    # The frontend is configured to send 'audio/webm' or 'audio/wav', so using a generic 'audio' for now.
    # librosa can often guess the format from the magic bytes, but a suffix helps.
    
    # Determine suffix based on mimetype if possible, otherwise default.
    mimetype = audio_file.mimetype
    if 'webm' in mimetype:
        suffix = '.webm'
    elif 'wav' in mimetype:
        suffix = '.wav'
    elif 'ogg' in mimetype:
        suffix = '.ogg'
    else:
        suffix = '.tmp' # Fallback for unknown types

    temp_audio_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            audio_file.save(temp_file)
            temp_audio_path = temp_file.name

        print(f"Received audio file saved temporarily to: {temp_audio_path}")

        # Perform accent prediction
        predicted_accent, confidence = predict_accent_from_audio(
            temp_audio_path, best_cnn_lstm_model, max_pad_len, n_mfcc, idx_to_class
        )

        if confidence is not None:
            return jsonify({
                "predicted_accent": predicted_accent,
                "confidence": confidence
            })
        else:
            return jsonify({"error": predicted_accent}), 500 # predicted_accent contains error message
            
    except Exception as e:
        print(f"Server-side error during prediction: {e}")
        return jsonify({"error": f"Server-side error: {e}"}), 500
    finally:
        # Clean up the temporary file
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            print(f"Temporary file {temp_audio_path} removed.")

# --- Run the Flask App ---
if __name__ == '__main__':
    # For development, run with debug=True
    # For production, use gunicorn or a similar WSGI server
    app.run(debug=True, host='0.0.0.0', port=5000)
