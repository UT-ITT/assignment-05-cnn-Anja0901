"""
Media Controller using Hand Gesture Recognition
Classifies three hand gestures (like, dislike, stop) and maps them to media controls
"""

import cv2
import json
import numpy as np
import os
import random
import time
import threading
from pathlib import Path

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, RandomFlip, RandomContrast
from keras.metrics import categorical_crossentropy
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Try to import pynput for media control
try:
    from pynput.media_control import Controller as MediaController
    from pynput.media_control import MediaKey
    PYNPUT_AVAILABLE = True
except ImportError:
    print("WARNING: pynput not installed. Media controls will not work.")
    print("Install with: pip install pynput")
    PYNPUT_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Gestures to train on (changed from rock, peace to dislike, stop)
CONDITIONS = ['like', 'dislike', 'stop']

# Image preprocessing
IMG_SIZE = 64
SIZE = (IMG_SIZE, IMG_SIZE)
COLOR_CHANNELS = 3

# Dataset path
DATASET_PATH = '../gesture_dataset_sample'
MODEL_PATH = './gesture_recognition.keras'

# Real-time inference configuration
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to register a gesture
NO_GESTURE_FRAMES = 3  # Frames to wait before sending "no gesture"
INFERENCE_SKIP_FRAMES = 2  # Process every Nth frame for latency optimization

# ============================================================================
# GESTURE-TO-MEDIA-CONTROL MAPPING
# ============================================================================

GESTURE_TO_ACTION = {
    'like': 'volume_up',
    'dislike': 'volume_down',
    'stop': 'pause'
}

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def load_annotations(conditions, dataset_path):
    """Load gesture annotations from JSON files"""
    annotations = {}
    for condition in conditions:
        anno_path = f'{dataset_path}/_annotations/{condition}.json'
        try:
            with open(anno_path) as f:
                annotations[condition] = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Annotation file not found: {anno_path}")
            annotations[condition] = {}
    return annotations


def preprocess_image(img, color_channels=3, size=(64, 64)):
    """Preprocess image: convert color space and resize"""
    if color_channels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img, size)
    return img_resized


def load_dataset(conditions, dataset_path, size, color_channels):
    """Load and preprocess all training images"""
    annotations = load_annotations(conditions, dataset_path)
    
    images = []
    labels = []
    label_names = []
    
    for condition in conditions:
        condition_path = f'{dataset_path}/{condition}'
        if not os.path.exists(condition_path):
            print(f"Warning: Dataset directory not found: {condition_path}")
            continue
            
        for filename in tqdm(os.listdir(condition_path), desc=f"Loading {condition}"):
            try:
                uid = filename.split('.')[0]
                img = cv2.imread(f'{condition_path}/{filename}')
                
                if img is None:
                    continue
                
                annotation = annotations[condition].get(uid)
                if annotation is None:
                    continue
                
                # Process each hand in the image
                for i, bbox in enumerate(annotation.get('bboxes', [])):
                    x1 = int(bbox[0] * img.shape[1])
                    y1 = int(bbox[1] * img.shape[0])
                    w = int(bbox[2] * img.shape[1])
                    h = int(bbox[3] * img.shape[0])
                    x2 = x1 + w
                    y2 = y1 + h
                    
                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    
                    preprocessed = preprocess_image(crop, color_channels, size)
                    
                    label = annotation['labels'][i]
                    if label not in label_names:
                        label_names.append(label)
                    
                    label_index = label_names.index(label)
                    images.append(preprocessed)
                    labels.append(label_index)
                    
            except Exception as e:
                continue
    
    return images, labels, label_names


def build_model(num_classes, img_size, color_channels):
    """Build CNN model"""
    model = Sequential()
    
    # Data augmentation
    model.add(RandomFlip('horizontal', input_shape=(img_size, img_size, color_channels)))
    model.add(RandomContrast(0.1))
    
    # Convolutional layers
    model.add(Conv2D(64, kernel_size=(9, 9), activation='leaky_relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(4, 4), padding='same'))
    
    model.add(Conv2D(32, (5, 5), activation='leaky_relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3), padding='same'))
    
    model.add(Conv2D(32, (3, 3), activation='leaky_relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    
    model.add(Dropout(0.2))
    
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=categorical_crossentropy, optimizer="adam", metrics=['accuracy'])
    return model


def train_model(images, labels, label_names, model_path):
    """Train the gesture recognition model"""
    print("\n=== Training Model ===")
    
    # Prepare data
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )
    
    X_train = np.array(X_train).astype('float32') / 255.0
    X_test = np.array(X_test).astype('float32') / 255.0
    
    y_train_one_hot = to_categorical(y_train)
    y_test_one_hot = to_categorical(y_test)
    
    X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, COLOR_CHANNELS)
    X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, COLOR_CHANNELS)
    
    print(f"Training data: {X_train.shape}, Test data: {X_test.shape}")
    
    # Build and train model
    num_classes = len(label_names)
    model = build_model(num_classes, IMG_SIZE, COLOR_CHANNELS)
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)
    stop_early = EarlyStopping(monitor='val_loss', patience=3)
    
    history = model.fit(
        X_train, y_train_one_hot,
        batch_size=8,
        epochs=50,
        verbose=1,
        validation_data=(X_test, y_test_one_hot),
        callbacks=[reduce_lr, stop_early]
    )
    
    # Save model
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test_one_hot)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    return model, label_names


def load_model(model_path, label_names_path=None):
    """Load pre-trained model"""
    if not os.path.exists(model_path):
        return None, None
    
    model = tf.keras.models.load_model(model_path)
    label_names = CONDITIONS  # Default to CONDITIONS if no mapping available
    
    return model, label_names


# ============================================================================
# REAL-TIME GESTURE RECOGNITION
# ============================================================================

class GestureRecognizer:
    """Real-time gesture recognition from camera"""
    
    def __init__(self, model, label_names):
        self.model = model
        self.label_names = label_names
        self.frame_count = 0
        self.last_prediction = None
        self.no_gesture_counter = 0
        
    def preprocess_frame(self, frame, crop_region=None):
        """Preprocess a video frame"""
        if crop_region:
            x1, y1, x2, y2 = crop_region
            frame = frame[y1:y2, x1:x2]
        
        preprocessed = preprocess_image(frame, COLOR_CHANNELS, SIZE)
        return preprocessed
    
    def predict(self, frame, crop_region=None):
        """Predict gesture from frame"""
        self.frame_count += 1
        
        # Skip frames for latency optimization
        if self.frame_count % INFERENCE_SKIP_FRAMES != 0:
            return self.last_prediction
        
        preprocessed = self.preprocess_frame(frame, crop_region)
        
        # Prepare for model prediction
        input_data = np.array([preprocessed]).astype('float32') / 255.0
        input_data = input_data.reshape(-1, IMG_SIZE, IMG_SIZE, COLOR_CHANNELS)
        
        # Get prediction
        prediction = self.model.predict(input_data, verbose=0)
        confidence = np.max(prediction)
        predicted_label = np.argmax(prediction)
        
        if predicted_label in self.label_names:
            gesture = self.label_names[predicted_label]
        
        # Only register gesture if confidence is high enough
        if predicted_label in self.label_names and confidence >= CONFIDENCE_THRESHOLD:
            self.last_prediction = (gesture, confidence)
            self.no_gesture_counter = 0
        else:
            self.no_gesture_counter += 1
            if self.no_gesture_counter > NO_GESTURE_FRAMES:
                self.last_prediction = None
        
        return self.last_prediction


class MediaControlHandler:
    """Handle media control commands"""
    
    def __init__(self):
        if PYNPUT_AVAILABLE:
            self.controller = MediaController()
        else:
            self.controller = None
        self.last_action_time = 0
        self.action_cooldown = 0.5  # Minimum time between actions (seconds)
    
    def execute_action(self, action):
        """Execute media control action"""
        current_time = time.time()
        
        # Prevent action spam with cooldown
        if current_time - self.last_action_time < self.action_cooldown:
            return
        
        if not PYNPUT_AVAILABLE:
            print(f"[Would execute] {action}")
            return
        
        try:
            if action == 'volume_up':
                self.controller.volume_up()
                print(f"↑ Volume Up")
            elif action == 'volume_down':
                self.controller.volume_down()
                print(f"↓ Volume Down")
            elif action == 'pause':
                self.controller.pause()
                print(f"⏸ Pause/Play")
            
            self.last_action_time = current_time
        except Exception as e:
            print(f"Error executing action: {e}")


# ============================================================================
# MAIN REAL-TIME CONTROLLER
# ============================================================================

def run_media_controller(model, label_names, use_fixed_roi=True):
    """Run real-time media controller from camera"""
    
    if not PYNPUT_AVAILABLE:
        print("\nNote: pynput not available. Actions will only be printed to console.")
    
    recognizer = GestureRecognizer(model, label_names)
    media_handler = MediaControlHandler()
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties for faster capture
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n=== Media Controller Running ===")
    print("Instructions:")
    print("  - Position your hand in the center region of the screen")
    print("  - 'like' gesture -> Volume Up")
    print("  - 'dislike' gesture -> Volume Down")
    print("  - 'stop' gesture -> Pause/Play")
    print("  - Press 'q' to quit")
    print("  - Press 'c' to recalibrate ROI")
    print()
    
    # Define fixed ROI (center region)
    frame_width = 640
    frame_height = 480
    roi_width = 200
    roi_height = 200
    roi_x1 = (frame_width - roi_width) // 2
    roi_y1 = (frame_height - roi_height) // 2
    roi_x2 = roi_x1 + roi_width
    roi_y2 = roi_y1 + roi_height
    
    crop_region = (roi_x1, roi_y1, roi_x2, roi_y2)
    
    fps_clock = time.time()
    fps_counter = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Mirror frame for better UX
            frame = cv2.flip(frame, 1)
            
            # Get prediction
            prediction = recognizer.predict(frame, crop_region)
            
            # Draw ROI
            cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
            
            # Display prediction and action
            if prediction:
                gesture, confidence = prediction
                action = GESTURE_TO_ACTION.get(gesture, 'unknown')
                
                text = f"{gesture}: {confidence:.2f}"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2)
                
                # Execute action
                if action != 'unknown':
                    media_handler.execute_action(action)
            else:
                cv2.putText(frame, "No gesture detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
            
            # FPS counter
            fps_counter += 1
            if time.time() - fps_clock > 1:
                fps = fps_counter
                fps_counter = 0
                fps_clock = time.time()
            else:
                fps = 0
            
            cv2.putText(frame, f"FPS: {fps}", (frame_width - 100, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('Media Controller', frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exiting...")
                break
            elif key == ord('c'):
                print("Calibration mode - draw bounding box for gesture region")
                print("(Feature not implemented yet, using fixed ROI)")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main function"""
    print("=== Hand Gesture Media Controller ===\n")
    
    # Check if model exists
    model_exists = os.path.exists(MODEL_PATH)
    
    if model_exists:
        print(f"Found existing model at {MODEL_PATH}")
        response = input("Load existing model? (y/n): ").lower()
        
        if response == 'y':
            print("Loading model...")
            model, label_names = load_model(MODEL_PATH)
            
            if model and label_names:
                print("Model loaded successfully")
                run_media_controller(model, label_names)
                return
            else:
                print("Failed to load model")
        
        print("Training new model...")
    else:
        print("No existing model found. Training new model...")
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        print("Please ensure gesture_dataset_sample is in the parent directory")
        return
    
    # Load and train
    print("\nLoading dataset...")
    images, labels, label_names = load_dataset(
        CONDITIONS, DATASET_PATH, SIZE, COLOR_CHANNELS
    )
    
    if not images:
        print("Error: No images loaded. Check dataset path.")
        return
    
    print(f"Loaded {len(images)} images with {len(label_names)} gesture classes")
    
    # Train model
    model, label_names = train_model(images, labels, label_names, MODEL_PATH)
    
    # Run controller
    run_media_controller(model, label_names)


if __name__ == "__main__":
    main()
