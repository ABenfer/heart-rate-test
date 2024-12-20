import streamlit as st
import numpy as np
import cv2
from PIL import Image
import time
import matplotlib.pyplot as plt

# Streamlit configuration
st.title("Heart Rate Measurement App")
st.write("""
This app uses your device's camera to measure your heart rate. Place your finger over the camera lens and take a picture.
""")

# Capture image using st.camera_input
img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # Convert the image to a format suitable for analysis
    img = Image.open(img_file_buffer)
    img_array = np.array(img)

    # Placeholder for heart rate calculation logic
    heart_rate = 75  # Placeholder value

    # Display the result
    st.success(f"Your estimated heart rate is {heart_rate} BPM.")

# App parameters
MEASUREMENT_TIME = 15  # seconds
FRAME_RATE = 30  # frames per second

# Function to process frames and calculate heart rate
def process_frames(frames, fps):
    """
    Process captured frames to calculate heart rate.

    Args:
        frames (list): List of frame brightness values.
        fps (int): Frames per second.

    Returns:
        float: Calculated heart rate in beats per minute (BPM).
    """
    # Normalize brightness values
    signal = np.array(frames) - np.mean(frames)

    # Apply Fast Fourier Transform (FFT)
    fft = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(len(signal), d=1/fps)

    # Find the frequency with the highest amplitude in the range of heart rates
    heart_rate_range = (frequencies >= 0.8) & (frequencies <= 3.0)
    peak_frequency = frequencies[heart_rate_range][np.argmax(np.abs(fft)[heart_rate_range])]

    # Convert frequency to BPM
    heart_rate_bpm = peak_frequency * 60
    return heart_rate_bpm

# Main function
def main():
    # Access the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Unable to access the camera. Please check permissions and try again.")
        return

    # Prepare to capture frames
    frames = []
    start_time = time.time()
    st.write("Measuring heart rate...")
    progress_bar = st.progress(0)

    while time.time() - start_time < MEASUREMENT_TIME:
        ret, frame = cap.read()
        if not ret:
            st.error("Error reading from the camera.")
            break

        # Extract the green channel's mean brightness
        brightness = np.mean(frame[:, :, 1])  # Green channel
        frames.append(brightness)

        # Update progress bar
        progress = (time.time() - start_time) / MEASUREMENT_TIME
        progress_bar.progress(min(int(progress * 100), 100))

    cap.release()

    if len(frames) < FRAME_RATE * MEASUREMENT_TIME * 0.5:
        st.error("Not enough data collected. Please try again.")
        return

    # Calculate heart rate
    heart_rate = process_frames(frames, FRAME_RATE)
    st.success(f"Your heart rate is approximately {heart_rate:.1f} BPM.")

    # Plot brightness values
    plt.figure(figsize=(10, 4))
    plt.plot(frames, label="Brightness")
    plt.title("Brightness Over Time")
    plt.xlabel("Frame")
    plt.ylabel("Brightness")
    plt.legend()
    st.pyplot(plt)

# Run the app
if __name__ == "__main__":
    main()
