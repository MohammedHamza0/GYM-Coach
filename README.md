# GYM Coach: Exercise Detection and Learning App

This project is a Streamlit-based application that helps users detect their exercise form in real-time using computer vision and pose estimation models. It currently supports push-ups, squats, and pull-ups, providing both real-time feedback and educational content on how to perform each exercise correctly.

## Features

- **Exercise Detection**: Automatically detects push-ups, squats, and pull-ups from uploaded videos using YOLOv8 pose estimation.
- **Real-Time Feedback**: Provides live feedback on whether you’re in the "up" or "down" position during exercises, along with a counter to track your reps.
- **Learning Section**: Educational content and video tutorials on how to correctly perform each exercise.
- **AI Agent**: A conversational agent provides additional guidance and answers exercise-related questions using GroqModel integration.

## Built With

- **YOLOv8**: For pose estimation and exercise detection.
- **Streamlit**: For building the user interface.
- **OpenCV**: For video processing and frame-by-frame analysis.
- **GroqModel**: For AI-powered conversational guidance.
- **Python**: The core programming language for the app.

## How It Works

1. **Upload Your Video**: Select the exercise type (Pushups, Squats, or Pullups) and upload a video.
2. **Real-Time Detection**: The app uses pose estimation to detect key joints, calculate angles, and determine exercise status (e.g., "up" or "down").
3. **Exercise Guidance**: The built-in AI agent offers advice based on the selected exercise and user inputs.

## Installation

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Set up the environment variables:
   - Create a `.env` file and add your `GROQ_API_KEY`:
     ```
     GROQ_API_KEY=your_api_key_here
     ```

3. Run the app:
    ```bash
    streamlit run app.py
    ```

## Usage

1. Navigate to the app interface in your browser.
2. Select the type of exercise (Pushups, Squats, or Pullups).
3. Upload a video of yourself performing the exercise.
4. Get real-time feedback on your performance and track your reps.
5. Optionally, learn the correct way to perform exercises through the educational section.


## Contact

Designed and developed by [Mohammed Hamza](https://www.linkedin.com/in/mohammed-hamza-4184b2251/). Connect with me on [LinkedIn](https://www.linkedin.com/in/mohammed-hamza-4184b2251/).

---

Feel free to modify this to suit your preferences!
