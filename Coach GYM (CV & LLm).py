import os
import cv2
import math
import tempfile
import streamlit as st
from ultralytics import YOLO
from dotenv import load_dotenv
from swarmauri.standard.llms.concrete.GroqModel import GroqModel
from swarmauri.standard.messages.concrete.SystemMessage import SystemMessage
from swarmauri.standard.agents.concrete.SimpleConversationAgent import SimpleConversationAgent
from swarmauri.standard.conversations.concrete.MaxSystemContextConversation import MaxSystemContextConversation

# Load environment variables
load_dotenv(dotenv_path=r'C:\Users\mhmdh\Yolo\.env')

API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    st.sidebar.error("API key not found. Please ensure the environment variable GROQ_API_KEY is set.")
# else:
#     st.sidebar.write("API key loaded successfully.")

# Initialize AI model and conversation
llm = GroqModel(api_key=API_KEY)
allowed_models = llm.allowed_models
conversation = MaxSystemContextConversation()

def load_model(selected_model):
    """Load the selected model using the API key."""
    return GroqModel(api_key=API_KEY, name=selected_model)

def is_repetitive_response(response, history):
    """Check if the current response is repetitive compared to the conversation history."""
    if history:
        last_response = history[-1]['bot']
        return response.strip().lower() == last_response.strip().lower()
    return False

def converse(input_text, history, system_context, model_name):
    """Perform conversation with the selected model and given system context."""
    st.sidebar.write(f"System context: {system_context}")
    st.sidebar.write(f"Selected model: {model_name}")

    llm = load_model(model_name)
    agent = SimpleConversationAgent(llm=llm, conversation=conversation)
    agent.conversation.system_context = SystemMessage(content=system_context)

    input_text = str(input_text)
    st.sidebar.write("Conversation history:", conversation.history)

    try:
        result = agent.exec(input_text)
        st.sidebar.write("Raw result:", result)

        response_text = str(result)

        
        if is_repetitive_response(response_text, history):
            return "It seems you've already received this answer. Is there something else you'd like to know?"

        return response_text

    except ValueError as ve:
        st.sidebar.error(f"JSON Parsing Error: {ve}")
        return "Error: Received response that is not valid JSON."
    
    except ConnectionError as ce:
        st.sidebar.error(f"Connection Error: {ce}")
        return "Error: Unable to connect to the server."
    
    except Exception as e:
        st.sidebar.error(f"Error: {e}")
        return f"Error occurred: {str(e)}"
    

# calc the fucken angle 
def calculate_angle(pointA, pointB, pointC):
    vectorAB = [pointA[0] - pointB[0], pointA[1] - pointB[1]]
    vectorCB = [pointC[0] - pointB[0], pointC[1] - pointB[1]]

    dot_product = vectorAB[0] * vectorCB[0] + vectorAB[1] * vectorCB[1]
    magnitudeAB = math.sqrt(vectorAB[0] ** 2 + vectorAB[1] ** 2)
    magnitudeCB = math.sqrt(vectorCB[0] ** 2 + vectorCB[1] ** 2)

    angle_rad = math.acos(dot_product / (magnitudeAB * magnitudeCB + 1e-6))
    angle_deg = math.degrees(angle_rad)
    return angle_deg


# Initialize session state for the learning section and chatbot
if "show_learning_section" not in st.session_state:
    st.session_state.show_learning_section = False

if "history" not in st.session_state:
    st.session_state.history = []

# Streamlit UI for Exercise Detection App
st.title("GYM Coach:Exercise Detection and Learning App")

# Chatbot Section - Always Visible
st.sidebar.title("AI Agent: Exercise Guidance")
st.sidebar.write("Chat with an agent to get guidance on performing the perfect Exercise.")

# Chatbot Input
system_context = st.sidebar.selectbox("System context", ("Push-up Guidance", "Squats Guidance", "Pullup Guidance")) 
model_name = st.sidebar.selectbox("Model name", allowed_models, index=0)
input_text = st.sidebar.text_input("Ask about your exercise")

# Button to interact with the AI agent
if st.sidebar.button("Ask AI"):
    if input_text and system_context and model_name:
        response = converse(input_text, st.session_state.history, system_context, model_name)
        st.sidebar.write(response)
        st.session_state.history.append({"user": input_text, "bot": response})

# Display conversation history
if st.sidebar.button("Show History"):
    for entry in st.session_state.history:
        st.sidebar.write(f"User: {entry['user']}")
        st.sidebar.write(f"AI: {entry['bot']}")

# Section for Exercise Detection
st.subheader("Exercise Detection Section")
st.write("Upload your video to detect exercises and get real-time feedback.")

# Upload video for exercise detection
exercise_type = st.selectbox("Choose the exercise", ("Pushups", "Squats", "pullup"))
counter = 0
down_counter = False
was_down = False

if exercise_type == "Pushups" or exercise_type == "Squats" or exercise_type == "pullup":
    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        model = YOLO("yolov8s-pose.pt")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1100, 700))
            results = model.predict(frame, conf=0.7)

            for result in results:
                boxes = result.boxes.xyxy
                keypoints = result.keypoints.xy

                for box, points in zip(boxes, keypoints):
                    x, y, w, h = map(int, box)
                    if exercise_type == "Pushups":
                        left_shoulder = points[5]
                        left_elbow = points[7]
                        left_wrist = points[9]
                        exercise_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                        angle_threshold = 150  

                        # Draw the points for push-ups
                        for point in [left_shoulder, left_elbow, left_wrist]:
                            cv2.circle(frame, (int(point[0]), int(point[1])), 5, [0, 0, 255], -1)
                            
                        cv2.line(frame, (int(left_shoulder[0]), int(left_shoulder[1])), 
                         (int(left_elbow[0]), int(left_elbow[1])), [255, 0, 0], 1)
                
                        cv2.line(frame, (int(left_elbow[0]), int(left_elbow[1])), 
                                (int(left_wrist[0]), int(left_wrist[1])), [255, 0, 0], 1)

                    elif exercise_type == "Squats":
                        left_hip = points[11]
                        left_knee = points[13]
                        left_ankle = points[15]
                        exercise_angle = calculate_angle(left_hip, left_knee, left_ankle)
                        angle_threshold = 140  

                        # Draw the points for squats
                        for point in [left_hip, left_knee, left_ankle]:
                            cv2.circle(frame, (int(point[0]), int(point[1])), 5, [0, 0, 255], -1)
                            
                        cv2.line(frame, (int(left_hip[0]), int(left_hip[1])), 
                         (int(left_knee[0]), int(left_knee[1])), [255, 0, 0], 1)
                
                        cv2.line(frame, (int(left_knee[0]), int(left_knee[1])), 
                                (int(left_ankle[0]), int(left_ankle[1])), [255, 0, 0], 1)
                        
                        
                    elif exercise_type == "pullup":
                        right_wrist = points[10]
                        right_elbow = points[8]
                        right_shoulder = points[6]
                        exercise_angle = calculate_angle(right_wrist, right_elbow, right_shoulder)
                        angle_threshold = 90  

                        # Draw the points for pull-ups
                        for point in [right_wrist, right_elbow, right_shoulder]:
                            cv2.circle(frame, (int(point[0]), int(point[1])), 5, [0, 0, 255], -1)
                            
                        cv2.line(frame, (int(right_wrist[0]), int(right_wrist[1])), 
                         (int(right_elbow[0]), int(right_elbow[1])), [255, 0, 0], 1)
                
                        cv2.line(frame, (int(right_elbow[0]), int(right_elbow[1])), 
                                (int(right_shoulder[0]), int(right_shoulder[1])), [255, 0, 0], 1)



                    current_down = False
                    if exercise_type == "Pushups" or exercise_type == "Squats":
                        if exercise_angle < angle_threshold :
                            cv2.putText(frame, "Activity: Down", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            current_down = True
                            if not was_down:
                                was_down = True
                                down_counter = False
                            elif not down_counter:
                                counter += 1
                                down_counter = True
                        elif exercise_angle > angle_threshold:
                            cv2.putText(frame, "Activity: Up", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        if not current_down:
                            was_down = False
                            down_counter = False
                    elif exercise_type == "pullup":
                        if exercise_angle < angle_threshold :
                            cv2.putText(frame, "Activity: Up", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            current_down = True
                            if not was_down:
                                was_down = True
                                down_counter = False
                            elif not down_counter:
                                counter += 1
                                down_counter = True
                        elif exercise_angle > angle_threshold:
                            cv2.putText(frame, "Activity: Down", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        if not current_down:
                            was_down = False
                            down_counter = False

            cv2.putText(frame, f"{exercise_type} counter: " + str(counter), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, [0, 0, 255], 3)
            stframe.image(frame, channels="BGR", use_column_width=True)

        cap.release()

# Button to show learning content
if st.button("Learn the Correct Way to Perform the Exercise"):
    st.session_state.show_learning_section = not st.session_state.show_learning_section

# Show the learning section if the state is True
if st.session_state.show_learning_section:
    st.subheader(f"Learn the Correct Way to Perform {exercise_type}")

    if exercise_type == "Pushups":
        st.write("""
        ### Steps for the Perfect Push-up:
        1. **Start Position**: Begin in a plank position with hands shoulder-width apart.
        2. **Lowering**: Lower your body until your chest almost touches the floor.
        3. **Pushing Back**: Push yourself back up while keeping your core engaged.
        4. **Hand Placement**: Hands should be slightly wider than shoulder-width.
        5. **Body Alignment**: Keep your body straight, core tight, and elbows tucked at about a 45-degree angle.
        """)

        # Video tutorial for push-ups
        st.video("https://www.youtube.com/watch?v=IODxDxX7oi4")

    elif exercise_type == "Squats":
        st.write("""
        ### Steps for the Perfect Squat:
        1. **Feet Position**: Stand with feet shoulder-width apart.
        2. **Lowering**: Push your hips back and lower your body until your thighs are parallel to the ground.
        3. **Knee Alignment**: Ensure your knees are in line with your toes.
        4. **Pushing Up**: Drive through your heels to return to the standing position.
        5. **Back Position**: Keep your back straight and chest up throughout the movement.
        """)

        # Video tutorial for squats
        st.video("https://www.youtube.com/watch?v=gsNoPYwWXeM")
        
        
    elif exercise_type == "pullup":
        st.write("""
        ### Steps for the Perfect Pull-up:
        1. **Grip**: Grab the bar with your hands slightly wider than shoulder-width apart, palms facing away from you.
        2. **Hang**: Allow your body to hang with arms fully extended and engage your core.
        3. **Pulling Up**: Pull your body up towards the bar by driving your elbows down.
        4. **Chin Over the Bar**: Continue pulling until your chin is above the bar.
        5. **Lowering**: Lower yourself in a controlled motion until your arms are fully extended again.
        6. **Body Control**: Avoid swinging or using momentum throughout the movement.
        """)

        # Video tutorial for pull-ups
        st.video("https://www.youtube.com/watch?v=eGo4IYlbE5g")
        
        
st.markdown("---")
st.markdown("<h3 style='color: #336699;'>Designed and developed by <span style='color: #FF5733; font-weight: bold;'>Mohammed Hamza</span></h3>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 18px; color: #666;'>Connect with me: <a href='https://www.linkedin.com/in/mohammed-hamza-4184b2251/' target='_blank'>LinkedIn</a></p>", unsafe_allow_html=True)


