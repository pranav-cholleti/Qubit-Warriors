import cv2
import mediapipe as mp
import math
import time
import streamlit as st
import numpy as np
from PIL import Image
import google.generativeai as genai
import io
import json  # Import the json module
import random
import re
import string
# 1. Streamlit UI Setup
st.title("Hyderabad Metro Gesture Control - Contactless & AI")

# Metro Station Data
METRO_DATA = {
    "MGBS": (17.3625, 78.4747),
    "JNTU": (17.4964, 78.3800),
    "Parade Ground": (17.4115, 78.4803),
    "Hitech City": (17.4484, 78.3784),
    "RTC X Road": (17.4412, 78.5048),
    "Ameerpet": (17.4153, 78.4477),
    "LB Nagar": (17.3518, 78.5639),
    "Nagole": (17.3923, 78.5983),
}

STATIONS = list(METRO_DATA.keys())

# --- UI Configuration ---
# Adjust these values to control button and text sizes
BUTTON_WIDTH_PERCENT = 0.20  # 20% of screen width
BUTTON_HEIGHT_PERCENT = 0.07  # 7% of screen height
BUTTON_SPACING_PERCENT = 0.02  # 2% of screen height
FONT_SCALE = 0.7  # Font scale for text on buttons
FONT_THICKNESS = 2  # Font thickness

# States
class AppState:
    IDLE = -1
    SELECTING_START = 0
    SELECTED_START = 1
    SELECTING_END = 2
    SELECTED_END = 3
    SHOW_FARE = 4
    CONFIRM_PAYMENT = 5
    PAYMENT_DONE = 6

if 'app_state' not in st.session_state:
    st.session_state.app_state = AppState.IDLE

if 'start_station' not in st.session_state:
    st.session_state.start_station = None

if 'end_station' not in st.session_state:
    st.session_state.end_station = None

if 'fare' not in st.session_state:
    st.session_state.fare = 0.0

if 'payment_confirmed' not in st.session_state:
    st.session_state.payment_confirmed = False

if 'station_info' not in st.session_state:
    st.session_state.station_info = None

# Placeholders
col1, col2 = st.columns([0.6, 0.4])

with col2:
    suggested_places_placeholder = st.empty()
    payment_complete_placeholder = st.empty()
    message_placeholder = st.empty()
    fare_placeholder = st.empty()
    station_info_placeholder = st.empty()  # Add placeholder for displaying station info

# 2. GenAI Setup
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash')  # Specify the desired model!
except KeyError:
    st.error("Please set the GEMINI_API_KEY secret in Streamlit.")
    st.stop()

def get_station_info(station_name):
    try:
        prompt = f"Provide user-friendly information about {station_name} metro station in Hyderabad in natural language. Include a very short description, a list of 2-3 nearby attractions with their distances along with nearby metros from list: METRO_DATA = MGBS,JNTU,Parade Ground,Hitech City,RTC X Road,Ameerpet, LB Nagar, Nagole, and a list of connecting metro lines. Format the information for easy reading, not as code or JSON. Be concise."

        response = model.generate_content(prompt)
        station_info_text = response.text.strip()

        print(f"Station Info from GenAI: {station_info_text}")

        return station_info_text  # Return the natural language text
    except Exception as e:
        return f"Error getting station info from GenAI: {e}"

def suggest_places(station_name, time_of_day="day"):
    prompt = f"Suggest 3 interesting places to visit near {station_name} metro station in Hyderabad, considering it is {time_of_day}. Provide brief(short) descriptions and approximate distances."
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error getting place suggestions: {e}"

# 3. Fare Calculation (GenAI Enhanced)
import streamlit as st
import time

# Add a decorator to track function execution time
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds")
        return result
    return wrapper

# Apply the timer decorator to functions you want to measure
@timer
def calculate_fare(start_station, end_station):
    try:
        prompt = f"Calculate the approximate Hyderabad Metro fare for a journey from {start_station} to {end_station}. Consider factors like distance and assume it is not peak hour.  Provide ONLY the fare amount in INR (e.g., '45')."
        response = model.generate_content(prompt)
        fare_text = response.text.lower().strip()

        # Improved parsing using regular expressions
        match = re.search(r"(\d+)", fare_text)  # Extract digits from the string
        if match:
            try:
                fare_value = float(match.group(1))
                return fare_value
            except ValueError:
                return "Error: Could not convert extracted fare to a number."
        else:
            return "Error: No fare amount found in GenAI response."

    except Exception as e:
        return f"Error calculating fare with GenAI: {e}"

# 4. OpenCV and MediaPipe Setup
mp_hands = mp.solutions.hands
max_num_hands = 1
min_detection_confidence = 0.5
min_tracking_confidence = 0.5
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=min_detection_confidence,
    min_tracking_confidence=min_tracking_confidence
)
drawLandmark = mp.solutions.drawing_utils
landmark_spec = drawLandmark.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3)
connection_spec = drawLandmark.DrawingSpec(color=(180, 180, 180), thickness=2)

# 5. Gesture and Click Detection (Paint App Style)
last_click_time = time.time()  # Initialize last click time

def set_cooldown_period():
   return 0.8

def disx(pt1, pt2):
    """Calculates the distance between two points."""
    x1, y1 = pt1
    x2, y2 = pt2
    return round(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), 3)

def detect_finger_click(hand_landmarks, width, height):
    """Detects a finger click based on distance between fingertip and thumb tip.
       Returns the midpoint coordinates regardless of button location.
       If no click detected, returns None.
    """
    index_finger_tip = hand_landmarks.landmark[8]
    thumb_tip = hand_landmarks.landmark[4]

    ix8, iy8 = int(index_finger_tip.x * width), int(index_finger_tip.y * height)
    ix4, iy4 = int(thumb_tip.x * width), int(thumb_tip.y * height)

    distance = disx((ix8, iy8), (ix4, iy4))
    midpoint = ((ix8 + ix4) // 2, (iy8 + iy4) // 2)

    click_threshold = 25
    if distance < click_threshold:
        return midpoint  # Return the midpoint even if not over a button
    else:
        return None

def handle_button_click(click_x, click_y, frame, current_state, frame_width, frame_height):
    """Handles button clicks based on the *provided* click coordinates."""
    global last_click_time

    current_time = time.time()
    cooldown_period = set_cooldown_period()
    if current_time - last_click_time > cooldown_period:  # Check cooldown
        last_click_time = current_time  # Update last click time

        # Check each button based on the click_x and click_y provided
        if current_state in [AppState.IDLE, AppState.PAYMENT_DONE]:
            book_ticket_button_coords = get_scaled_book_ticket_button_coords(frame_width, frame_height)
            if is_point_in_button(click_x, click_y, book_ticket_button_coords):  # Use the provided coordinates
                reset_state()
                st.session_state.app_state = AppState.SELECTING_START
                message_placeholder.info("Select a starting station.")
                return

        elif current_state == AppState.SELECTING_START:
            for station, coords in get_scaled_buttons(frame_width, frame_height).items():
                if is_point_in_button(click_x, click_y, coords):  # Use the provided coordinates
                    st.session_state.start_station = station
                    st.session_state.app_state = AppState.SELECTED_START
                    message_placeholder.info(f"Starting station selected: {st.session_state.start_station}. Select end station.")

                    st.session_state.station_info = get_station_info(st.session_state.start_station)
                    if isinstance(st.session_state.station_info, str):
                        station_info_placeholder.error(st.session_state.station_info)
                        st.session_state.station_info = None
                    else:
                        station_info_placeholder.markdown(st.session_state.station_info)
                    return

        elif current_state == AppState.SELECTED_START:
            st.session_state.app_state = AppState.SELECTING_END
            message_placeholder.info(f"Select the destination station")
            return

        elif current_state == AppState.SELECTING_END:
            for station, coords in get_scaled_buttons(frame_width, frame_height).items():
                if is_point_in_button(click_x, click_y, coords) and station != st.session_state.start_station:  # Use the provided coordinates
                    st.session_state.end_station = station
                    st.session_state.app_state = AppState.SELECTED_END
                    message_placeholder.info(f"Ending station selected: {st.session_state.end_station}. Calculating fare...")
                    st.session_state.fare = calculate_fare(st.session_state.start_station, st.session_state.end_station)
                    if isinstance(st.session_state.fare, str):
                        fare_placeholder.error(st.session_state.fare)
                        st.session_state.fare = 0.0
                        st.session_state.app_state = AppState.SELECTING_END
                    else:
                        st.session_state.app_state = AppState.SHOW_FARE
                        fare_placeholder.info(f"The calculated fare from {st.session_state.start_station} to {st.session_state.end_station} is INR {st.session_state.fare:.2f}. Confirm payment.")
                    return

        elif current_state == AppState.SHOW_FARE:
            payment_button_coords = get_scaled_payment_button_coords(frame_width, frame_height)
            if is_point_in_button(click_x, click_y, payment_button_coords):  # Use the provided coordinates
                st.session_state.payment_confirmed = True
                st.session_state.app_state = AppState.PAYMENT_DONE

                def generate_ticket_id():
                    letters = ''.join(random.choices(string.ascii_uppercase, k=3))
                    numbers = ''.join(random.choices(string.digits, k=4))
                    return letters + numbers

                ticket_id = generate_ticket_id()
                payment_complete_placeholder.success(f"Payment Successful! Enjoy your ride.\nTicket ID: {ticket_id}")
            return
        elif current_state == AppState.PAYMENT_DONE:
            book_ticket_button_coords = get_scaled_book_ticket_button_coords(frame_width, frame_height)
            if is_point_in_button(click_x, click_y, book_ticket_button_coords):  # Use the provided coordinates
                reset_state()
                st.session_state.app_state = AppState.SELECTING_START
                message_placeholder.info("Select a starting station.")
            return

        suggest_button_coords = get_scaled_suggest_button_coords(frame_width, frame_height)
        if is_point_in_button(click_x, click_y, suggest_button_coords) and st.session_state.start_station is not None:
            place_suggestions = suggest_places(st.session_state.start_station)
            suggested_places_placeholder.markdown(place_suggestions)

        reset_button_coords = get_scaled_reset_button_coords(frame_width, frame_height)
        if is_point_in_button(click_x, click_y, reset_button_coords):
            reset_state()

def is_point_in_button(x, y, button_coords):
    x1, y1, x2, y2 = button_coords
    return x1 <= x <= x2 and y1 <= y <= y2

def reset_state():
    st.session_state.app_state = AppState.IDLE
    st.session_state.start_station = None
    st.session_state.end_station = None
    st.session_state.fare = 0.0
    st.session_state.payment_confirmed = False
    st.session_state.station_info = None
    message_placeholder.empty()
    fare_placeholder.empty()
    payment_complete_placeholder.empty()
    station_info_placeholder.empty()
    suggested_places_placeholder.empty()

def get_scaled_button_dimensions(frame_width, frame_height):
    """Calculates button dimensions based on percentages of screen size."""
    button_width = int(frame_width * BUTTON_WIDTH_PERCENT)
    button_height = int(frame_height * BUTTON_HEIGHT_PERCENT)
    button_spacing = int(frame_height * BUTTON_SPACING_PERCENT)
    button_start_x = int(frame_width * 0.05)  # 5% from the left
    button_start_y = int(frame_height * 0.1)  # 10% from the top
    return button_width, button_height, button_spacing, button_start_x, button_start_y

def get_scaled_buttons(frame_width, frame_height):
    """Calculates the scaled button coordinates."""
    button_width, button_height, button_spacing, button_start_x, button_start_y = get_scaled_button_dimensions(frame_width, frame_height)
    buttons = {}
    for i, station in enumerate(STATIONS):
        x1 = button_start_x
        y1 = button_start_y + i * (button_height + button_spacing)
        x2 = x1 + button_width
        y2 = y1 + button_height
        buttons[station] = (x1, y1, x2, y2)
    return buttons

def get_scaled_payment_button_coords(frame_width, frame_height):
    """Calculates the scaled payment button coordinates."""
    button_width, button_height, button_spacing, button_start_x, button_start_y = get_scaled_button_dimensions(frame_width, frame_height)
    payment_button_x1 = int(frame_width * 0.05)  # 5% from the left
    payment_button_y1 = button_start_y + len(STATIONS) * (button_height + button_spacing) + int(frame_height * 0.02)
    payment_button_x2 = payment_button_x1 + button_width
    payment_button_y2 = payment_button_y1 + button_height
    return (payment_button_x1, payment_button_y1, payment_button_x2, payment_button_y2)

def get_scaled_suggest_button_coords(frame_width, frame_height):
    """Calculates the scaled suggest button coordinates."""
    button_width, button_height, button_spacing, button_start_x, button_start_y = get_scaled_button_dimensions(frame_width, frame_height)
    suggest_button_x1 = int(frame_width * 0.30)  # 30% from the left
    suggest_button_y1 = int(frame_height * 0.1)  # 10% from the top
    suggest_button_x2 = suggest_button_x1 + button_width
    suggest_button_y2 = suggest_button_y1 + button_height
    return (suggest_button_x1, suggest_button_y1, suggest_button_x2, suggest_button_y2)

def get_scaled_reset_button_coords(frame_width, frame_height):
    """Calculates the scaled reset button coordinates."""
    button_width, button_height, button_spacing, button_start_x, button_start_y = get_scaled_button_dimensions(frame_width, frame_height)
    payment_button_x1 = int(frame_width * 0.05)  # 5% from the left
    payment_button_y1 = button_start_y + len(STATIONS) * (button_height + button_spacing) + int(frame_height * 0.02)
    reset_button_y1 = payment_button_y1 + button_height + button_spacing
    reset_button_x1 = payment_button_x1
    reset_button_x2 = reset_button_x1 + button_width
    reset_button_y2 = reset_button_y1 + button_height
    return (reset_button_x1, reset_button_y1, reset_button_x2, reset_button_y2)

def get_scaled_book_ticket_button_coords(frame_width, frame_height):
    """Calculates the scaled "Book Ticket" button coordinates."""
    button_width, button_height, button_spacing, button_start_x, button_start_y = get_scaled_button_dimensions(frame_width, frame_height)
    book_ticket_x1 = int(frame_width * 0.05)  # 5% from the left
    book_ticket_y1 = int(frame_height * 0.1)  # 10% from the top
    book_ticket_x2 = book_ticket_x1 + button_width
    book_ticket_y2 = book_ticket_y1 + button_height
    return (book_ticket_x1, book_ticket_y1, book_ticket_x2, book_ticket_y2)

# 6. Main Loop
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Error: Could not open webcam.")
    st.stop()

with col1:
    frame_placeholder = st.empty()

while True:
    stat, frame = cap.read()
    if not stat:
        print("Error: Couldn't read frame.")
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # Calculate scaled button dimensions and positions
    button_width, button_height, button_spacing, button_start_x, button_start_y = get_scaled_button_dimensions(width, height)
    scaled_buttons = get_scaled_buttons(width, height)
    payment_button_coords = get_scaled_payment_button_coords(width, height)
    suggest_button_coords = get_scaled_suggest_button_coords(width, height)
    reset_button_coords = get_scaled_reset_button_coords(width, height)
    book_ticket_button_coords = get_scaled_book_ticket_button_coords(width, height) # Get the "Book Ticket" button coordinates


    # Draw UI elements based on app state
    if st.session_state.app_state in [AppState.SELECTING_START, AppState.SELECTED_START, AppState.SELECTING_END]:
        for station, coords in scaled_buttons.items():
            x1, y1, x2, y2 = coords
            color = (0, 255, 0) if (st.session_state.app_state == AppState.SELECTED_START and station == st.session_state.start_station) else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            # Use calculated font scale and thickness
            cv2.putText(frame, station, (x1 + int(button_width * 0.05), y1 + int(button_height * 0.6)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)

        #Suggest Place button
        x1, y1, x2, y2 = suggest_button_coords
        color = (255, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.putText(frame, "Suggest Places", (x1 + int(button_width * 0.05), y1 + int(button_height * 0.6)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), FONT_THICKNESS)  # Black text

    # Draw the "Book Ticket" button in IDLE and PAYMENT_DONE states
    if st.session_state.app_state in [AppState.IDLE, AppState.PAYMENT_DONE]:
        x1, y1, x2, y2 = book_ticket_button_coords
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), -1)
        cv2.putText(frame, "Book Ticket", (x1 + int(button_width * 0.05), y1 + int(button_height * 0.6)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), FONT_THICKNESS)

    if st.session_state.app_state == AppState.SHOW_FARE:
        x1, y1, x2, y2 = payment_button_coords
        color = (0, 255, 0)  # Green
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.putText(frame, "Confirm Payment", (x1 + int(button_width * 0.05), y1 + int(button_height * 0.6)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), FONT_THICKNESS)  # Black text

    # Reset Button
    if st.session_state.app_state != AppState.IDLE and st.session_state.app_state != AppState.PAYMENT_DONE:
        x1, y1, x2, y2 = reset_button_coords
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), -1)  # Red for Reset
        cv2.putText(frame, "Reset", (x1 + int(button_width * 0.05), y1 + int(button_height * 0.6)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)

    rgb = image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            drawLandmark.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          landmark_drawing_spec=landmark_spec,
                                          connection_drawing_spec=connection_spec)

            # Detect finger click
            click_coordinates = detect_finger_click(hand_landmarks, width, height)
            if click_coordinates:
                cx, cy = click_coordinates
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)  # Green click indicator
                handle_button_click(cx, cy, frame, st.session_state.app_state, width, height)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB")

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()