import speech_recognition as sr
import pyttsx3
import datetime
import webbrowser
import requests
import json
import os
import subprocess
import logging
import threading
import time
import re
import tkinter as tk
from tkinter import scrolledtext
from typing import Optional, Dict, List
from urllib.parse import quote
from groq import Groq

CONFIG = {
    'GROQ_MODEL': 'llama3-8b-8192',
    'GROQ_MAX_TOKENS': 2048,
    'GROQ_TEMPERATURE': 0.7,
    'CONTENT_DATA_DIR': 'Data',
    'USER_AGENT': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36",
    'BROWSER_CLOSE_CUTOFF': 0.7,
    'PROCESS_KILL_CUTOFF': 0.8,
    'AI_CHAT_HISTORY_LIMIT': 5
}

# Configure logging for debugging and tracking
logging.basicConfig(
    filename='jarvis.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# User preferences class
class UserPreferences:
    """Manage user preferences stored in a JSON file."""
    def __init__(self, file_path: str = "user_prefs.json"):
        self.file_path = file_path
        self.prefs = self.load_prefs()
    
    def load_prefs(self) -> Dict:
        """Load preferences from file or return defaults."""
        default_prefs = {
            "default_city": "New York",
            "voice_rate": 150,
            "volume_level": 0.9,
            "favorite_commands": {}
        }
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, "r") as f:
                    return json.load(f)
            logging.info("No preferences file found. Using defaults.")
            return default_prefs
        except Exception as e:
            logging.error(f"Error loading preferences: {e}")
            return default_prefs
    
    def save_prefs(self) -> None:
        """Save preferences to file."""
        try:
            with open(self.file_path, "w") as f:
                json.dump(self.prefs, f, indent=4)
            logging.info("Preferences saved.")
        except Exception as e:
            logging.error(f"Error saving preferences: {e}")
    
    def update_pref(self, key: str, value: any) -> None:
        """Update a preference and save."""
        self.prefs[key] = value
        self.save_prefs()
    
    def increment_command_usage(self, command: str) -> None:
        """Track command usage frequency."""
        self.prefs["favorite_commands"][command] = self.prefs["favorite_commands"].get(command, 0) + 1
        self.save_prefs()

# Conversation context class
class ConversationContext:
    """Track recent interactions for context-aware responses."""
    def __init__(self, max_history: int = CONFIG['AI_CHAT_HISTORY_LIMIT']):
        self.max_history = max_history
        self.history: List[Dict[str, str]] = []
    
    def add_interaction(self, user_input: str, response: str) -> None:
        """Add a user-AI interaction to history."""
        self.history.append({"user": user_input, "response": response})
        if len(self.history) > self.max_history:
            self.history.pop(0)
        logging.info(f"Added to context: {user_input} -> {response}")
    
    def get_context(self) -> str:
        """Return formatted context for AI queries."""
        return "\n".join([f"User: {item['user']}\nAssistant: {item['response']}" for item in self.history])

# Initialize text-to-speech engine
def init_tts_engine(prefs: UserPreferences) -> pyttsx3.Engine:
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', prefs.prefs["voice_rate"])
        engine.setProperty('volume', prefs.prefs["volume_level"])
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[0].id)
        logging.info("Text-to-speech engine initialized successfully.")
        return engine
    except Exception as e:
        logging.error(f"Failed to initialize TTS engine: {e}")
        print("Error: Could not initialize speech engine. Using print statements.")
        return None

# Initialize speech recognition
def init_speech_recognizer() -> sr.Recognizer:
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 4000
    recognizer.dynamic_energy_threshold = True
    logging.info("Speech recognizer initialized successfully.")
    return recognizer

# Speak text using TTS engine and update GUI
def speak(engine: pyttsx3.Engine, text: str, response_text: scrolledtext.ScrolledText = None) -> None:
    try:
        if engine:
            engine.say(text)
            engine.runAndWait()
        else:
            print(f"JARVIS: {text}")
        if response_text:
            response_text.insert(tk.END, f"JARVIS: {text}\n")
            response_text.see(tk.END)
        logging.info(f"Spoke: {text}")
    except Exception as e:
        logging.error(f"Speech error: {e}")
        print(f"Error speaking: {text}")

# Listen for voice input
def listen(recognizer: sr.Recognizer, timeout: int = 5) -> Optional[str]:
    with sr.Microphone() as source:
        try:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
            text = recognizer.recognize_google(audio).lower()
            logging.info(f"Recognized: {text}")
            return text
        except sr.WaitTimeoutError:
            logging.warning("Listening timed out.")
            return None
        except sr.UnknownValueError:
            logging.warning("Could not understand audio.")
            return None
        except sr.RequestError as e:
            logging.error(f"Speech recognition error: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error during listening: {e}")
            return None

# Query Groq AI model
def query_ai(query: str, context: ConversationContext, api_key: str) -> str:
    try:
        client = Groq(api_key=api_key)
        messages = [
            {"role": "system", "content": "You are JARVIS, a helpful AI assistant inspired by Iron Man. Provide concise, accurate, and conversational responses."}
        ]
        if context.get_context():
            messages.append({"role": "user", "content": context.get_context()})
        messages.append({"role": "user", "content": query})
        
        response = client.chat.completions.create(
            model=CONFIG['GROQ_MODEL'],
            messages=messages,
            max_tokens=CONFIG['GROQ_MAX_TOKENS'],
            temperature=CONFIG['GROQ_TEMPERATURE']
        )
        result = response.choices[0].message.content.strip()
        logging.info(f"Groq AI response: {result}")
        return result
    except Exception as e:
        logging.error(f"Groq AI query error: {e}")
        return "Sorry, I couldn't connect to the AI service. Please try again."

# Get current time
def get_time() -> str:
    now = datetime.datetime.now()
    time_str = now.strftime("%I:%M %p")
    logging.info(f"Retrieved time: {time_str}")
    return f"The current time is {time_str}."

# Get current date
def get_date() -> str:
    now = datetime.datetime.now()
    date_str = now.strftime("%B %d, %Y")
    logging.info(f"Retrieved date: {date_str}")
    return f"Today is {date_str}."

# Perform a web search
def web_search(query: str) -> str:
    try:
        encoded_query = quote(query)
        url = f"https://www.google.com/search?q={encoded_query}"
        webbrowser.open(url)
        logging.info(f"Web search performed: {query}")
        return f"Searching for {query} on Google."
    except Exception as e:
        logging.error(f"Web search error: {e}")
        return "Sorry, I couldn't perform the search."

# Get weather information
def get_weather(city: str, api_key: str, prefs: UserPreferences) -> str:
    if not city:
        city = prefs.prefs["default_city"]
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url)
        data = response.json()
        if data.get("cod") != 200:
            logging.error(f"Weather API error: {data.get('message')}")
            return "Sorry, I couldn't fetch the weather."
        weather = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        logging.info(f"Weather fetched for {city}: {weather}, {temp}°C")
        return f"The weather in {city} is {weather} with a temperature of {temp} degrees Celsius."
    except Exception as e:
        logging.error(f"Weather fetch error: {e}")
        return "Sorry, I couldn't fetch the weather."

# Set a reminder
def set_reminder(task: str, time_str: str) -> str:
    try:
        reminder_time = datetime.datetime.strptime(time_str, "%H:%M").replace(
            year=datetime.datetime.now().year,
            month=datetime.datetime.now().month,
            day=datetime.datetime.now().day
        )
        if reminder_time < datetime.datetime.now():
            reminder_time = reminder_time + datetime.timedelta(days=1)
        with open("reminders.json", "a") as f:
            json.dump({"task": task, "time": reminder_time.isoformat()}, f)
            f.write("\n")
        logging.info(f"Reminder set: {task} at {time_str}")
        return f"Reminder set for {task} at {time_str}."
    except Exception as e:
        logging.error(f"Reminder error: {e}")
        return "Sorry, I couldn't set the reminder."

# Check reminders
def check_reminders(engine: pyttsx3.Engine, response_text: scrolledtext.ScrolledText = None) -> None:
    while True:
        try:
            reminders = []
            if os.path.exists("reminders.json"):
                with open("reminders.json", "r") as f:
                    for line in f:
                        if line.strip():
                            reminders.append(json.loads(line.strip()))
            now = datetime.datetime.now()
            for reminder in reminders:
                reminder_time = datetime.datetime.fromisoformat(reminder["time"])
                if now >= reminder_time:
                    reminder_msg = f"Reminder: {reminder['task']}"
                    speak(engine, reminder_msg, response_text)
                    reminders.remove(reminder)
                    with open("reminders.json", "w") as f:
                        for rem in reminders:
                            json.dump(rem, f)
                            f.write("\n")
            time.sleep(60)
        except Exception as e:
            logging.error(f"Reminder check error: {e}")
            time.sleep(60)

# Open an application or website
def open_app(app_name: str) -> str:
    try:
        # Handle websites
        website_map = {
            "youtube": "https://www.youtube.com",
            "google": "https://www.google.com",
            "facebook": "https://www.facebook.com",
            "twitter": "https://www.twitter.com",
            "x": "https://www.x.com"
        }
        app_name_lower = app_name.lower().strip()
        for site, url in website_map.items():
            if site in app_name_lower:
                webbrowser.open(url)
                logging.info(f"Opened website: {url}")
                return f"Opening {site.capitalize()}."
        
        # Handle system applications
        if os.name == "nt":
            result = subprocess.run(f"start {app_name}", shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, f"start {app_name}")
        else:
            subprocess.run(["open", app_name], check=True)
        logging.info(f"Opened application: {app_name}")
        return f"Opening {app_name}."
    except subprocess.CalledProcessError as e:
        logging.error(f"App open error: {e}")
        return f"Sorry, I couldn't open {app_name}. It may not be installed or recognized."
    except Exception as e:
        logging.error(f"App open error: {e}")
        return f"Sorry, I couldn't open {app_name}."

# Close browser tabs or process
def close_browser(browser_name: str = "") -> str:
    try:
        if not browser_name:
            browser_name = "chrome"  # Default to Chrome
        
        # Map common browser names to process names
        browser_process_map = {
            "chrome": ["chrome", "Google Chrome"],
            "firefox": ["firefox"],
            "safari": ["safari"]
        }
        
        process_names = browser_process_map.get(browser_name.lower(), [browser_name])
        success = False
        error_msg = ""
        
        for process in process_names:
            try:
                if os.name == "nt":
                    result = subprocess.run(f"taskkill /IM {process}.exe /F", shell=True, capture_output=True, text=True)
                    if result.returncode == 0:
                        success = True
                    else:
                        error_msg += f"Failed to close {process}.exe: {result.stderr}\n"
                else:
                    result = subprocess.run(["pkill", "-9", process], capture_output=True, text=True)
                    if result.returncode == 0:
                        success = True
                    else:
                        error_msg += f"Failed to close {process}: {result.stderr}\n"
            except subprocess.CalledProcessError as e:
                error_msg += f"Error closing {process}: {e}\n"
                continue
        
        if success:
            logging.info(f"Closed browser: {browser_name}")
            return f"Closed {browser_name.capitalize()}."
        else:
            logging.error(f"Browser close error: {error_msg}")
            return f"Sorry, I couldn't close {browser_name}. It may not be running or requires admin privileges."
    except Exception as e:
        logging.error(f"Browser close error: {e}")
        return f"Sorry, I couldn't close {browser_name}. Please ensure it's running and try again."

# Control system volume
def set_volume(level: int, prefs: UserPreferences) -> str:
    try:
        if os.name == "nt":
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            volume.SetMasterVolumeLevelScalar(level / 100, None)
            prefs.update_pref("volume_level", level / 100)
            logging.info(f"Set volume to {level}%")
            return f"Volume set to {level}%."
        else:
            logging.warning("Volume control not supported on this OS.")
            return "Volume control is only supported on Windows."
    except Exception as e:
        logging.error(f"Volume control error: {e}")
        return "Sorry, I couldn't adjust the volume."

# Process user command
def process_command(
    engine: pyttsx3.Engine,
    command: str,
    weather_api_key: str,
    ai_api_key: str,
    prefs: UserPreferences,
    context: ConversationContext,
    response_text: scrolledtext.ScrolledText
) -> bool:
    if not command:
        return True

    try:
        prefs.increment_command_usage(command)
        response_text.insert(tk.END, f"User: {command}\n")
        response_text.see(tk.END)
        response = ""

        if "exit" in command or "quit" in command:
            response = "Goodbye, sir."
            speak(engine, response, response_text)
            context.add_interaction(command, response)
            logging.info("Exiting JARVIS.")
            return False

        elif "time" in command:
            response = get_time()
            speak(engine, response, response_text)
            context.add_interaction(command, response)

        elif "date" in command:
            response = get_date()
            speak(engine, response, response_text)
            context.add_interaction(command, response)

        elif "search for" in command:
            query = command.replace("search for", "").strip()
            response = web_search(query)
            speak(engine, response, response_text)
            context.add_interaction(command, response)

        elif "weather in" in command:
            city = command.replace("weather in", "").strip()
            response = get_weather(city, weather_api_key, prefs)
            speak(engine, response, response_text)
            context.add_interaction(command, response)
            if city:
                prefs.update_pref("default_city", city)

        elif "weather" in command:
            response = get_weather("", weather_api_key, prefs)
            speak(engine, response, response_text)
            context.add_interaction(command, response)

        elif "set reminder" in command:
            match = re.search(r"set reminder for (.+) at (\d{1,2}:\d{2})", command)
            if match:
                task, time_str = match.groups()
                response = set_reminder(task, time_str)
                speak(engine, response, response_text)
                context.add_interaction(command, response)
            else:
                response = "Please specify the task and time, e.g., 'set reminder for meeting at 14:30'."
                speak(engine, response, response_text)
                context.add_interaction(command, response)

        elif "open" in command:
            app = command.replace("open", "").strip()
            response = open_app(app)
            speak(engine, response, response_text)
            context.add_interaction(command, response)

        elif "close tab" in command or "close browser" in command:
            browser = ""
            if "chrome" in command:
                browser = "chrome"
            elif "firefox" in command:
                browser = "firefox"
            elif "safari" in command:
                browser = "safari"
            response = close_browser(browser)
            speak(engine, response, response_text)
            context.add_interaction(command, response)

        elif "volume" in command:
            match = re.search(r"set volume to (\d+)", command)
            if match:
                level = int(match.group(1))
                if 0 <= level <= 100:
                    response = set_volume(level, prefs)
                    speak(engine, response, response_text)
                    context.add_interaction(command, response)
                else:
                    response = "Volume level must be between 0 and 100."
                    speak(engine, response, response_text)
                    context.add_interaction(command, response)
            else:
                response = "Please specify a volume level, e.g., 'set volume to 50'."
                speak(engine, response, response_text)
                context.add_interaction(command, response)

        elif "set voice speed" in command:
            match = re.search(r"set voice speed to (\d+)", command)
            if match:
                rate = int(match.group(1))
                if 50 <= rate <= 300:
                    prefs.update_pref("voice_rate", rate)
                    engine.setProperty('rate', rate)
                    response = f"Voice speed set to {rate}."
                    speak(engine, response, response_text)
                    context.add_interaction(command, response)
                else:
                    response = "Voice speed must be between 50 and 300."
                    speak(engine, response, response_text)
                    context.add_interaction(command, response)
            else:
                response = "Please specify a speed, e.g., 'set voice speed to 200'."
                speak(engine, response, response_text)
                context.add_interaction(command, response)

        else:
            response = query_ai(command, context, ai_api_key)
            speak(engine, response, response_text)
            context.add_interaction(command, response)

        return True
    except Exception as e:
        logging.error(f"Command processing error: {e}")
        response = query_ai("An error occurred while processing the command. Please respond as JARVIS.", context, ai_api_key)
        speak(engine, response, response_text)
        context.add_interaction(command, response)
        return True

# GUI Application
class JarvisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("JARVIS Assistant")
        self.root.geometry("600x400")
        
        # Configuration
        self.WEATHER_API_KEY = "80fcfdc394edb614f0b526b4c8d675ba"
        self.AI_API_KEY = "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"
        self.WAKE_WORD = "jarvis"
        
        # Initialize components
        self.prefs = UserPreferences()
        self.context = ConversationContext()
        self.engine = init_tts_engine(self.prefs)
        self.recognizer = init_speech_recognizer()
        self.listening = False
        
        # GUI elements
        self.label = tk.Label(root, text="Enter Command:")
        self.label.pack(pady=5)
        
        self.command_entry = tk.Entry(root, width=50)
        self.command_entry.pack(pady=5)
        self.command_entry.bind("<Return>", self.submit_command)
        
        self.response_text = scrolledtext.ScrolledText(root, width=60, height=15, wrap=tk.WORD)
        self.response_text.pack(pady=5)
        
        self.submit_button = tk.Button(root, text="Submit", command=self.submit_command)
        self.submit_button.pack(side=tk.LEFT, padx=5)
        
        self.voice_button = tk.Button(root, text="Start Voice Input", command=self.toggle_voice)
        self.voice_button.pack(side=tk.LEFT, padx=5)
        
        self.clear_button = tk.Button(root, text="Clear", command=self.clear_response)
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        self.status_label = tk.Label(root, text="Voice Input: Off")
        self.status_label.pack(pady=5)
        
        # Start reminder checker
        self.reminder_thread = threading.Thread(
            target=check_reminders, args=(self.engine, self.response_text), daemon=True
        )
        self.reminder_thread.start()
        
        # Welcome message
        welcome_msg = query_ai("Greet the user as JARVIS, the AI assistant, and ask how you can assist.", 
                             self.context, self.AI_API_KEY)
        speak(self.engine, welcome_msg, self.response_text)
        logging.info("JARVIS started with GUI and Groq AI integration.")
        
        # Start voice input loop in a separate thread
        self.root.after(100, self.check_voice_input)
    
    def submit_command(self, event=None):
        command = self.command_entry.get().lower().strip()
        if command:
            self.command_entry.delete(0, tk.END)
            if not process_command(
                self.engine, command, self.WEATHER_API_KEY, self.AI_API_KEY, 
                self.prefs, self.context, self.response_text
            ):
                self.root.quit()
    
    def toggle_voice(self):
        self.listening = not self.listening
        if self.listening:
            self.voice_button.config(text="Stop Voice Input")
            self.status_label.config(text="Voice Input: On")
        else:
            self.voice_button.config(text="Start Voice Input")
            self.status_label.config(text="Voice Input: Off")
    
    def check_voice_input(self):
        if self.listening:
            command = listen(self.recognizer)
            if command:
                if self.WAKE_WORD in command:
                    speak(self.engine, "Yes, sir?", self.response_text)
                    command = listen(self.recognizer, timeout=10)
                    if not command:
                        speak(self.engine, "No command received.", self.response_text)
                    else:
                        if not process_command(
                            self.engine, command, self.WEATHER_API_KEY, self.AI_API_KEY, 
                            self.prefs, self.context, self.response_text
                        ):
                            self.root.quit()
                else:
                    if not process_command(
                        self.engine, command, self.WEATHER_API_KEY, self.AI_API_KEY, 
                        self.prefs, self.context, self.response_text
                    ):
                        self.root.quit()
        self.root.after(100, self.check_voice_input)
    
    def clear_response(self):
        self.response_text.delete(1.0, tk.END)

# Main function
def main():
    root = tk.Tk()
    app = JarvisGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

# Setup instructions
# 1. Install dependencies:
#    pip install speechrecognition pyttsx3 requests pycaw comtypes groq
# 2. Ensure OpenWeatherMap API key is valid
# 3. Ensure Groq API key is valid for Groq service
# 4. On Windows, pycaw handles volume control; run as administrator for browser closing
# 5. Reminders stored in reminders.json
# 6. Preferences stored in user_prefs.json
# 7. Logs saved to jarvis.log
# 8. Use GUI to type commands or click 'Start Voice Input' for voice commands
# 9. Voice input uses 'jarvis' wake word, e.g., 'jarvis open youtube'
# 10. Text input accepts commands, e.g., 'open youtube', 'close browser'
# 11. All non-system commands query Groq AI and display/speak the response
# 12. To close browser, use 'close browser' or 'close chrome/firefox/safari'