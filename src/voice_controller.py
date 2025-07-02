import speech_recognition as sr
import pyttsx3
import threading
import time
from typing import Dict, List, Optional, Tuple
from fuzzywuzzy import fuzz

class VoiceController:
    def __init__(self):
        # Speech recognition setup
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Text-to-speech for feedback
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 200)  # Speed
        self.tts.setProperty('volume', 0.8)  # Volume
        
        # Voice control state
        self.listening = False
        self.enabled = True
        self.audio_feedback = True
        
        # Recognition settings - Optimized for frequent recognition
        self.recognition_timeout = 0.1  # Very short timeout to listen more frequently
        self.phrase_timeout = 0.8      # Quick phrase capture
        self.energy_threshold = 150    # Even more sensitive
        
        # Fuzzy matching threshold
        self.match_threshold = 70  # 70% similarity required
        
        # Background listening thread
        self.listen_thread = None
        
        # Last of Us controls - hardcoded for testing
        self.action_mappings = {
            'sprint': {
                'name': 'Sprint',
                'key': 'shift',
                'intents': ['sprint', 'run', 'fast', 'hurry', 'speed']
            },
            'crouch': {
                'name': 'Crouch',
                'key': 'c',
                'intents': ['crouch', 'duck', 'hide', 'stealth', 'prone']
            },
            'interact': {
                'name': 'Interact',
                'key': 'e',
                'intents': ['interact', 'use', 'open', 'pick up', 'grab', 'take']
            },
            'shield': {
                'name': 'Shield',
                'key': 'e',
                'intents': ['shield', 'parry', 'defend', 'block', 'guard', 'protection']
            },
            'dodge': {
                'name': 'Dodge',
                'key': 'alt',
                'intents': ['dodge', 'evade', 'avoid', 'roll']
            },
            'escape': {
                'name': 'Escape',
                'key': 'f',
                'intents': ['escape', 'miss', 'get away', 'flee', 'run away']
            },
            'listen': {
                'name': 'Listen Mode',
                'key': 'q',
                'intents': ['listen', 'focus', 'hearing', 'detect', 'sense']
            },
            'reload': {
                'name': 'Reload',
                'key': 'left_click',
                'intents': ['reload', 'ammo', 'bullets', 'magazine', 'clip']
            },
            'melee': {
                'name': 'Melee',
                'key': 'f',
                'intents': ['melee', 'punch', 'hit', 'strike', 'attack', 'knife']
            },
            'flashlight': {
                'name': 'Flashlight',
                'key': 't',
                'intents': ['flashlight', 'light', 'torch', 'illuminate']
            },
            'shake_flashlight': {
                'name': 'Shake Flashlight',
                'key': 'j',
                'intents': ['shake', 'shake light', 'recharge', 'power up']
            },
            'backpack': {
                'name': 'Backpack',
                'key': 'tab',
                'intents': ['backpack', 'inventory', 'items', 'bag', 'gear']
            },
            'pause': {
                'name': 'Pause Game',
                'key': 'esc',
                'intents': ['pause', 'menu', 'stop', 'break']
            },
            'slow_move': {
                'name': 'Slow Move',
                'key': 'ctrl',
                'intents': ['slow', 'walk', 'careful', 'quiet', 'sneak']
            },
            'aim': {
                'name': 'Aim',
                'key': 'right_click',
                'intents': ['aim', 'target', 'sight', 'focus aim']
            },
            'fire': {
                'name': 'Fire',
                'key': 'left_click',
                'intents': ['fire', 'shoot', 'bang', 'attack', 'kill']
            },
            'scope': {
                'name': 'Scope',
                'key': 'e',
                'intents': ['scope', 'zoom', 'magnify', 'look closer']
            },
            'crafting': {
                'name': 'Crafting',
                'key': 'space',
                'intents': ['craft', 'make', 'build', 'create', 'crafting']
            },
            'long_gun': {
                'name': 'Long Gun',
                'key': '2',
                'intents': ['rifle', 'long gun', 'primary', 'big gun']
            },
            'short_gun': {
                'name': 'Short Gun',
                'key': '3',
                'intents': ['pistol', 'short gun', 'handgun', 'sidearm']
            },
            'health_kit': {
                'name': 'Health Kit',
                'key': '7',
                'intents': ['health', 'heal', 'medkit', 'first aid', 'bandage']
            },
            'molotov': {
                'name': 'Molotov',
                'key': '9',
                'intents': ['molotov', 'fire bomb', 'cocktail', 'burn']
            },
            'brick': {
                'name': 'Brick',
                'key': '8',
                'intents': ['brick', 'bottle', 'throw', 'distract']
            },
            'bomb': {
                'name': 'Bomb',
                'key': '5',
                'intents': ['bomb', 'explosive', 'grenade', 'boom']
            }
        }
        
        print("🎤 Voice Controller initialized - Last of Us controls loaded")
        print(f"📋 Available voice commands: {len(self.action_mappings)} actions")
        self._calibrate_microphone()
    
    def _calibrate_microphone(self):
        """Calibrate microphone for ambient noise"""
        try:
            print("🎧 Calibrating microphone for ambient noise...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            self.recognizer.energy_threshold = self.energy_threshold
            print("✅ Microphone calibrated")
        except Exception as e:
            print(f"⚠️ Microphone calibration warning: {e}")
    
    def find_best_intent_match(self, spoken_text: str) -> Optional[Tuple[str, str, int]]:
        """Find best matching intent using fuzzy matching"""
        best_match = None
        best_score = 0
        best_action = None
        
        spoken_text = spoken_text.lower().strip()
        
        for action_id, config in self.action_mappings.items():
            for intent in config['intents']:
                # Calculate fuzzy match score
                score = fuzz.ratio(spoken_text, intent.lower())
                
                if score > best_score and score >= self.match_threshold:
                    best_score = score
                    best_match = intent
                    best_action = action_id
        
        if best_match:
            return best_action, best_match, best_score
        return None
    
    def execute_voice_command(self, spoken_text: str):
        """Execute voice command if recognized"""
        print(f"🔍 Trying to match: '{spoken_text}'")
        match_result = self.find_best_intent_match(spoken_text)
        if not match_result:
            print(f"❌ No match found for: '{spoken_text}'")
            return None
        
        action_id, matched_intent, confidence = match_result
        action_config = self.action_mappings[action_id]
        
        print(f"✅ Voice Command: '{spoken_text}' → '{matched_intent}' → {action_config['name']} ({confidence}%)")
        
        # Audio feedback
        if self.audio_feedback:
            threading.Thread(target=self._speak_feedback, args=(action_config['name'],), daemon=True).start()
        
        # Return the action info for InputController to execute
        return {
            'action_id': action_id,
            'key': action_config['key'],
            'name': action_config['name'],
            'confidence': confidence
        }
    
    def _speak_feedback(self, action_name: str):
        """Provide audio feedback for executed action"""
        try:
            self.tts.say(action_name)
            self.tts.runAndWait()
        except:
            pass  # Ignore TTS errors
    
    def start_listening(self):
        """Start background voice recognition"""
        if self.listening:
            return
        
        self.listening = True
        self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listen_thread.start()
        print("🎤 Voice recognition started")
    
    def stop_listening(self):
        """Stop background voice recognition"""
        self.listening = False
        if self.listen_thread:
            self.listen_thread.join(timeout=1.0)
        print("🎤 Voice recognition stopped")
    
    def _listen_loop(self):
        """Main listening loop running in background thread"""
        print("🎤 Voice listening loop started - HIGH FREQUENCY MODE...")
        
        # Keep microphone context open for faster response
        with self.microphone as source:
            while self.listening and self.enabled:
                try:
                    # Continuous listening with very short timeout
                    audio = self.recognizer.listen(
                        source, 
                        timeout=self.recognition_timeout,
                        phrase_time_limit=self.phrase_timeout
                    )
                    
                    # Recognize speech using Google Web Speech API
                    try:
                        spoken_text = self.recognizer.recognize_google(audio, language='en-US')
                        if spoken_text:
                            print(f"🎤 Raw speech detected: '{spoken_text}'")
                            # Process the command immediately
                            self._on_speech_recognized(spoken_text)
                            
                    except sr.UnknownValueError:
                        # Speech not understood - normal, ignore silently
                        pass
                    except sr.RequestError as e:
                        print(f"⚠️ Speech recognition service error: {e}")
                        time.sleep(0.2)  # Brief pause on service error
                        
                except sr.WaitTimeoutError:
                    # Timeout waiting for speech - immediately continue listening
                    continue
                except Exception as e:
                    print(f"⚠️ Voice recognition error: {e}")
                    time.sleep(0.05)  # Very brief pause
    
    def _on_speech_recognized(self, spoken_text: str):
        """Handle recognized speech - to be overridden by main system"""
        # This is a placeholder - the main system will override this
        # to handle the voice commands
        print(f"🎤 Recognized: '{spoken_text}'")
    
    def toggle(self) -> bool:
        """Toggle voice recognition on/off"""
        self.enabled = not self.enabled
        if self.enabled:
            print("✅ Voice recognition ENABLED")
        else:
            print("❌ Voice recognition DISABLED")
        return self.enabled
    
    def set_audio_feedback(self, enabled: bool):
        """Enable/disable audio feedback"""
        self.audio_feedback = enabled
        status = "ENABLED" if enabled else "DISABLED"
        print(f"🔊 Audio feedback {status}")
    
    def list_available_commands(self):
        """Print all available voice commands"""
        print("\n🎤 Available Voice Commands:")
        print("=" * 50)
        for action_id, config in self.action_mappings.items():
            intents_str = ", ".join(f"'{intent}'" for intent in config['intents'])
            print(f"📋 {config['name']} ({config['key']}): {intents_str}")
        print("=" * 50)
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_listening()
        try:
            self.tts.stop()
        except:
            pass
        print("🎤 Voice controller cleanup complete")