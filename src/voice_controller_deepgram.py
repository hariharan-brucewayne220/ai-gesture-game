import speech_recognition as sr
import pyttsx3
import threading
import time
import json
import tempfile
import wave
import requests
import os
from typing import Dict, List, Optional, Tuple
from fuzzywuzzy import fuzz

class VoiceController:
    def __init__(self, groq_api_key=None, openai_api_key=None, deepgram_api_key=None):
        # Speech recognition setup
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Enhanced recognition options
        self.openai_api_key = openai_api_key
        self.use_whisper_api = bool(openai_api_key)
        
        # Deepgram integration
        self.deepgram_api_key = deepgram_api_key
        self.use_deepgram = bool(deepgram_api_key)
        
        # Local Whisper recognition option
        self.use_local_whisper = False
        self.whisper_model = None
        
        # Text-to-speech for feedback
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 200)  # Speed
        self.tts.setProperty('volume', 0.8)  # Volume
        
        # Voice control state
        self.listening = False
        self.enabled = True
        self.audio_feedback = True
        
        # Recognition settings - RESTORED WORKING SETTINGS
        self.recognition_timeout = 0.1  # Very short timeout to listen more frequently
        self.phrase_timeout = 0.8      # Quick phrase capture
        self.energy_threshold = 150    # Even more sensitive (was 250, now back to 150)
        self.real_time_mode = False    # DISABLED real-time mode that was causing issues
        
        # Enhanced Google Speech settings
        self.google_language = 'en-US'  # Optimize for US English
        self.google_show_all = True     # Get confidence scores
        self.google_with_confidence = True  # Enable confidence reporting
        
        # Game audio handling - SIMPLIFIED
        self.noise_suppression = False  # DISABLED - was causing audio issues
        self.gaming_mode = False        # DISABLED - was setting threshold too high
        self.adaptive_threshold = True  # Keep this for basic adaptation
        
        # Fuzzy matching threshold  
        self.match_threshold = 70  # 70% similarity required (was 50, back to 70)
        
        # Voice command optimization
        self.min_confidence = 0.7  # Minimum confidence for Google Speech
        self.recent_commands = []  # Track recent commands for context
        self.command_history_size = 5  # Remember last 5 commands
        
        # AI Integration
        self.groq_api_key = groq_api_key
        self.ai_mode = False  # Default to traditional mode
        
        # Debug mode for filtering
        self.debug_mode = False
        
        # Background listening thread
        self.listen_thread = None
        
        # Last of Us controls - hardcoded for testing
        self.action_mappings = {
            'sprint': {
                'name': 'Sprint',
                'key': 'shift',
                'intents': ['sprwdnt', 'run', 'fast', 'hurry', 'speed']
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
                'intents': ['flashlight', 'light', 'torch', 'illuminate', 'flash']
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
            'walk': {
                'name': 'Walk',
                'key': 'shift',  # Maps to sprint handler which will handle walk as stop sprint
                'intents': ['walk', 'slow down', 'normal speed', 'stop sprinting']
            },
            'sneak': {
                'name': 'Sneak',
                'key': 'ctrl',
                'intents': ['sneak', 'careful', 'quiet', 'crouch']
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
                'name': 'Rifle',
                'key': '2',
                'intents': ['rifle', 'gun', 'two']
            },
            'short_gun': {
                'name': 'Pistol',
                'key': '3',
                'intents': ['pistol', 'three']
            },
            'next_weapon': {
                'name': 'Next',
                'key': 'mouse_wheel_up',
                'intents': ['next', 'switch', 'change']
            },
            'previous_weapon': {
                'name': 'Back', 
                'key': 'mouse_wheel_down',
                'intents': ['back', 'prev']
            },
            'health_kit': {
                'name': 'Health Kit',
                'key': '7',
                'intents': ['health', 'heal', 'medkit', 'first aid', 'bandage', 'med']
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
            },
            'flame_on': {
                'name': 'Hand Camera On',
                'key': 'hand_camera_on',
                'intents': ['flame on', 'hand track on', 'camera on', 'tracking on']
            },
            'flame_off': {
                'name': 'Hand Camera Off', 
                'key': 'hand_camera_off',
                'intents': ['flame off', 'hand track off', 'camera off', 'tracking off']
            },
            'test': {
                'name': 'Test Command',
                'key': 'space',
                'intents': ['hello', 'hi', 'test', 'check']
            }
        }
        
        # Determine API status
        if self.use_deepgram:
            api_status = "Deepgram (Premium Accuracy)"
        elif self.use_local_whisper:
            api_status = "Local Whisper (Free & Fast)"
        elif self.use_whisper_api:
            api_status = "OpenAI Whisper API"
        else:
            api_status = "Google Web Speech (Free)"
        print(f"Voice Controller initialized - {api_status}")
        print(f"Available voice commands: {len(self.action_mappings)} actions")
        
        # Try to load custom voice commands
        self.load_voice_config()
        
        self._calibrate_microphone()
    
    def _calibrate_microphone(self):
        """Calibrate microphone for ambient noise - SIMPLIFIED VERSION"""
        try:
            print("Calibrating microphone for ambient noise...")
            
            with self.microphone as source:
                # Simple calibration like the old working version
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
            # Set energy threshold to working value
            self.recognizer.energy_threshold = self.energy_threshold
            self.recognizer.dynamic_energy_threshold = self.adaptive_threshold
            self.recognizer.operation_timeout = None
            
            print(f"Microphone calibrated - Energy threshold: {self.recognizer.energy_threshold}")
            
        except Exception as e:
            print(f"Microphone calibration warning: {e}")
    
    def _recognize_with_deepgram(self, audio):
        """Recognize speech using Deepgram API"""
        if not self.deepgram_api_key:
            return None
            
        try:
            # Save audio to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                with wave.open(temp_file.name, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(16000)
                    wav_file.writeframes(audio.get_raw_data(convert_rate=16000, convert_width=2))
                
                # Call Deepgram API
                url = "https://api.deepgram.com/v1/listen"
                headers = {
                    "Authorization": f"Token {self.deepgram_api_key}",
                    "Content-Type": "audio/wav"
                }
                params = {
                    "model": "nova-2",
                    "language": "en-US",
                    "smart_format": "true",
                    "punctuate": "true"
                }
                
                with open(temp_file.name, 'rb') as audio_file:
                    response = requests.post(
                        url, 
                        headers=headers, 
                        params=params, 
                        data=audio_file.read(),
                        timeout=10
                    )
                
                # Clean up temp file
                os.unlink(temp_file.name)
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('results') and result['results'].get('channels'):
                        alternatives = result['results']['channels'][0].get('alternatives', [])
                        if alternatives:
                            transcript = alternatives[0].get('transcript', '').strip()
                            confidence = alternatives[0].get('confidence', 0)
                            return transcript, confidence
                
                return None, 0
                
        except Exception as e:
            print(f"Deepgram recognition error: {e}")
            return None, 0
            
    def add_contextual_boost(self, spoken_text: str):
        """Add contextual boosting based on recent commands"""
        # Add to recent commands history
        self.recent_commands.append(spoken_text.lower())
        if len(self.recent_commands) > self.command_history_size:
            self.recent_commands.pop(0)
            
        # Boost fuzzy matching for recently used commands
        for recent_cmd in self.recent_commands:
            for action_id, config in self.action_mappings.items():
                for intent in config['intents']:
                    if recent_cmd in intent.lower():
                        return True  # Command is contextually relevant
        return False
    
    def find_best_intent_match(self, spoken_text: str) -> Optional[Tuple[str, str, int]]:
        """Enhanced fuzzy matching with contextual boosting"""
        best_match = None
        best_score = 0
        best_action = None
        
        spoken_text = spoken_text.lower().strip()
        is_contextual = self.add_contextual_boost(spoken_text)
        
        for action_id, config in self.action_mappings.items():
            for intent in config['intents']:
                # Calculate multiple fuzzy match scores for better accuracy
                ratio_score = fuzz.ratio(spoken_text, intent.lower())
                partial_score = fuzz.partial_ratio(spoken_text, intent.lower())
                token_sort_score = fuzz.token_sort_ratio(spoken_text, intent.lower())
                
                # Use the best score from different algorithms
                score = max(ratio_score, partial_score, token_sort_score)
                
                # Contextual boosting - lower threshold for recent commands
                effective_threshold = self.match_threshold - 10 if is_contextual else self.match_threshold
                
                # Additional boosting for common gaming words
                if any(word in spoken_text for word in ['fire', 'shoot', 'run', 'move', 'go']):
                    score += 5
                
                if score > best_score and score >= effective_threshold:
                    best_score = score
                    best_match = intent
                    best_action = action_id
        
        if best_match:
            return best_action, best_match, best_score
        return None
    
    def execute_voice_command(self, spoken_text: str):
        """Execute voice command if recognized"""
        # Filter out random conversation - only process short commands
        if len(spoken_text.split()) > 3 or len(spoken_text) > 20:
            print(f"Ignored (too long): '{spoken_text}'")
            return None
            
        print(f"Trying to match: '{spoken_text}'")
        
        action_id = None
        confidence = 0
        matched_intent = ""
        
        # Try AI parsing first if enabled
        if self.ai_mode and self.groq_api_key:
            action_id = self.ai_parse_command(spoken_text)
            if action_id:
                confidence = 95  # High confidence for AI matches
                matched_intent = "AI-parsed"
        
        # Fallback to traditional fuzzy matching
        if not action_id:
            match_result = self.find_best_intent_match(spoken_text)
            if not match_result:
                print(f"ERROR No match found for: '{spoken_text}'")
                return None
            action_id, matched_intent, confidence = match_result
        
        action_config = self.action_mappings[action_id]
        
        mode_indicator = "AI" if self.ai_mode and confidence == 95 else "SEARCH"
        print(f"OK {mode_indicator} Voice Command: '{spoken_text}' -> '{matched_intent}' -> {action_config['name']} ({confidence}%)")
        
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
        print("Voice recognition started")
    
    def stop_listening(self):
        """Stop background voice recognition"""
        self.listening = False
        if self.listen_thread:
            self.listen_thread.join(timeout=1.0)
        print("Voice recognition stopped")
    
    def _listen_loop(self):
        """RESTORED WORKING listening loop with Deepgram support"""
        print("Voice listening loop started - HIGH FREQUENCY MODE...")
        
        # Keep microphone context open for faster response
        with self.microphone as source:
            while self.listening and self.enabled:
                try:
                    # Continuous listening with very short timeout - ORIGINAL WORKING SETTINGS
                    audio = self.recognizer.listen(
                        source, 
                        timeout=self.recognition_timeout,  # 0.1 seconds
                        phrase_time_limit=self.phrase_timeout  # 0.8 seconds
                    )
                    
                    # Recognize speech using best available API
                    try:
                        spoken_text = None
                        api_used = "Unknown"
                        
                        if self.use_deepgram and self.deepgram_api_key:
                            # Try Deepgram first (highest accuracy)
                            result = self._recognize_with_deepgram(audio)
                            if result and result[0]:
                                spoken_text, confidence = result
                                api_used = "Deepgram"
                                if self.debug_mode:
                                    print(f"Deepgram confidence: {confidence:.2f}")
                        
                        if not spoken_text and self.use_whisper_api and self.openai_api_key:
                            # Fallback to Whisper
                            spoken_text = self.recognizer.recognize_whisper_api(
                                audio, 
                                api_key=self.openai_api_key
                            )
                            api_used = "Whisper"
                        
                        if not spoken_text:
                            # Final fallback to Google Speech
                            spoken_text = self.recognizer.recognize_google(audio, language='en-US')
                            api_used = "Google"
                        
                        if spoken_text:
                            print(f"Raw speech detected ({api_used}): '{spoken_text}'")
                            # Process the command immediately
                            self._on_speech_recognized(spoken_text)
                            
                    except sr.UnknownValueError:
                        # Speech not understood - normal, ignore silently
                        pass
                    except sr.RequestError as e:
                        print(f"Speech recognition service error: {e}")
                        time.sleep(0.2)  # Brief pause on service error
                        
                except sr.WaitTimeoutError:
                    # Timeout waiting for speech - immediately continue listening
                    continue
                except Exception as e:
                    print(f"Voice recognition error: {e}")
                    time.sleep(0.05)  # Very brief pause
    
    def _on_speech_recognized(self, spoken_text: str):
        """Handle recognized speech - to be overridden by main system"""
        # This is a placeholder - the main system will override this
        # to handle the voice commands
        print(f"Recognized: '{spoken_text}'")
    
    def toggle(self) -> bool:
        """Toggle voice recognition on/off"""
        self.enabled = not self.enabled
        if self.enabled:
            print("Voice recognition ENABLED")
        else:
            print("Voice recognition DISABLED")
        return self.enabled
    
    def toggle_deepgram(self) -> bool:
        """Toggle Deepgram recognition on/off"""
        if not self.deepgram_api_key:
            print("Deepgram unavailable - no API key provided")
            return False
            
        self.use_deepgram = not self.use_deepgram
        status = "ENABLED" if self.use_deepgram else "DISABLED"
        print(f"Deepgram recognition {status}")
        return self.use_deepgram
    
    def set_audio_feedback(self, enabled: bool):
        """Enable/disable audio feedback"""
        self.audio_feedback = enabled
        status = "ENABLED" if enabled else "DISABLED"
        print(f"Audio feedback {status}")
    
    def list_available_commands(self):
        """Print all available voice commands"""
        print("\nAvailable Voice Commands:")
        print("=" * 50)
        for action_id, config in self.action_mappings.items():
            intents_str = ", ".join(f"'{intent}'" for intent in config['intents'])
            print(f"{config['name']} ({config['key']}): {intents_str}")
        print("=" * 50)
    
    def add_voice_command(self, action_id: str, new_intent: str):
        """Add a new voice command for an existing action"""
        if action_id in self.action_mappings:
            if new_intent not in self.action_mappings[action_id]['intents']:
                self.action_mappings[action_id]['intents'].append(new_intent)
                print(f"Added '{new_intent}' for {self.action_mappings[action_id]['name']}")
                return True
            else:
                print(f"'{new_intent}' already exists for {action_id}")
        else:
            print(f"Action '{action_id}' not found")
        return False
    
    def remove_voice_command(self, action_id: str, intent_to_remove: str):
        """Remove a voice command for an action"""
        if action_id in self.action_mappings:
            if intent_to_remove in self.action_mappings[action_id]['intents']:
                self.action_mappings[action_id]['intents'].remove(intent_to_remove)
                print(f"Removed '{intent_to_remove}' from {self.action_mappings[action_id]['name']}")
                return True
            else:
                print(f"'{intent_to_remove}' not found for {action_id}")
        else:
            print(f"Action '{action_id}' not found")
        return False
    
    def save_voice_config(self, filename="voice_commands.json"):
        """Save current voice commands to file"""
        try:
            import json
            with open(filename, 'w') as f:
                json.dump(self.action_mappings, f, indent=2)
            print(f"Voice commands saved to {filename}")
            return True
        except Exception as e:
            print(f"Failed to save voice commands: {e}")
            return False
    
    def load_voice_config(self, filename="voice_commands.json"):
        """Load voice commands from file"""
        try:
            import json
            import os
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    self.action_mappings = json.load(f)
                print(f"Voice commands loaded from {filename}")
                return True
            else:
                print(f"File {filename} not found, using defaults")
        except Exception as e:
            print(f"Failed to load voice commands: {e}")
        return False
    
    def toggle_ai_mode(self):
        """Toggle between AI and traditional voice parsing"""
        if not self.groq_api_key:
            print("AI mode unavailable - no API key provided")
            return False
            
        self.ai_mode = not self.ai_mode
        mode = "AI-POWERED" if self.ai_mode else "TRADITIONAL"
        print(f"Voice parsing mode: {mode}")
        return self.ai_mode
    
    def ai_parse_command(self, spoken_text: str) -> Optional[str]:
        """Parse voice command using Groq AI"""
        if not self.groq_api_key:
            return None
            
        try:
            import requests
            
            # Create game context from available commands
            available_actions = []
            for action_id, config in self.action_mappings.items():
                available_actions.append(f"{action_id}: {config['name']} - {', '.join(config['intents'])}")
            
            game_context = "\n".join(available_actions)
            
            prompt = f"""You are a gaming voice command parser. 
            
Available game actions:
{game_context}

User said: "{spoken_text}"

Parse this into ONE action ID from the list above. Consider context like:
- "roll roll roll" likely means dodge/crouch for defensive action
- "go go go" likely means sprint/run
- "get down" means crouch
- "move fast" means sprint

Return ONLY the action ID (like 'sprint', 'crouch', etc.) or 'none' if no match."""

            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "messages": [{"role": "user", "content": prompt}],
                "model": "llama-3.1-8b-instant",
                "temperature": 0.1,
                "max_tokens": 50
            }
            
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", 
                                   headers=headers, json=data, timeout=3)
            
            if response.status_code == 200:
                result = response.json()
                action_id = result['choices'][0]['message']['content'].strip().lower()
                
                if action_id in self.action_mappings:
                    print(f"AI parsed '{spoken_text}' -> {action_id}")
                    return action_id
                    
            return None
            
        except Exception as e:
            print(f"AI parsing failed: {e}")
            return None

    def cleanup(self):
        """Clean up resources"""
        self.stop_listening()
        try:
            self.tts.stop()
        except:
            pass
        print("Voice controller cleanup complete")