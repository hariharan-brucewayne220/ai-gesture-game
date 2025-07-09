import speech_recognition as sr
import pyttsx3
import threading
import time
import json
from typing import Dict, List, Optional, Tuple
from fuzzywuzzy import fuzz

class VoiceController:
    def __init__(self, groq_api_key=None, openai_api_key=None):
        # Speech recognition setup
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Enhanced recognition options
        self.openai_api_key = openai_api_key
        self.use_whisper_api = bool(openai_api_key)
        
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
        
        # Recognition settings - REAL-TIME gaming optimization
        self.recognition_timeout = 0.01   # Ultra-short timeout for instant response
        self.phrase_timeout = 0.3         # Quick phrase capture for fast commands
        self.energy_threshold = 250       # Lower threshold for better sensitivity
        self.real_time_mode = True        # Enable continuous real-time listening
        
        # Enhanced Google Speech settings
        self.google_language = 'en-US'  # Optimize for US English
        self.google_show_all = True     # Get confidence scores
        self.google_with_confidence = True  # Enable confidence reporting
        
        # Game audio handling
        self.noise_suppression = True   # Enable noise suppression
        self.gaming_mode = True         # Optimize for gaming environment
        self.adaptive_threshold = True  # Dynamically adjust for game audio
        
        # Fuzzy matching threshold  
        self.match_threshold = 50  # 50% similarity required (more lenient for better recognition)
        
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
        if self.use_local_whisper:
            api_status = "Local Whisper (Free & Fast)"
        elif self.use_whisper_api:
            api_status = "OpenAI Whisper API"
        else:
            api_status = "Google Web Speech (Free)"
        print(f"🎤 Voice Controller initialized - {api_status}")
        print(f"📋 Available voice commands: {len(self.action_mappings)} actions")
        
        # Try to load custom voice commands
        self.load_voice_config()
        
        self._calibrate_microphone()
    
    def _calibrate_microphone(self):
        """Enhanced microphone calibration for gaming environments"""
        try:
            print("🎧 Calibrating microphone for gaming environment...")
            
            if self.gaming_mode:
                print("🎮 Gaming mode: Optimizing for game audio interference")
                print("🔇 Please pause game audio or be quiet for 2 seconds...")
                
            with self.microphone as source:
                # Gaming-optimized calibration
                calibration_duration = 2.0 if self.gaming_mode else 1.0
                self.recognizer.adjust_for_ambient_noise(source, duration=calibration_duration)
                
            # Gaming-specific settings
            original_threshold = self.recognizer.energy_threshold
            
            if self.real_time_mode:
                # REAL-TIME MODE: Optimized for instant response
                self.recognizer.energy_threshold = max(self.energy_threshold, original_threshold * 0.9)
                self.recognizer.pause_threshold = 0.3   # Very short pause detection
                self.recognizer.phrase_threshold = 0.1  # Minimal phrase threshold
                self.recognizer.non_speaking_duration = 0.2  # Quick response
            elif self.gaming_mode:
                # GAMING MODE: Balanced for game audio
                self.recognizer.energy_threshold = max(self.energy_threshold, original_threshold * 1.2)
                self.recognizer.pause_threshold = 0.8  # Longer pause for game audio
                self.recognizer.phrase_threshold = 0.4  # Higher phrase threshold
                self.recognizer.non_speaking_duration = 0.6  # Longer wait time
            else:
                # STANDARD MODE: Original settings
                self.recognizer.energy_threshold = max(self.energy_threshold, original_threshold * 0.8)
                self.recognizer.pause_threshold = 0.6
                self.recognizer.phrase_threshold = 0.3
                self.recognizer.non_speaking_duration = 0.5
                
            self.recognizer.dynamic_energy_threshold = self.adaptive_threshold
            self.recognizer.operation_timeout = None
            
            print(f"✅ Gaming microphone calibrated!")
            print(f"   Energy threshold: {self.recognizer.energy_threshold:.0f}")
            print(f"   Gaming mode: {'ON' if self.gaming_mode else 'OFF'}")
            print(f"   Noise suppression: {'ON' if self.noise_suppression else 'OFF'}")
            print(f"💡 For best results with game audio:")
            print(f"   • Speak louder and closer to microphone")
            print(f"   • Use push-to-talk if possible")
            print(f"   • Lower game volume slightly")
            
        except Exception as e:
            print(f"⚠️ Microphone calibration warning: {e}")
            
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
        # Gaming mode: more lenient filtering due to potential audio interference
        max_words = 4 if self.gaming_mode else 3
        max_length = 25 if self.gaming_mode else 20
        
        # Filter out random conversation - only process short commands
        if len(spoken_text.split()) > max_words or len(spoken_text) > max_length:
            if not self.gaming_mode:
                print(f"🔇 Ignored (too long): '{spoken_text}'")
            return None
            
        print(f"🔍 Trying to match: '{spoken_text}'")
        
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
        print("🎤 Voice recognition started")
    
    def stop_listening(self):
        """Stop background voice recognition"""
        self.listening = False
        if self.listen_thread:
            self.listen_thread.join(timeout=1.0)
        print("🎤 Voice recognition stopped")
    
    def _listen_loop(self):
        """REAL-TIME listening loop - always active for gaming"""
        print("🎤 REAL-TIME Voice listening started - ALWAYS ACTIVE MODE...")
        
        # Keep microphone context open for maximum responsiveness
        with self.microphone as source:
            # Initial quick calibration
            self.recognizer.adjust_for_ambient_noise(source, duration=0.1)
            
            while self.listening and self.enabled:
                try:
                    if self.real_time_mode:
                        # REAL-TIME MODE: Continuous listening with no timeouts
                        audio = self.recognizer.listen(
                            source,
                            timeout=None,  # No timeout - always listening
                            phrase_time_limit=self.phrase_timeout
                        )
                        
                        # Apply noise filtering if enabled
                        if self.noise_suppression:
                            audio = self._apply_noise_suppression(audio)
                            
                    else:
                        # LEGACY MODE: With timeouts (fallback)
                        audio = self.recognizer.listen(
                            source, 
                            timeout=self.recognition_timeout,
                            phrase_time_limit=self.phrase_timeout
                        )
                    
                    # Recognize speech using best available API
                    try:
                        if self.use_local_whisper:
                            # Use Local Whisper (free, fast) - will load model if needed
                            spoken_text = self._recognize_with_local_whisper(audio)
                        elif self.use_whisper_api and self.openai_api_key:
                            # Use OpenAI Whisper API (most accurate)
                            spoken_text = self.recognizer.recognize_whisper_api(
                                audio, 
                                api_key=self.openai_api_key
                            )
                        else:
                            # ULTRA-FAST Google Web Speech API for real-time gaming
                            if self.real_time_mode:
                                # Real-time mode: Fast and direct recognition
                                spoken_text = self.recognizer.recognize_google(
                                    audio, 
                                    language=self.google_language,
                                    # Remove confidence checks for speed
                                )
                            else:
                                # Legacy mode: Enhanced with confidence
                                try:
                                    # Try with confidence scores first
                                    results = self.recognizer.recognize_google(
                                        audio, 
                                        language=self.google_language,
                                        show_all=True,  # Get all alternatives with confidence
                                        with_confidence=True
                                    )
                                    
                                    if results and len(results['alternative']) > 0:
                                        # Get the best result with confidence
                                        best_result = results['alternative'][0]
                                        if 'confidence' in best_result and best_result['confidence'] >= self.min_confidence:
                                            spoken_text = best_result['transcript']
                                            if self.debug_mode:
                                                print(f"🎤 Google confidence: {best_result['confidence']:.2f}")
                                        elif 'transcript' in best_result:
                                            # Use result even without confidence if it exists
                                            spoken_text = best_result['transcript']
                                        else:
                                            spoken_text = None
                                    else:
                                        spoken_text = None
                                        
                                except Exception:
                                    # Fallback to simple recognition
                                    spoken_text = self.recognizer.recognize_google(
                                        audio, 
                                        language=self.google_language
                                    )
                        
                        if spoken_text:
                            if self.use_local_whisper:
                                api_indicator = "⚡ Local Whisper"
                            elif self.use_whisper_api:
                                api_indicator = "🤖 Whisper API"
                            else:
                                api_indicator = "🔍 Google"
                            print(f"🎤 {api_indicator} Raw speech detected: '{spoken_text}'")
                            # Process the command immediately
                            self._on_speech_recognized(spoken_text)
                            
                    except sr.UnknownValueError:
                        # Speech not understood - normal in real-time mode, ignore silently
                        if self.real_time_mode:
                            continue  # Immediately continue listening
                        else:
                            pass
                    except sr.RequestError as e:
                        if self.real_time_mode:
                            # In real-time mode, minimal error handling for speed
                            continue
                        else:
                            print(f"⚠️ Speech recognition service error: {e}")
                            time.sleep(0.1)  # Very brief pause
                        
                except sr.WaitTimeoutError:
                    # In real-time mode, this shouldn't happen (no timeout)
                    if self.real_time_mode:
                        continue  # Immediately restart
                    else:
                        continue  # Legacy behavior
                except Exception as e:
                    if self.real_time_mode:
                        # In real-time mode, ignore errors and continue
                        continue
                    else:
                        print(f"⚠️ Voice recognition error: {e}")
                        time.sleep(0.02)  # Minimal pause
    
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
    
    def add_voice_command(self, action_id: str, new_intent: str):
        """Add a new voice command for an existing action"""
        if action_id in self.action_mappings:
            if new_intent not in self.action_mappings[action_id]['intents']:
                self.action_mappings[action_id]['intents'].append(new_intent)
                print(f"✅ Added '{new_intent}' for {self.action_mappings[action_id]['name']}")
                return True
            else:
                print(f"⚠️ '{new_intent}' already exists for {action_id}")
        else:
            print(f"❌ Action '{action_id}' not found")
        return False
    
    def remove_voice_command(self, action_id: str, intent_to_remove: str):
        """Remove a voice command for an action"""
        if action_id in self.action_mappings:
            if intent_to_remove in self.action_mappings[action_id]['intents']:
                self.action_mappings[action_id]['intents'].remove(intent_to_remove)
                print(f"✅ Removed '{intent_to_remove}' from {self.action_mappings[action_id]['name']}")
                return True
            else:
                print(f"⚠️ '{intent_to_remove}' not found for {action_id}")
        else:
            print(f"❌ Action '{action_id}' not found")
        return False
    
    def save_voice_config(self, filename="voice_commands.json"):
        """Save current voice commands to file"""
        try:
            import json
            with open(filename, 'w') as f:
                json.dump(self.action_mappings, f, indent=2)
            print(f"✅ Voice commands saved to {filename}")
            return True
        except Exception as e:
            print(f"❌ Failed to save voice commands: {e}")
            return False
    
    def load_voice_config(self, filename="voice_commands.json"):
        """Load voice commands from file"""
        try:
            import json
            import os
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    self.action_mappings = json.load(f)
                print(f"✅ Voice commands loaded from {filename}")
                return True
            else:
                print(f"⚠️ File {filename} not found, using defaults")
        except Exception as e:
            print(f"❌ Failed to load voice commands: {e}")
        return False
    
    def toggle_ai_mode(self):
        """Toggle between AI and traditional voice parsing"""
        if not self.groq_api_key:
            print("❌ AI mode unavailable - no API key provided")
            return False
            
        self.ai_mode = not self.ai_mode
        mode = "AI-POWERED" if self.ai_mode else "TRADITIONAL"
        print(f"🤖 Voice parsing mode: {mode}")
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
                    print(f"🤖 AI parsed '{spoken_text}' → {action_id}")
                    return action_id
                    
            return None
            
        except Exception as e:
            print(f"❌ AI parsing failed: {e}")
            return None
    
    def _recognize_with_local_whisper(self, audio):
        """Recognize speech using Local Whisper (without FFmpeg dependency)"""
        try:
            # Check if model is loaded
            if not self.whisper_model:
                if not self._init_local_whisper_model():
                    return None
            
            # Convert audio directly to numpy array (bypass FFmpeg)
            import numpy as np
            import wave
            import tempfile
            import os
            import time
            
            # Get raw audio data
            audio_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
            
            # Convert to numpy array (16kHz, 16-bit)
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            try:
                # Transcribe directly with numpy array (no temp file needed)
                result = self.whisper_model.transcribe(audio_np)
                transcription = result["text"].strip()
                return transcription
            except Exception as whisper_error:
                print(f"⚠️ Direct transcription failed: {whisper_error}")
                # Fallback: try with temporary WAV file but use librosa instead of FFmpeg
                return self._whisper_fallback_with_wav(audio_data)
            
        except Exception as e:
            print(f"⚠️ Local Whisper recognition error: {e}")
            return None
            
    def _whisper_fallback_with_wav(self, audio_data):
        """Fallback method using temporary WAV file"""
        try:
            import tempfile
            import os
            import time
            import wave
            import numpy as np
            
            # Create temporary WAV file with proper format
            temp_dir = tempfile.gettempdir()
            tmp_path = os.path.join(temp_dir, f"whisper_audio_{int(time.time() * 1000)}.wav")
            
            # Write WAV file manually (avoid FFmpeg)
            with wave.open(tmp_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)  # 16kHz
                wav_file.writeframes(audio_data)
            
            try:
                # Try to read with librosa if available (doesn't need FFmpeg)
                try:
                    import librosa
                    audio_np, _ = librosa.load(tmp_path, sr=16000)
                    result = self.whisper_model.transcribe(audio_np)
                    return result["text"].strip()
                except ImportError:
                    print("⚠️ FFmpeg not available and librosa not installed")
                    print("💡 Install FFmpeg or librosa for Local Whisper support")
                    return None
                    
            finally:
                # Clean up
                try:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                except:
                    pass
                    
        except Exception as e:
            print(f"⚠️ WAV fallback failed: {e}")
            return None
    
    
    def _init_local_whisper_model(self):
        """Initialize Local Whisper model"""
        try:
            print("🔄 Loading Local Whisper model (first time may take a moment)...")
            import whisper
            import warnings
            # Suppress FP16 warning - it's expected on CPU
            warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")
            
            # Use base model - good balance of speed and accuracy
            model_name = "base"  # ~140MB, fast for gaming
            self.whisper_model = whisper.load_model(model_name)
            
            print("✅ Local Whisper model loaded successfully!")
            return True
            
        except ImportError:
            print("❌ Whisper not installed. Install with: pip install openai-whisper")
            return False
        except Exception as e:
            print(f"❌ Failed to load Whisper model: {e}")
            return False
    
    def toggle_local_whisper(self):
        """Toggle Local Whisper recognition"""
        if not self.use_local_whisper:
            # Try to enable Local Whisper
            if self.whisper_model or self._init_local_whisper_model():
                self.use_local_whisper = True
                self.use_whisper_api = False  # Disable other APIs
                print("✅ Switched to Local Whisper (Free, Offline)")
                return True
            else:
                print("❌ Failed to enable Local Whisper")
                return False
        else:
            # Disable Local Whisper
            self.use_local_whisper = False
            print("✅ Disabled Local Whisper - using fallback API")
            return True

    def toggle_debug_mode(self):
        """Toggle voice recognition debug mode"""
        self.debug_mode = not self.debug_mode
        status = "ON" if self.debug_mode else "OFF"
        print(f"🎤 Voice debug mode: {status}")
        if self.debug_mode:
            print("💡 Debug shows confidence scores and recognition details")
        return self.debug_mode
        
    def recalibrate_microphone(self):
        """Recalibrate microphone sensitivity"""
        print("🔄 Recalibrating microphone...")
        self._calibrate_microphone()
        print("✅ Microphone recalibrated!")
        
    def _apply_noise_suppression(self, audio):
        """Apply basic noise suppression for gaming environments"""
        try:
            import numpy as np
            from scipy import signal
            
            # Convert audio to numpy array
            audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
            
            # Apply simple high-pass filter to reduce low-frequency game audio
            # Most human speech is 300Hz-3000Hz, game audio often has more bass
            nyquist = 22050  # Half of typical sample rate
            low_cutoff = 300  # Hz - filter out bass/music
            high_cutoff = 3000  # Hz - focus on speech range
            
            # Design bandpass filter for human speech
            low = low_cutoff / nyquist
            high = high_cutoff / nyquist
            b, a = signal.butter(4, [low, high], btype='band')
            
            # Apply filter
            filtered_audio = signal.filtfilt(b, a, audio_data.astype(float))
            
            # Convert back to audio format
            filtered_audio = np.clip(filtered_audio, -32768, 32767).astype(np.int16)
            
            # Create new AudioData object
            import speech_recognition as sr
            filtered_audio_data = sr.AudioData(filtered_audio.tobytes(), audio.sample_rate, audio.sample_width)
            
            return filtered_audio_data
            
        except ImportError:
            # If scipy not available, return original audio
            print("⚠️ Install scipy for better noise suppression: pip install scipy")
            return audio
        except Exception as e:
            # If filtering fails, return original audio
            print(f"⚠️ Noise suppression failed: {e}")
            return audio
            
    def toggle_gaming_mode(self):
        """Toggle gaming mode optimization"""
        self.gaming_mode = not self.gaming_mode
        status = "ON" if self.gaming_mode else "OFF"
        print(f"🎮 Gaming mode: {status}")
        if self.gaming_mode:
            print("💡 Optimized for game audio interference")
            print("💡 Recalibrate microphone (press 'c') for best results")
        else:
            print("💡 Standard voice recognition mode")
        return self.gaming_mode
        
    def toggle_noise_suppression(self):
        """Toggle noise suppression"""
        self.noise_suppression = not self.noise_suppression
        status = "ON" if self.noise_suppression else "OFF"
        print(f"🔇 Noise suppression: {status}")
        if self.noise_suppression:
            print("💡 Filtering game audio frequencies")
        return self.noise_suppression
        
    def toggle_real_time_mode(self):
        """Toggle real-time voice recognition mode"""
        self.real_time_mode = not self.real_time_mode
        status = "ON" if self.real_time_mode else "OFF"
        print(f"⚡ Real-time mode: {status}")
        
        if self.real_time_mode:
            print("💡 Always listening - instant response")
            print("💡 Ultra-fast Google Speech API")
            print("💡 Recalibrate microphone (press 'c') for best results")
        else:
            print("💡 Standard mode with timeouts restored")
            
        return self.real_time_mode

    def cleanup(self):
        """Clean up resources"""
        self.stop_listening()
        try:
            self.tts.stop()
        except:
            pass
        print("🎤 Voice controller cleanup complete")