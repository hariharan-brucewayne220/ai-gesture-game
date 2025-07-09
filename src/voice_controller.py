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
import pyaudio
import numpy as np
from queue import Queue

# Vosk integration for ultra-fast gaming voice recognition
try:
    import vosk
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

class VoiceController:
    def __init__(self, groq_api_key=None, openai_api_key=None, deepgram_api_key=None):
        # Voice recognition engine priorities: Vosk (primary) → Google Speech (secondary) → Deepgram (disabled)
        self.voice_engine = "vosk"  # Primary: ultra-fast Vosk for gaming
        self.use_deepgram = False   # Disabled by default - can be toggled
        
        # Speech recognition setup
        self.recognizer = sr.Recognizer()
        
        # Initialize microphone with specific device selection
        self.microphone = None
        self.microphone_device_index = None
        self._initialize_microphone()
        
        # Vosk integration as PRIMARY engine
        self.vosk_model = None
        self.vosk_recognizer = None
        self.vosk_audio_queue = Queue()
        self.vosk_sample_rate = 16000
        self.vosk_chunk_size = 1024
        self.vosk_initialized = False
        
        # Deepgram integration - DISABLED by default, can be toggled
        self.deepgram_api_key = deepgram_api_key
        self.use_deepgram_fallback = False  # Disabled by default
        
        # Enhanced recognition options
        self.openai_api_key = openai_api_key
        
        # Text-to-speech for feedback
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 200)  # Speed
        self.tts.setProperty('volume', 0.8)  # Volume
        
        # Voice control state
        self.listening = False
        self.enabled = True
        self.audio_feedback = True
        
        # Recognition settings - OPTIMIZED FOR GAMING WITH HEADSET
        self.recognition_timeout = 0.5   # Balanced for headset gaming
        self.phrase_timeout = 1.2        # Quick commands but not cut off
        self.energy_threshold = 100      # More sensitive for headset
        self.real_time_mode = True       # ENABLED for gaming responsiveness
        
        # Enhanced Google Speech settings
        self.google_language = 'en-US'  # Optimize for US English
        self.google_show_all = True     # Get confidence scores
        self.google_with_confidence = True  # Enable confidence reporting
        
        # Game audio isolation - SMART FILTERING
        self.noise_suppression = True   # ENABLED with smart filtering
        self.confidence_threshold = 0.4   # LOWERED FURTHER - was blocking everything at 0.6
        self.min_voice_energy = 50     # LOWERED MORE - was too strict at 100
        self.game_audio_filter = False  # DISABLED by default - let user enable with 'i'
        
        # Audio monitoring for game interference detection
        self.background_noise_level = 100
        self.voice_to_noise_ratio = 2.0  # Voice must be 2x louder than background
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
        self.debug_mode = False  # Disable debug for clean testing
        
        # Emergency mode - completely disable filtering
        self.emergency_mode = False  # Set to True to disable ALL filtering
        
        # Vosk partial result throttling to prevent spam
        self.last_partial_text = ""
        self.last_partial_time = 0
        self.partial_throttle_interval = 0.2  # Only show partial results every 200ms
        self.partial_change_threshold = 3  # Only show if text changed by 3+ characters
        
        # Command smoothing/debouncing to prevent spam (like eye movement smoothing)
        self.last_executed_command = ""
        self.last_execution_time = 0
        self.command_debounce_interval = 1.0  # 1 second between same commands
        self.command_execution_buffer = []  # Buffer recent commands
        self.command_buffer_size = 3  # Require 3 consistent detections
        
        # Add missing command execution attributes
        self.last_command_time = 0
        self.command_cooldown = 0.5
        
        # Background listening thread
        self.listen_thread = None
        
        # NATURAL GAMING VOICE COMMANDS - Maximum 2 words, what gamers actually say
        self.action_mappings = {
            'sprint': {
                'name': 'Sprint',
                'key': 'shift',
                'intents': ['sprint', 'run', 'fast', 'speed', 'go', 'move']
            },
            'crouch': {
                'name': 'Crouch',
                'key': 'c',
                'intents': ['crouch', 'duck', 'hide', 'down', 'low', 'stealth']
            },
            'interact': {
                'name': 'Interact',
                'key': 'e',
                'intents': ['use', 'open', 'grab', 'take', 'get', 'pick']
            },
            'block': {
                'name': 'Block',
                'key': 'e',
                'intents': ['block', 'shield', 'defend', 'guard', 'parry', 'protect']
            },
            'dodge': {
                'name': 'Dodge',
                'key': 'alt',
                'intents': ['dodge', 'roll', 'evade', 'avoid', 'dive', 'move']
            },
            'escape': {
                'name': 'Escape',
                'key': 'f',
                'intents': ['escape', 'flee', 'run', 'away', 'back', 'retreat']
            },
            'focus': {
                'name': 'Listen Mode',
                'key': 'q',
                'intents': ['listen', 'focus', 'hear', 'detect', 'sense', 'radar']
            },
            'reload': {
                'name': 'Reload',
                'key': 'left_click',
                'intents': ['reload', 'ammo', 'bullets', 'clip', 'magazine', 'refill']
            },
            'melee': {
                'name': 'Melee',
                'key': 'f',
                'intents': ['melee', 'punch', 'hit', 'strike', 'knife', 'stab']
            },
            'light': {
                'name': 'Flashlight',
                'key': 't',
                'intents': ['light', 'torch', 'flash', 'lamp', 'beam', 'bright']
            },
            'shake': {
                'name': 'Shake Flashlight',
                'key': 'j',
                'intents': ['shake', 'charge', 'power', 'battery', 'fix', 'recharge']
            },
            'bag': {
                'name': 'Inventory',
                'key': 'tab',
                'intents': ['bag', 'items', 'gear', 'stuff', 'pack', 'inventory']
            },
            'menu': {
                'name': 'Pause Game',
                'key': 'esc',
                'intents': ['menu', 'pause', 'stop', 'options', 'settings', 'break']
            },
            'walk': {
                'name': 'Walk',
                'key': 'shift',
                'intents': ['walk', 'slow', 'normal', 'stop', 'calm', 'easy']
            },
            'sneak': {
                'name': 'Sneak',
                'key': 'ctrl',
                'intents': ['sneak', 'quiet', 'stealth', 'careful', 'silent', 'creep']
            },
            'aim': {
                'name': 'Aim',
                'key': 'right_click',
                'intents': ['aim', 'target', 'sight', 'zoom', 'focus', 'lock']
            },
            'shoot': {
                'name': 'Fire',
                'key': 'left_click',
                'intents': ['shoot', 'fire', 'bang', 'attack', 'kill', 'blast']
            },
            'zoom': {
                'name': 'Scope',
                'key': 'e',
                'intents': ['scope', 'zoom', 'look', 'peek', 'see', 'view']
            },
            'craft': {
                'name': 'Crafting',
                'key': 'space',
                'intents': ['craft', 'make', 'build', 'create', 'work', 'fix']
            },
            'rifle': {
                'name': 'Rifle',
                'key': '2',
                'intents': ['rifle', 'gun', 'long', 'two', 'weapon', 'primary']
            },
            'pistol': {
                'name': 'Pistol',
                'key': '3',
                'intents': ['pistol', 'gun', 'short', 'three', 'side', 'backup']
            },
            'next': {
                'name': 'Next Weapon',
                'key': 'mouse_wheel_up',
                'intents': ['next', 'switch', 'change', 'cycle', 'swap', 'other']
            },
            'back': {
                'name': 'Previous Weapon', 
                'key': 'mouse_wheel_down',
                'intents': ['back', 'previous', 'last', 'return', 'before', 'old']
            },
            'heal': {
                'name': 'Health Kit',
                'key': '7',
                'intents': ['heal', 'health', 'med', 'kit', 'fix', 'bandage']
            },
            'molotov': {
                'name': 'Molotov',
                'key': '9',
                'intents': ['molotov', 'burn', 'flame', 'bottle', 'cocktail', 'nine']
            },
            'throw': {
                'name': 'Throw Item',
                'key': '8',
                'intents': ['throw', 'brick', 'bottle', 'distract', 'toss', 'eight']
            },
            'bomb': {
                'name': 'Bomb',
                'key': '5',
                'intents': ['bomb', 'grenade', 'boom', 'explosive', 'blast', 'five']
            },
            'camera': {
                'name': 'Hand Camera On',
                'key': 'hand_camera_on',
                'intents': ['camera', 'track', 'hand', 'on', 'start', 'enable']
            },
            'stop': {
                'name': 'Hand Camera Off', 
                'key': 'hand_camera_off',
                'intents': ['stop', 'off', 'disable', 'end', 'quit', 'halt']
            },
            'test': {
                'name': 'Test Command',
                'key': 'space',
                'intents': ['test', 'hello', 'check', 'working', 'good', 'yes']
            }
        }
        
        # Determine API status
        api_status = "Google Web Speech (Free)"
        print(f"Voice Controller initialized - {api_status}")
        print(f"Available voice commands: {len(self.action_mappings)} actions")
        
        # Try to load custom voice commands
        self.load_voice_config()
        
        # Initialize Vosk if available
        if VOSK_AVAILABLE and self.voice_engine == "vosk":
            if not self._initialize_vosk():
                print("🔄 Vosk failed - using Google Speech as primary")
                self.voice_engine = "google"
        else:
            print("🔄 Using Google Speech as primary (Vosk not available)")
            self.voice_engine = "google"
        
        self._calibrate_microphone()
    
    def _initialize_microphone(self):
        """Initialize microphone with smart audio isolation"""
        try:
            print("Setting up smart audio isolation...")
            
            # Try to find best microphone device (headset preferred)
            mic_list = sr.Microphone.list_microphone_names()
            
            best_mic_index = None
            for i, name in enumerate(mic_list):
                name_lower = name.lower()
                # Prioritize headset/microphone devices over system audio
                if any(keyword in name_lower for keyword in ['headset', 'microphone', 'mic', 'usb']):
                    best_mic_index = i
                    print(f"Found preferred microphone: {name}")
                    break
            
            if best_mic_index is not None:
                self.microphone = sr.Microphone(device_index=best_mic_index)
                self.microphone_device_index = best_mic_index
            else:
                self.microphone = sr.Microphone()
                self.microphone_device_index = None
                print("Using default microphone")
                
        except Exception as e:
            print(f"Microphone setup warning: {e}")
            self.microphone = sr.Microphone()
            self.microphone_device_index = None
    
    def _calibrate_microphone(self):
        """Calibrate microphone for ambient noise - SIMPLIFIED VERSION"""
        try:
            print("Calibrating microphone for ambient noise...")
            
            with self.microphone as source:
                # Simple calibration like the old working version
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
            # Set energy threshold optimized for headset gaming
            headset_threshold = min(self.energy_threshold, self.recognizer.energy_threshold * 0.7)
            self.recognizer.energy_threshold = headset_threshold
            self.recognizer.dynamic_energy_threshold = self.adaptive_threshold
            self.recognizer.operation_timeout = None
            
            # Gaming-specific tuning for headset
            self.recognizer.pause_threshold = 0.5      # Shorter pause for gaming
            self.recognizer.phrase_threshold = 0.2     # Quick phrase detection
            
            print(f"Microphone calibrated - Energy threshold: {self.recognizer.energy_threshold}")
            
        except Exception as e:
            print(f"Microphone calibration warning: {e}")
    
    def _initialize_vosk(self):
        """Initialize Vosk for ultra-fast gaming voice recognition"""
        try:
            print("Initializing Vosk ultra-fast gaming voice recognition...")
            
            # Try to find Vosk model - use same path as working test
            model_dir = os.path.join(os.path.dirname(__file__), "..", "vosk_models")
            model_paths = [
                # Try the exact path that worked in test_vosk_realtime.py
                os.path.join(os.path.dirname(__file__), "..", "vosk_models", "vosk-model-small-en-us-0.15"),
                "vosk_models/vosk-model-small-en-us-0.15",
                os.path.join(model_dir, "vosk-model-small-en-us-0.15"),
                os.path.join(model_dir, "vosk-model-en-us-0.22"),
                "vosk_models/vosk-model-en-us-0.22"
            ]
            
            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if not model_path:
                print("Vosk model not found - falling back to Google Speech")
                print("To install Vosk model (for ultra-fast gaming):")
                print("1. Download: https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip")
                print("2. Extract to: vosk_models/vosk-model-small-en-us-0.15/")
                print("Note: Google Speech will work as backup - system is still functional")
                return False
            
            # Initialize Vosk model
            self.vosk_model = vosk.Model(model_path)
            self.vosk_recognizer = vosk.KaldiRecognizer(self.vosk_model, self.vosk_sample_rate)
            self.vosk_recognizer.SetMaxAlternatives(1)  # Single best result for speed
            self.vosk_initialized = True
            
            print(f"Vosk initialized with model: {model_path}")
            print(f"Voice engine: Vosk (primary) -> Google Speech (secondary)")
            return True
            
        except Exception as e:
            print(f"Vosk initialization failed: {e}")
            print("Falling back to Google Speech recognition")
            self.voice_engine = "google"
            self.vosk_initialized = False
            return False
    
    def _recognize_with_vosk(self, audio_data):
        """Recognize speech using Vosk with gaming-optimized throttling (from working test)"""
        if not self.vosk_initialized:
            return None, 0
        
        try:
            if self.vosk_recognizer.AcceptWaveform(audio_data):
                # Complete phrase recognized
                result = json.loads(self.vosk_recognizer.Result())
                text = result.get('text', '').strip()
                
                if text:
                    # Use the same command matching as the working test
                    cmd_name, key = self._find_vosk_command_match(text)
                    if cmd_name:
                        print(f"🎮 VOSK COMMAND: '{text}' → {cmd_name} ({key})")
                        return text.lower(), 0.9
                    else:
                        # Return text even if no command match for filtering
                        return text.lower(), 0.9
            else:
                # Partial result with throttling
                partial = json.loads(self.vosk_recognizer.PartialResult())
                partial_text = partial.get('partial', '')
                if partial_text:
                    current_time = time.time()
                    
                    # Only show partial results if meaningful change occurred
                    time_passed = current_time - self.last_partial_time
                    text_changed = abs(len(partial_text) - len(self.last_partial_text)) >= self.partial_change_threshold
                    different_content = partial_text != self.last_partial_text
                    
                    if (time_passed >= self.partial_throttle_interval and different_content) or text_changed:
                        if self.debug_mode:
                            print(f"🎧 Vosk hearing: '{partial_text}'")
                        self.last_partial_text = partial_text
                        self.last_partial_time = current_time
            
            return None, 0
            
        except Exception as e:
            if self.debug_mode:
                print(f"Vosk recognition error: {e}")
            return None, 0
    
    def _find_vosk_command_match(self, text):
        """Find best matching gaming command using ONLY current game's loaded commands with fuzzy"""
        text = text.lower().strip()
        words = text.split()
        
        # Use EXACT same command data as main voice controller - no conversion needed
        # This ensures Vosk matches against the same intents as the main fuzzy matching
        
        # Limit to maximum 2 words for gaming speed
        if len(words) > 2:
            text = ' '.join(words[:2])
            words = words[:2]
        
        # Use EXACT same fuzzy matching logic as main voice controller
        # This ensures both systems match against identical intent lists for current game
        best_match = None
        best_score = 0
        best_action = None
        
        # Check all current game's loaded commands (same as main voice controller)
        for action_id, config in self.action_mappings.items():
            for intent in config['intents']:
                intent_lower = intent.lower()
                
                # Calculate multiple fuzzy match scores (same as main system)
                ratio_score = fuzz.ratio(text, intent_lower)
                partial_score = fuzz.partial_ratio(text, intent_lower)
                token_sort_score = fuzz.token_sort_ratio(text, intent_lower)
                
                # Use the best score from different algorithms
                score = max(ratio_score, partial_score, token_sort_score)
                
                # SAME BOOSTS as main voice controller
                # Boost for exact word matches within the spoken text
                if any(word in intent_lower for word in words):
                    score += 20
                    
                # Boost for short, natural words (what gamers actually say)
                if len(intent_lower) <= 6 and len(words) == 1:
                    score += 15
                
                # Track best match
                if score > best_score:
                    best_score = score
                    best_match = intent
                    best_action = action_id
        
        # Return best match if above higher threshold for more accurate matching
        if best_score >= 80:  # Higher threshold for more accurate Vosk matching
            return best_action, self.action_mappings[best_action]['key']
        
        return None, None
    
    def toggle_voice_engine(self):
        """Toggle between Vosk and Google Speech recognition"""
        if self.voice_engine == "vosk" and VOSK_AVAILABLE:
            self.voice_engine = "google"
            print("🔄 Switched to Google Speech recognition")
        else:
            if VOSK_AVAILABLE:
                self.voice_engine = "vosk"
                if not self.vosk_initialized:
                    self._initialize_vosk()
                print("🔄 Switched to Vosk recognition")
            else:
                print("❌ Vosk not available - staying with Google Speech")
        return self.voice_engine
    
    def toggle_deepgram(self):
        """Toggle Deepgram on/off (disabled by default)"""
        self.use_deepgram = not self.use_deepgram
        status = "ENABLED" if self.use_deepgram else "DISABLED"
        print(f"📡 Deepgram: {status}")
        if self.use_deepgram:
            self.use_deepgram_fallback = True
        else:
            self.use_deepgram_fallback = False
        return self.use_deepgram
    
    def _is_valid_voice_command(self, spoken_text: str, confidence: float, audio) -> bool:
        """Smart filtering to distinguish voice commands from game audio - SIMPLIFIED TO FIX AUDIO FAILURE"""
        if not spoken_text or not spoken_text.strip():
            return False
        
        spoken_text = spoken_text.strip().lower()
        
        # EMERGENCY MODE: Completely disable filtering for debugging
        if self.emergency_mode:
            print(f"EMERGENCY MODE: Accepting '{spoken_text}' without filtering")
            return True
        
        # EMERGENCY FIX: If game audio filter is disabled, be very permissive
        if not self.game_audio_filter:
            # Only reject obviously invalid input
            return len(spoken_text.strip()) > 0 and len(spoken_text) < 200
        
        # MUCH MORE PERMISSIVE FILTERING to fix audio failure
        # 1. Only reject extremely low confidence
        if confidence < self.confidence_threshold:
            if self.debug_mode:
                print(f"Low confidence: {confidence:.2f} < {self.confidence_threshold}")
            return False
        
        # 2. Only reject extremely long text (game dialogue)
        if len(spoken_text) > 100:  # Much more lenient
            if self.debug_mode:
                print(f"Too long: '{spoken_text[:20]}...' ({len(spoken_text)} chars)")
            return False
        
        # 3. Allow more words - only reject very long sentences
        words = spoken_text.split()
        if len(words) > 10:  # Much more lenient
            if self.debug_mode:
                print(f"Too many words: {len(words)} words")
            return False
        
        # 4. SIMPLIFIED pattern filtering - only reject obvious game dialogue
        # If it's a long sentence of common words, likely game dialogue
        if len(words) >= 5:
            common_words = ['the', 'and', 'you', 'that', 'with', 'for', 'are', 'this', 'but', 'not',
                           'they', 'have', 'from', 'one', 'had', 'word', 'what', 'were', 'said']
            common_count = sum(1 for word in words if word in common_words)
            if common_count >= len(words) * 0.7:  # 70% common words
                if self.debug_mode:
                    print(f"Likely game dialogue: '{spoken_text}'")
                return False
        
        # 5. Check if it matches any known voice command pattern (boost valid commands)
        if self._matches_voice_command_pattern(spoken_text):
            return True
        
        # 6. SIMPLIFIED energy check - only reject extremely quiet audio
        try:
            if hasattr(audio, 'frame_data'):
                import audioop
                energy = audioop.rms(audio.frame_data, 2)
                if energy < self.min_voice_energy:
                    if self.debug_mode:
                        print(f"Low voice energy: {energy} < {self.min_voice_energy}")
                    return False
        except:
            pass  # Energy check failed, continue anyway
        
        # DEFAULT: ALLOW MOST THINGS to fix audio failure
        return True
    
    def _matches_voice_command_pattern(self, spoken_text: str) -> bool:
        """Check if text matches known voice command patterns - OPTIMIZED for natural gaming words"""
        spoken_words = spoken_text.lower().split()
        
        # HIGH PRIORITY: Check for exact single word matches (what gamers say)
        gaming_priority_words = {
            'run', 'sprint', 'fast', 'go', 'move', 'speed',
            'duck', 'crouch', 'hide', 'down', 'low',
            'use', 'grab', 'take', 'get', 'pick', 'open',
            'block', 'shield', 'defend', 'guard', 'parry',
            'dodge', 'roll', 'evade', 'dive', 'avoid',
            'aim', 'target', 'sight', 'zoom', 'focus',
            'shoot', 'fire', 'bang', 'attack', 'kill',
            'heal', 'health', 'med', 'fix',
            'light', 'torch', 'flash', 'bright',
            'rifle', 'pistol', 'gun', 'weapon',
            'next', 'switch', 'change', 'swap',
            'bomb', 'grenade', 'throw', 'boom',
            'test', 'hello', 'check', 'working'
        }
        
        # If ANY word is a high-priority gaming word, immediately accept
        for word in spoken_words:
            if word in gaming_priority_words:
                return True
        
        # Check exact matches in our intents
        for action_id, config in self.action_mappings.items():
            for intent in config['intents']:
                if intent in spoken_text or any(word in intent for word in spoken_words):
                    return True
        
        # Check fuzzy matches for known commands (lower threshold for natural words)
        best_score = 0
        for action_id, config in self.action_mappings.items():
            for intent in config['intents']:
                score = fuzz.ratio(spoken_text, intent)
                if score > best_score:
                    best_score = score
        
        # More lenient fuzzy matching for natural gaming language
        return best_score > 60  # Lowered from 70 for better recognition
            
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
        """Enhanced fuzzy matching optimized for NATURAL GAMING LANGUAGE"""
        best_match = None
        best_score = 0
        best_action = None
        
        spoken_text = spoken_text.lower().strip()
        spoken_words = spoken_text.split()
        is_contextual = self.add_contextual_boost(spoken_text)
        
        # PRIORITY 1: Direct single-word gaming commands (highest priority)
        if len(spoken_words) == 1:
            single_word = spoken_words[0]
            for action_id, config in self.action_mappings.items():
                for intent in config['intents']:
                    if single_word == intent.lower():
                        print(f"🎯 EXACT MATCH: '{single_word}' → {action_id}")
                        return (action_id, config['key'], 100)
        
        # PRIORITY 2: Check all intents with optimized scoring
        for action_id, config in self.action_mappings.items():
            is_combo = config.get('is_combo', False)
            
            for intent in config['intents']:
                intent_lower = intent.lower()
                
                # Calculate multiple fuzzy match scores
                ratio_score = fuzz.ratio(spoken_text, intent_lower)
                partial_score = fuzz.partial_ratio(spoken_text, intent_lower)
                token_sort_score = fuzz.token_sort_ratio(spoken_text, intent_lower)
                
                # Use the best score from different algorithms
                score = max(ratio_score, partial_score, token_sort_score)
                
                # NATURAL LANGUAGE BOOSTS
                # Boost for exact word matches within the spoken text
                if any(word in intent_lower for word in spoken_words):
                    score += 20
                    
                # Boost for short, natural words (what gamers actually say)
                if len(intent_lower) <= 6 and len(spoken_words) == 1:
                    score += 15
                
                # COMBO KEY PRIORITY BOOST
                if is_combo:
                    score += 10
                
                # High-priority gaming words get extra boost
                priority_words = ['run', 'shoot', 'heal', 'dodge', 'aim', 'block', 'use', 'light']
                if any(word in spoken_text for word in priority_words):
                    score += 10
                
                # Track best match
                if score > best_score:
                    best_score = score
                    best_match = intent
                    best_action = action_id
        
        # Return best match if above threshold
        if best_score >= 60:  # Lowered threshold for natural gaming language
            print(f"🎮 MATCHED: '{spoken_text}' → '{best_match}' (score: {best_score})")
            return (best_action, self.action_mappings[best_action]['key'], best_score)
        
        # No good match found
        if best_score > 0:
            print(f"❌ LOW SCORE: '{spoken_text}' → '{best_match}' (score: {best_score}) - threshold: 60")
        
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
        combo_indicator = " [COMBO]" if action_config.get('is_combo', False) else ""
        print(f"OK {mode_indicator} Voice Command: '{spoken_text}' -> '{matched_intent}' -> {action_config['name']} ({confidence}%){combo_indicator}")
        
        # Audio feedback (disabled for gaming to prevent blocking)
        # if self.audio_feedback:
        #     threading.Thread(target=self._speak_feedback, args=(action_config['name'],), daemon=True).start()
        
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
    
    def _recognize_with_deepgram(self, audio):
        """Ultra-fast Deepgram recognition for gaming"""
        if not self.deepgram_api_key:
            return None
            
        try:
            # Create proper WAV format for Deepgram
            import io
            import struct
            
            # Get raw audio data
            raw_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
            
            # Create WAV header
            wav_buffer = io.BytesIO()
            
            # WAV header
            wav_buffer.write(b'RIFF')
            wav_buffer.write(struct.pack('<I', 36 + len(raw_data)))
            wav_buffer.write(b'WAVE')
            wav_buffer.write(b'fmt ')
            wav_buffer.write(struct.pack('<I', 16))  # PCM format chunk size
            wav_buffer.write(struct.pack('<H', 1))   # PCM format
            wav_buffer.write(struct.pack('<H', 1))   # 1 channel
            wav_buffer.write(struct.pack('<I', 16000))  # Sample rate
            wav_buffer.write(struct.pack('<I', 32000))  # Byte rate
            wav_buffer.write(struct.pack('<H', 2))   # Block align
            wav_buffer.write(struct.pack('<H', 16))  # Bits per sample
            wav_buffer.write(b'data')
            wav_buffer.write(struct.pack('<I', len(raw_data)))
            wav_buffer.write(raw_data)
            
            wav_data = wav_buffer.getvalue()
            
            # Direct API call
            url = "https://api.deepgram.com/v1/listen"
            headers = {
                "Authorization": f"Token {self.deepgram_api_key}",
                "Content-Type": "audio/wav"
            }
            params = {
                "model": "nova-2",
                "language": "en-US",
                "smart_format": "true",
                "punctuate": "false"  # Disable for faster processing
            }
            
            response = requests.post(
                url, 
                headers=headers, 
                params=params, 
                data=wav_data,
                timeout=3  # Fast timeout for gaming
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('results') and result['results'].get('channels'):
                    alternatives = result['results']['channels'][0].get('alternatives', [])
                    if alternatives:
                        transcript = alternatives[0].get('transcript', '').strip()
                        confidence = alternatives[0].get('confidence', 0)
                        return transcript, confidence
            else:
                if self.debug_mode:
                    print(f"Deepgram API error: {response.status_code}")
            
            return None, 0
            
        except Exception as e:
            if self.debug_mode:
                print(f"Deepgram error: {e}")
            return None, 0
    
    def _listen_loop(self):
        """Gaming optimized listen loop - Vosk primary, Google Speech secondary"""
        current_engine = self.voice_engine
        print(f"🎤 Voice listening loop started - ENGINE: {current_engine.upper()}")
        print(f"🔧 AUDIO SETTINGS: Confidence={self.confidence_threshold}, Energy={self.min_voice_energy}, Filter={self.game_audio_filter}")
        print(f"🎯 Available engines: Vosk={VOSK_AVAILABLE}, Google=True, Deepgram={self.use_deepgram}")
        
        if current_engine == "vosk" and self.vosk_initialized:
            self._listen_loop_vosk()
        else:
            self._listen_loop_google()
    
    def _listen_loop_vosk(self):
        """Vosk-based ultra-fast listening loop for gaming - EXACT copy from working test"""
        print("🎮 VOSK ULTRA-FAST GAMING MODE")
        
        # Use exact same audio setup as working test
        audio_interface = pyaudio.PyAudio()
        
        try:
            stream = audio_interface.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.vosk_sample_rate,
                input=True,
                frames_per_buffer=self.vosk_chunk_size
            )
            
            last_recognition_time = time.time()
            command_cooldown = 0.5
            
            print("🎤 Vosk listening for gaming commands...")
            
            while self.listening and self.enabled:
                try:
                    # Read audio chunk - EXACT same as working test
                    audio_data = stream.read(self.vosk_chunk_size, exception_on_overflow=False)
                    
                    # Process with EXACT same logic as working test
                    result = self._process_vosk_audio_chunk(audio_data)
                    if result:
                        cmd_name, key = result
                        current_time = time.time()
                        if current_time - last_recognition_time > command_cooldown:
                            last_recognition_time = current_time
                            print(f"✅ VOSK EXECUTED: {cmd_name} ({key})")
                            # Send to voice command handler using the original text
                            self._on_speech_recognized(cmd_name)
                
                except Exception as e:
                    if self.debug_mode:
                        print(f"Vosk listening error: {e}")
                    time.sleep(0.001)
                    continue
                    
        except Exception as e:
            print(f"❌ Vosk audio stream error: {e}")
            print("🔄 Falling back to Google Speech")
            self._listen_loop_google()
        finally:
            try:
                stream.stop_stream()
                stream.close()
                audio_interface.terminate()
            except:
                pass
    
    def _process_vosk_audio_chunk(self, audio_data):
        """Process single audio chunk - EXACT copy from working test"""
        if not self.vosk_initialized:
            return None
            
        try:
            if self.vosk_recognizer.AcceptWaveform(audio_data):
                # Complete phrase recognized
                result = json.loads(self.vosk_recognizer.Result())
                text = result.get('text', '').strip()
                
                if text:
                    current_time = time.time()
                    if current_time - self.last_command_time > self.command_cooldown:
                        self.last_command_time = current_time
                        
                        # Move to new line for final result
                        print()  # Just a clean new line
                        
                        # Find gaming command match using EXACT same logic
                        cmd_name, key = self._find_vosk_command_match(text)
                        
                        if cmd_name:
                            print(f"🎮 COMMAND: '{text}' → {cmd_name} ({key})")
                            print("🔊 Ready for next command...")
                            return cmd_name, key
                        else:
                            print(f"❓ UNKNOWN: '{text}' (not a gaming command)")
                            print("🔊 Ready for next command...")
                            
            else:
                # Partial result for real-time feedback with throttling
                partial = json.loads(self.vosk_recognizer.PartialResult())
                partial_text = partial.get('partial', '')
                if partial_text:
                    current_time = time.time()
                    
                    # CHECK: If partial contains a valid command, apply smoothing like eye movement
                    cmd_name, key = self._find_vosk_command_match(partial_text)
                    
                    # No hardcoded debug - let fuzzy matching work naturally
                    
                    if cmd_name and len(partial_text.split()) <= 2:  # Single/double word commands only
                        
                        # Apply command smoothing/debouncing
                        current_time = time.time()
                        
                        # Add to buffer for consistency checking
                        self.command_execution_buffer.append(cmd_name)
                        if len(self.command_execution_buffer) > self.command_buffer_size:
                            self.command_execution_buffer.pop(0)
                        
                        # Check if we have consistent detections AND enough time has passed
                        consistent_commands = len(set(self.command_execution_buffer)) == 1
                        enough_time_passed = current_time - self.last_execution_time > self.command_debounce_interval
                        different_command = cmd_name != self.last_executed_command
                        
                        if len(self.command_execution_buffer) >= self.command_buffer_size and consistent_commands and (enough_time_passed or different_command):
                            print(f"🎮 SMOOTHED COMMAND: '{partial_text}' → {cmd_name} ({key})")
                            print("🔊 Ready for next command...")
                            self.last_executed_command = cmd_name
                            self.last_execution_time = current_time
                            self.command_execution_buffer = []  # Clear buffer after execution
                            return cmd_name, key
                        else:
                            # Show debug info for smoothing
                            if self.debug_mode:
                                buffer_status = f"Buffer: {len(self.command_execution_buffer)}/{self.command_buffer_size}"
                                time_status = f"Time: {current_time - self.last_execution_time:.1f}s"
                                print(f"🔄 Smoothing: {cmd_name} | {buffer_status} | {time_status}")
                            return None  # Don't execute yet, still smoothing
                    
                    # Only show partial results if meaningful change occurred
                    time_passed = current_time - self.last_partial_time
                    text_changed = abs(len(partial_text) - len(self.last_partial_text)) >= self.partial_change_threshold
                    different_content = partial_text != self.last_partial_text
                    
                    if (time_passed >= self.partial_throttle_interval and different_content) or text_changed:
                        print(f"🎧 Hearing: '{partial_text}'")
                        self.last_partial_text = partial_text
                        self.last_partial_time = current_time
            
            return None
            
        except Exception as e:
            if self.debug_mode:
                print(f"Vosk processing error: {e}")
            return None
    
    def _try_google_fallback(self, audio_data, current_time, last_recognition_time, command_cooldown):
        """Try Google Speech as fallback when Vosk fails"""
        try:
            # Convert audio data to speech_recognition format
            import io
            audio_buffer = io.BytesIO()
            
            # Create minimal WAV header for speech_recognition
            audio_buffer.write(b'RIFF')
            audio_buffer.write((len(audio_data) + 36).to_bytes(4, 'little'))
            audio_buffer.write(b'WAVE')
            audio_buffer.write(b'fmt ')
            audio_buffer.write((16).to_bytes(4, 'little'))
            audio_buffer.write((1).to_bytes(2, 'little'))
            audio_buffer.write((1).to_bytes(2, 'little'))
            audio_buffer.write((self.vosk_sample_rate).to_bytes(4, 'little'))
            audio_buffer.write((self.vosk_sample_rate * 2).to_bytes(4, 'little'))
            audio_buffer.write((2).to_bytes(2, 'little'))
            audio_buffer.write((16).to_bytes(2, 'little'))
            audio_buffer.write(b'data')
            audio_buffer.write((len(audio_data)).to_bytes(4, 'little'))
            audio_buffer.write(audio_data)
            
            audio_buffer.seek(0)
            audio = sr.AudioData(audio_data, self.vosk_sample_rate, 2)
            
            # Try Google Speech
            spoken_text = self.recognizer.recognize_google(audio, language='en-US')
            if spoken_text and current_time - last_recognition_time > command_cooldown:
                print(f"🔄 GOOGLE FALLBACK: '{spoken_text}'")
                
                if self._is_valid_voice_command(spoken_text, 0.8, audio):
                    print(f"✅ GOOGLE ACCEPTED: '{spoken_text}'")
                    self._on_speech_recognized(spoken_text)
                else:
                    print(f"❌ GOOGLE FILTERED: '{spoken_text}'")
                    
        except Exception as e:
            if self.debug_mode:
                print(f"Google fallback failed: {e}")
    
    def _listen_loop_google(self):
        """Google Speech listening loop (fallback/secondary)"""
        print("🎤 GOOGLE SPEECH MODE (Secondary)")
        
        loop_count = 0
        last_recognition_time = time.time()
        
        # Keep microphone context open for fastest response
        with self.microphone as source:
            # Quick re-calibration
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            while self.listening and self.enabled:
                try:
                    loop_count += 1
                    current_time = time.time()
                    
                    # Show periodic status
                    if loop_count % 50 == 0:
                        time_since_last = current_time - last_recognition_time
                        print(f"Google listening active - loop {loop_count} (last: {time_since_last:.1f}s ago)")
                    
                    # Listen for audio
                    audio = self.recognizer.listen(
                        source, 
                        timeout=self.recognition_timeout,
                        phrase_time_limit=self.phrase_timeout
                    )
                    
                    # Google Speech recognition
                    try:
                        if self.google_with_confidence:
                            try:
                                result = self.recognizer.recognize_google(audio, language='en-US', show_all=True)
                                if result and 'alternative' in result and result['alternative']:
                                    spoken_text = result['alternative'][0]['transcript']
                                    confidence = result['alternative'][0].get('confidence', 0.5)
                                    
                                    print(f"🔄 GOOGLE: '{spoken_text}' (conf: {confidence:.2f})")
                                    
                                    if self._is_valid_voice_command(spoken_text, confidence, audio):
                                        last_recognition_time = current_time
                                        print(f"✅ GOOGLE ACCEPTED: '{spoken_text}'")
                                        self._on_speech_recognized(spoken_text)
                                    else:
                                        print(f"❌ GOOGLE FILTERED: '{spoken_text}'")
                            except:
                                # Simple fallback
                                spoken_text = self.recognizer.recognize_google(audio, language='en-US')
                                if spoken_text:
                                    print(f"🔄 GOOGLE (simple): '{spoken_text}'")
                                    if self._is_valid_voice_command(spoken_text, 0.7, audio):
                                        last_recognition_time = current_time
                                        print(f"✅ GOOGLE ACCEPTED: '{spoken_text}'")
                                        self._on_speech_recognized(spoken_text)
                                    else:
                                        print(f"❌ GOOGLE FILTERED: '{spoken_text}'")
                        else:
                            spoken_text = self.recognizer.recognize_google(audio, language='en-US')
                            if spoken_text:
                                print(f"🔄 GOOGLE: '{spoken_text}'")
                                if self._is_valid_voice_command(spoken_text, 0.7, audio):
                                    last_recognition_time = current_time
                                    print(f"✅ GOOGLE ACCEPTED: '{spoken_text}'")
                                    self._on_speech_recognized(spoken_text)
                                else:
                                    print(f"❌ GOOGLE FILTERED: '{spoken_text}'")
                            
                    except sr.UnknownValueError:
                        # Try Deepgram if enabled
                        if self.use_deepgram and self.deepgram_api_key:
                            try:
                                deepgram_result = self._recognize_with_deepgram(audio)
                                if deepgram_result and deepgram_result[0]:
                                    spoken_text, confidence = deepgram_result
                                    if spoken_text.strip() and self._is_valid_voice_command(spoken_text, confidence, audio):
                                        last_recognition_time = current_time
                                        print(f"📡 DEEPGRAM: '{spoken_text}' ({confidence:.2f})")
                                        self._on_speech_recognized(spoken_text)
                            except Exception as deepgram_error:
                                if self.debug_mode:
                                    print(f"Deepgram error: {deepgram_error}")
                        
                    except sr.RequestError as e:
                        if self.debug_mode:
                            print(f"Google Speech error: {e}")
                        time.sleep(0.1)
                        
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    print(f"Listen error: {e}")
                    time.sleep(0.05)
                    continue
    
    def _on_speech_recognized(self, spoken_text: str):
        """Handle recognized speech with full fuzzy matching and AI processing"""
        try:
            # Process with full fuzzy matching and AI intent parsing
            result = self.execute_voice_command(spoken_text)
            
            # Return result for main system to execute the key press
            if result:
                return result
            return None
        except Exception as e:
            print(f"Voice command processing error: {e}")
            return None
    
    def toggle(self) -> bool:
        """Toggle voice recognition on/off"""
        self.enabled = not self.enabled
        if self.enabled:
            print("Voice recognition ENABLED")
        else:
            print("Voice recognition DISABLED")
        return self.enabled
    
    def set_audio_feedback(self, enabled: bool):
        """Enable/disable audio feedback"""
        self.audio_feedback = enabled
        status = "ENABLED" if enabled else "DISABLED"
        print(f"Audio feedback {status}")
    
    def recalibrate_microphone(self):
        """Recalibrate microphone sensitivity"""
        print("🔄 Recalibrating microphone...")
        self._calibrate_microphone()
        print("✅ Microphone recalibrated!")
    
    def list_available_commands(self):
        """Print all available voice commands with COMBO PRIORITY"""
        print("\nAvailable Voice Commands:")
        print("=" * 60)
        
        # Show combo commands first
        combo_commands = []
        normal_commands = []
        
        for action_id, config in self.action_mappings.items():
            intents_str = ", ".join(f"'{intent}'" for intent in config['intents'])
            command_line = f"{config['name']} ({config['key']}): {intents_str}"
            
            if config.get('is_combo', False):
                combo_commands.append(f"[COMBO-PRIORITY] {command_line}")
            else:
                normal_commands.append(command_line)
        
        # Display combo commands first
        if combo_commands:
            print("COMBO COMMANDS (HIGH PRIORITY):")
            for cmd in combo_commands:
                print(f"  {cmd}")
            print()
        
        if normal_commands:
            print("NORMAL COMMANDS:")
            for cmd in normal_commands:
                print(f"  {cmd}")
        
        print("=" * 60)
        print(f"Total: {len(combo_commands)} combo + {len(normal_commands)} normal = {len(self.action_mappings)} commands")
    
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
    
    def add_custom_voice_command(self, action_name: str, voice_phrase: str, key_to_press: str):
        """Add a completely new voice command with voice phrase and key"""
        # Create new action mapping
        new_action = {
            "key": key_to_press,
            "name": voice_phrase.title(),  # Capitalize for display
            "intents": [voice_phrase]  # Start with the main phrase
        }
        
        # Add to action mappings
        self.action_mappings[action_name] = new_action
        
        print(f"Created new voice command: '{voice_phrase}' → {key_to_press}")
        return True
    
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
                print(f"Total commands loaded: {len(self.action_mappings)}")
                return True
            else:
                print(f"File {filename} not found, using defaults")
        except Exception as e:
            print(f"Failed to load voice commands: {e}")
        return False
    
    def clear_and_reload_commands(self, filename="voice_commands.json"):
        """Clear all commands and reload from JSON file"""
        print("🔄 Clearing all voice commands and reloading from JSON...")
        
        # Clear all existing commands
        self.action_mappings = {}
        print("✅ All voice commands cleared")
        
        # Reload from JSON
        success = self.load_voice_config(filename)
        if success:
            print(f"✅ Reloaded {len(self.action_mappings)} voice commands from {filename}")
            
            # Detect and set aim key for wink mapping
            self._detect_and_set_dynamic_aim_key()
            
            # Show sample of loaded commands
            print("\nSample loaded commands:")
            for i, (action_id, config) in enumerate(self.action_mappings.items()):
                if i < 5:  # Show first 5 commands
                    print(f"  {action_id}: {config['intents']} → {config['key']}")
                elif i == 5:
                    print(f"  ... and {len(self.action_mappings) - 5} more")
                    break
        else:
            print("❌ Failed to reload commands")
        
        return success
    
    def _detect_and_set_dynamic_aim_key(self):
        """Detect aim key from loaded commands and set it in input controller"""
        # This will be called by the main system to update the input controller
        # We need access to the input controller for this
        pass
    
    def reset_to_default_commands(self):
        """DEPRECATED - Use clear_and_add_game_commands() instead for proper game isolation"""
        print("WARNING: reset_to_default_commands() is deprecated")
        print("Use clear_and_add_game_commands() to properly reset for each game")
        
        # Clear everything - no defaults should persist between games
        self.action_mappings = {}
        print("All voice commands cleared - add game-specific commands only")
        return True
    
    def clear_and_add_game_commands(self, game_voice_commands: dict):
        """RESET ONLY VOICE COMMANDS for new game - PRESERVE gesture training data"""
        print("RESETTING VOICE COMMANDS FOR NEW GAME")
        print("=" * 50)
        print("NOTE: Preserving all trained gesture data - only voice commands reset")
        
        # COMPLETELY CLEAR all previous VOICE commands only
        self.action_mappings = {}
        print("All previous voice commands cleared")
        
        # Add ONLY the new game-specific VOICE commands
        if game_voice_commands:
            print(f"Adding {len(game_voice_commands)} NEW game-specific voice commands...")
            
            for action_name, command_details in game_voice_commands.items():
                if isinstance(command_details, dict):
                    key = command_details.get('key', 'space')
                    phrases = command_details.get('phrases', [action_name])
                    description = command_details.get('description', f'{action_name} action')
                    
                    # Detect if this is a combo key and mark priority
                    is_combo = '+' in key or 'combo' in key.lower()
                    priority = 'high' if is_combo else 'normal'
                    
                    # Add new voice command with combo priority
                    self.action_mappings[action_name] = {
                        'name': description,
                        'key': key,
                        'intents': phrases,
                        'is_combo': is_combo,
                        'priority': priority
                    }
                    
                    combo_indicator = " [COMBO-PRIORITY]" if is_combo else ""
                    print(f"   + {action_name} -> {key} ({', '.join(phrases)}){combo_indicator}")
            
            print(f"TOTAL ACTIVE VOICE COMMANDS: {len(self.action_mappings)}")
            print("All voice commands are now specific to current game only")
            print("Trained gesture data remains intact for reuse")
        else:
            print("WARNING: No game commands provided - voice control will be empty")
        
        # RESET audio recognition settings for new game
        self._reset_audio_for_new_game()
        
        return True
    
    def _reset_audio_for_new_game(self):
        """Reset all audio recognition settings when switching games"""
        print("Resetting audio recognition for new game...")
        
        # Clear command history from previous game
        self.recent_commands = []
        
        # Reset energy threshold to default for new audio environment
        self.recognizer.energy_threshold = self.energy_threshold
        
        # Force recalibration for new game audio
        try:
            print("Recalibrating microphone for new game environment...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1.5)
                
            # Apply headset optimization again
            headset_threshold = min(self.energy_threshold, self.recognizer.energy_threshold * 0.7)
            self.recognizer.energy_threshold = headset_threshold
            
            print(f"Audio recalibrated - New threshold: {self.recognizer.energy_threshold:.0f}")
            
        except Exception as e:
            print(f"Audio recalibration warning: {e}")
            
        print("Audio system reset complete for new game")
    
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
            
            # Create game context from available commands with COMBO PRIORITY
            combo_actions = []
            normal_actions = []
            
            for action_id, config in self.action_mappings.items():
                action_line = f"{action_id}: {config['name']} - {', '.join(config['intents'])}"
                if config.get('is_combo', False):
                    combo_actions.append(f"[COMBO] {action_line}")
                else:
                    normal_actions.append(action_line)
            
            # Put combo actions first in context
            all_actions = combo_actions + normal_actions
            game_context = "\n".join(all_actions)
            
            prompt = f"""You are a gaming voice command parser with COMBO KEY PRIORITY.

Available game actions:
{game_context}

User said: "{spoken_text}"

IMPORTANT PRIORITY RULES:
1. [COMBO] actions have HIGHEST priority - prefer these when there's any reasonable match
2. Combo keys are complex moves that should be prioritized over simple keys
3. If user says "dodge" or "roll", strongly prefer [COMBO] dodge/roll over simple actions

Parse this into ONE action ID from the list above. Consider context like:
- "roll roll roll" likely means dodge/crouch for defensive action
- "go go go" likely means sprint/run  
- "get down" means crouch
- "move fast" means sprint
- "dodge" or "roll" should prefer [COMBO] versions if available

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
    
    def toggle_game_audio_filter(self):
        """Toggle smart game audio filtering"""
        self.game_audio_filter = not self.game_audio_filter
        status = "ENABLED" if self.game_audio_filter else "DISABLED"
        print(f"🎮 Game audio filtering: {status}")
        
        if self.game_audio_filter:
            print(f"   Confidence threshold: {self.confidence_threshold}")
            print(f"   Voice energy minimum: {self.min_voice_energy}")
        
        return self.game_audio_filter
    
    def adjust_audio_sensitivity(self, increase: bool = True):
        """Adjust audio filtering sensitivity"""
        if increase:
            # More sensitive - lower thresholds
            self.confidence_threshold = max(0.2, self.confidence_threshold - 0.1)  # More aggressive lowering
            self.min_voice_energy = max(25, self.min_voice_energy - 25)  # More aggressive lowering
            print(f"🎤 More sensitive - Confidence: {self.confidence_threshold:.2f}, Energy: {self.min_voice_energy}")
        else:
            # Less sensitive - higher thresholds  
            self.confidence_threshold = min(0.95, self.confidence_threshold + 0.1)
            self.min_voice_energy = min(500, self.min_voice_energy + 50)
            print(f"🎤 Less sensitive - Confidence: {self.confidence_threshold:.2f}, Energy: {self.min_voice_energy}")
        
        return self.confidence_threshold, self.min_voice_energy
    
    def toggle_emergency_mode(self):
        """Toggle emergency mode - disables ALL filtering for debugging"""
        self.emergency_mode = not self.emergency_mode
        status = "ENABLED" if self.emergency_mode else "DISABLED"
        print(f"🚨 EMERGENCY MODE: {status}")
        if self.emergency_mode:
            print("   ALL filtering disabled - will accept any recognized speech")
        else:
            print(f"   Filtering restored - Confidence: {self.confidence_threshold}, Filter: {self.game_audio_filter}")
        return self.emergency_mode

    def force_complete_reset(self):
        """Force complete reset of VOICE CONTROLLER ONLY - preserves gesture training"""
        print("FORCING COMPLETE VOICE CONTROLLER RESET")
        print("=" * 50)
        print("NOTE: This only resets voice controls - gesture training data is preserved")
        
        # Stop listening during reset
        was_listening = self.listening
        if was_listening:
            self.stop_listening()
        
        # Clear ALL voice commands only
        self.action_mappings = {}
        
        # Clear ALL voice recognition history
        self.recent_commands = []
        
        # Reset ALL audio settings to initial state
        self.energy_threshold = 100  # Reset to headset default
        self.recognition_timeout = 0.5
        self.phrase_timeout = 1.2
        
        # Force microphone recalibration
        try:
            print("Performing complete microphone recalibration...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=2.0)
            
            # Apply fresh headset optimization
            headset_threshold = min(self.energy_threshold, self.recognizer.energy_threshold * 0.7)
            self.recognizer.energy_threshold = headset_threshold
            self.recognizer.dynamic_energy_threshold = self.adaptive_threshold
            self.recognizer.pause_threshold = 0.5
            self.recognizer.phrase_threshold = 0.2
            
            print(f"Voice reset complete - Ready for new game voice commands")
            print(f"Energy threshold: {self.recognizer.energy_threshold:.0f}")
            print("Gesture training data remains intact")
            
        except Exception as e:
            print(f"Reset warning: {e}")
        
        # Restart listening if it was active
        if was_listening:
            self.start_listening()
            
        print("VOICE CONTROLLER RESET - Gesture data preserved, ready for new voice commands")
        return True

    def test_headset_sensitivity(self):
        """Test headset microphone sensitivity - useful for debugging"""
        print("\nHEADSET SENSITIVITY TEST")
        print("=" * 40)
        print("Speak normally into your headset for 5 seconds...")
        print("This will help determine if sensitivity is correct.")
        
        try:
            with self.microphone as source:
                print("Listening...")
                
                # Test current threshold
                for i in range(5):
                    try:
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=2)
                        try:
                            text = self.recognizer.recognize_google(audio, language='en-US')
                            print(f"Heard: '{text}' (Energy: {self.recognizer.energy_threshold:.0f})")
                        except sr.UnknownValueError:
                            print(f"Audio detected but not understood (Energy: {self.recognizer.energy_threshold:.0f})")
                    except sr.WaitTimeoutError:
                        print(f"No audio detected (Energy threshold: {self.recognizer.energy_threshold:.0f})")
                
                print("\nResults:")
                if self.recognizer.energy_threshold > 200:
                    print("   • Threshold too high - try speaking louder or closer to mic")
                elif self.recognizer.energy_threshold < 50:
                    print("   • Threshold very low - good for quiet headset use")
                else:
                    print("   • Threshold looks good for headset gaming")
                    
        except Exception as e:
            print(f"Test failed: {e}")

    def cleanup(self):
        """Clean up resources"""
        self.stop_listening()
        try:
            self.tts.stop()
        except:
            pass
        print("Voice controller cleanup complete")