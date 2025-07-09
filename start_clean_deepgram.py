#!/usr/bin/env python3
"""
Clean startup with Deepgram speech recognition for better accuracy
"""

import sys
import os
import asyncio
import threading
import time
import pyaudio
import wave
import tempfile
from typing import Optional, Dict

# Simple print override
def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except:
        import builtins
        builtins.print("Audio system running...")

import builtins
builtins.print = safe_print

# Set path
sys.path.append('src')

# Deepgram imports
try:
    from deepgram import (
        DeepgramClient,
        PrerecordedOptions,
        LiveTranscriptionEvents,
        LiveOptions
    )
    DEEPGRAM_AVAILABLE = True
except ImportError:
    DEEPGRAM_AVAILABLE = False
    print("Deepgram SDK not available")

class DeepgramVoiceController:
    def __init__(self, deepgram_api_key: str):
        self.api_key = deepgram_api_key
        self.deepgram = DeepgramClient(api_key)
        self.listening = False
        self.enabled = True
        
        # Audio settings
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        self.audio_format = pyaudio.paInt16
        
        # PyAudio setup
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Voice commands (same as working voice controller)
        self.action_mappings = {
            'sprint': {'name': 'Sprint', 'key': 'shift', 'intents': ['sprint', 'run', 'fast', 'hurry', 'speed']},
            'crouch': {'name': 'Crouch', 'key': 'c', 'intents': ['crouch', 'duck', 'hide', 'stealth', 'prone']},
            'interact': {'name': 'Interact', 'key': 'e', 'intents': ['interact', 'use', 'open', 'pick up', 'grab', 'take']},
            'fire': {'name': 'Fire', 'key': 'left_click', 'intents': ['fire', 'shoot', 'bang', 'attack', 'kill']},
            'flashlight': {'name': 'Flashlight', 'key': 't', 'intents': ['flashlight', 'light', 'torch', 'illuminate', 'flash']},
            'health_kit': {'name': 'Health Kit', 'key': '7', 'intents': ['health', 'heal', 'medkit', 'first aid', 'bandage', 'med']},
            'reload': {'name': 'Reload', 'key': 'left_click', 'intents': ['reload', 'ammo', 'bullets', 'magazine', 'clip']},
            'aim': {'name': 'Aim', 'key': 'right_click', 'intents': ['aim', 'target', 'sight', 'focus aim']},
            'dodge': {'name': 'Dodge', 'key': 'alt', 'intents': ['dodge', 'evade', 'avoid', 'roll']},
            'backpack': {'name': 'Backpack', 'key': 'tab', 'intents': ['backpack', 'inventory', 'items', 'bag', 'gear']},
            'test': {'name': 'Test Command', 'key': 'space', 'intents': ['hello', 'hi', 'test', 'check']}
        }
        
        print(f"Deepgram Voice Controller initialized")
        print(f"Available commands: {len(self.action_mappings)} actions")
        
        # Callback function
        self._on_speech_recognized = None
    
    def find_best_match(self, spoken_text: str) -> Optional[Dict]:
        """Find best matching command using simple word matching"""
        spoken_text = spoken_text.lower().strip()
        
        # Direct word matching for better accuracy
        best_match = None
        best_score = 0
        
        for action_id, config in self.action_mappings.items():
            for intent in config['intents']:
                # Check for exact word matches
                spoken_words = set(spoken_text.split())
                intent_words = set(intent.lower().split())
                
                # Calculate overlap score
                if intent_words & spoken_words:  # If there's any overlap
                    overlap = len(intent_words & spoken_words)
                    total = len(intent_words)
                    score = (overlap / total) * 100
                    
                    if score > best_score and score >= 50:  # 50% overlap required
                        best_score = score
                        best_match = {
                            'action_id': action_id,
                            'key': config['key'],
                            'name': config['name'],
                            'confidence': score,
                            'matched_intent': intent
                        }
        
        return best_match
    
    def execute_voice_command(self, spoken_text: str):
        """Execute voice command if recognized"""
        print(f"Deepgram heard: '{spoken_text}'")
        
        # Filter out very long sentences
        if len(spoken_text.split()) > 5:
            print(f"Ignored (too long): '{spoken_text}'")
            return None
        
        result = self.find_best_match(spoken_text)
        if result:
            print(f"DEEPGRAM MATCH: '{spoken_text}' -> {result['name']} ({result['confidence']:.1f}%)")
            return result
        else:
            print(f"No match found for: '{spoken_text}'")
            return None
    
    async def transcribe_audio_chunk(self, audio_data: bytes):
        """Transcribe audio chunk using Deepgram"""
        try:
            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                # Write WAV header and data
                with wave.open(temp_file.name, 'wb') as wav_file:
                    wav_file.setnchannels(self.channels)
                    wav_file.setsampwidth(self.audio.get_sample_size(self.audio_format))
                    wav_file.setframerate(self.sample_rate)
                    wav_file.writeframes(audio_data)
                
                # Read the file and transcribe
                with open(temp_file.name, 'rb') as audio_file:
                    buffer_data = audio_file.read()
                
                payload = {'buffer': buffer_data}
                
                options = PrerecordedOptions(
                    model="nova-2",
                    language="en-US",
                    smart_format=True,
                    punctuate=True,
                    diarize=False
                )
                
                response = self.deepgram.listen.prerecorded.v("1").transcribe_file(
                    payload, options, timeout=5
                )
                
                # Extract transcript
                if response.results and response.results.channels:
                    alternatives = response.results.channels[0].alternatives
                    if alternatives and alternatives[0].transcript:
                        transcript = alternatives[0].transcript.strip()
                        confidence = alternatives[0].confidence
                        
                        # Clean up temp file
                        os.unlink(temp_file.name)
                        
                        return transcript, confidence
                
                # Clean up temp file
                os.unlink(temp_file.name)
                return None, 0
                
        except Exception as e:
            print(f"Deepgram transcription error: {e}")
            return None, 0
    
    def start_listening(self):
        """Start listening with Deepgram"""
        if self.listening:
            return
        
        self.listening = True
        print("Starting Deepgram voice recognition...")
        
        # Start audio capture thread
        self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listen_thread.start()
        print("Deepgram voice recognition started!")
    
    def stop_listening(self):
        """Stop listening"""
        self.listening = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        print("Deepgram voice recognition stopped")
    
    def _listen_loop(self):
        """Audio capture and transcription loop"""
        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            print("Deepgram listening loop started...")
            
            # Buffer for collecting audio
            audio_buffer = bytearray()
            silence_threshold = 500  # Adjust based on environment
            min_audio_length = self.sample_rate * 1  # 1 second minimum
            max_audio_length = self.sample_rate * 3  # 3 seconds maximum
            silence_count = 0
            max_silence = 10  # Number of silent chunks before processing
            
            while self.listening and self.enabled:
                try:
                    # Read audio chunk
                    audio_chunk = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    
                    # Calculate audio level (simple RMS)
                    import struct
                    audio_ints = struct.unpack(f'{len(audio_chunk)//2}h', audio_chunk)
                    rms = (sum(x*x for x in audio_ints) / len(audio_ints)) ** 0.5
                    
                    if rms > silence_threshold:
                        # Audio detected
                        audio_buffer.extend(audio_chunk)
                        silence_count = 0
                    else:
                        # Silence detected
                        silence_count += 1
                        if len(audio_buffer) > 0:
                            audio_buffer.extend(audio_chunk)  # Include some silence
                    
                    # Process buffer if we have enough audio and silence
                    if len(audio_buffer) > min_audio_length and (
                        silence_count >= max_silence or len(audio_buffer) > max_audio_length
                    ):
                        # Process the audio buffer
                        audio_data = bytes(audio_buffer)
                        audio_buffer.clear()
                        silence_count = 0
                        
                        # Transcribe in a separate thread to avoid blocking
                        def transcribe_and_process():
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            try:
                                transcript, confidence = loop.run_until_complete(
                                    self.transcribe_audio_chunk(audio_data)
                                )
                                if transcript and confidence > 0.5:
                                    if self._on_speech_recognized:
                                        self._on_speech_recognized(transcript)
                            except Exception as e:
                                print(f"Transcription thread error: {e}")
                            finally:
                                loop.close()
                        
                        threading.Thread(target=transcribe_and_process, daemon=True).start()
                
                except Exception as e:
                    print(f"Audio capture error: {e}")
                    time.sleep(0.1)
        
        except Exception as e:
            print(f"Listen loop error: {e}")
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_listening()
        self.audio.terminate()

def main():
    print("=== AI GESTURE GAMING WITH DEEPGRAM ===")
    
    # Get Deepgram API key
    api_key = input("Enter Deepgram API key (get free at https://console.deepgram.com/): ").strip()
    
    if not api_key:
        print("No API key provided. Get a free key at https://console.deepgram.com/")
        print("Free tier includes 45,000 minutes/month!")
        return
    
    if not DEEPGRAM_AVAILABLE:
        print("Deepgram SDK not installed. Run: pip install deepgram-sdk")
        return
    
    try:
        print("Creating Deepgram voice controller...")
        voice = DeepgramVoiceController(api_key)
        
        print("Deepgram voice controller ready!")
        print("Testing command recognition...")
        
        # Test commands
        test_commands = ["fire", "shoot", "run", "sprint", "heal", "health", "flashlight", "light"]
        for cmd in test_commands:
            result = voice.execute_voice_command(cmd)
            if result:
                print(f"TEST: '{cmd}' -> {result['name']} ({result['confidence']:.1f}%)")
        
        print("\nStarting live Deepgram recognition...")
        
        def handle_speech(transcript):
            result = voice.execute_voice_command(transcript)
            if result:
                print(f"COMMAND RECOGNIZED: {result['name']} -> {result['key']}")
            # Here you would send to input controller in the full system
        
        voice._on_speech_recognized = handle_speech
        voice.start_listening()
        
        print("Deepgram is now listening!")
        print("Try saying: 'fire', 'run', 'heal', 'flashlight', 'test'")
        print("Press Ctrl+C to stop")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
        voice.cleanup()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()