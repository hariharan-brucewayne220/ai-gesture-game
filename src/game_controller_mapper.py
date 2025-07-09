"""
Game Controller Mapper - Uses LLM to intelligently map gestures and voice commands to game controls
"""

import json
import os
from typing import Dict, List, Tuple, Optional
import requests

class GameControllerMapper:
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        # Get the project root directory (parent of src/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        self.controls_dir = os.path.join(project_root, "custom_controller")
        
        # Reserved gesture mappings (always the same)
        self.reserved_gestures = {
            "forward": "w",
            "left": "a", 
            "backward": "s",
            "right": "d",
            "jump": "space",
            "attack": "click"
        }
    
    def load_game_controls(self, game_name: str) -> Optional[Dict]:
        """Load game controls from JSON file"""
        print(f"Searching for game controls in: {self.controls_dir}")
        
        # Store game name for camera mode detection
        self.current_game_name = game_name
        
        # Try various filename formats
        possible_names = [
            f"{game_name.lower().replace(' ', '_')}.json",
            f"{game_name.lower().replace(' ', '_')}_controls.json", 
            f"{game_name.lower()}.json",
            "god_of_war_controls.json"  # Add specific case for God of War
        ]
        
        # List all available files first
        try:
            available_files = os.listdir(self.controls_dir)
            print(f"Available control files: {available_files}")
        except FileNotFoundError:
            print(f"Controls directory not found: {self.controls_dir}")
            return None
        
        for filename in possible_names:
            filepath = os.path.join(self.controls_dir, filename)
            print(f"Trying: {filepath}")
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        print(f"Successfully loaded: {filename}")
                        return json.load(f)
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
                    continue
        
        print(f"No control file found for '{game_name}'")
        print(f"Searched for: {possible_names}")
        return None
    
    def get_existing_custom_gestures(self, mlp_trainer) -> Dict:
        """Get currently trained custom gestures (excluding reserved WASD+Jump+Attack)"""
        if not mlp_trainer or not mlp_trainer.gesture_classes:
            return {}
        
        custom_gestures = {}
        for gesture_name, class_id in mlp_trainer.gesture_classes.items():
            if gesture_name not in self.reserved_gestures:
                key_mapping = mlp_trainer.get_gesture_key_mapping(gesture_name)
                custom_gestures[gesture_name] = key_mapping
        
        return custom_gestures

    def analyze_game_controls(self, game_controls: Dict, current_gesture_count: int, mlp_trainer=None) -> Dict:
        """Use LLM to analyze game controls and recommend mappings"""
        
        # Flatten game controls for analysis - keys are already in pynput format
        all_actions = {}
        for category, actions in game_controls.items():
            if isinstance(actions, dict):
                all_actions.update(actions)
        
        # Get existing custom gestures and determine available slots
        existing_custom = self.get_existing_custom_gestures(mlp_trainer) if mlp_trainer else {}
        available_slots = max(0, 8 - current_gesture_count)
        
        prompt = f"""
🚨 CREATE NATURAL GOD OF WAR VOICE COMMANDS - NO TECHNICAL NAMES! 🚨

AVAILABLE GOD OF WAR CONTROLS:
{json.dumps(all_actions, indent=2)}

CRITICAL REQUIREMENTS:
1. Create 10-15 voice commands (NOT just 4!)
2. Use NATURAL gaming words gamers actually yell: "block", "axe", "boy", "rage"
3. NO technical names like "block_parry" or "son_action" 
4. Multiple phrases per command for better recognition
5. 1-2 words maximum per phrase

🎮 MANDATORY EXAMPLES TO FOLLOW:
- For "block_parry" → Use command name "block" with phrases ["block", "shield", "defend", "guard"]
- For "axe_recall" → Use command name "axe" with phrases ["axe", "throw", "recall", "back"]  
- For "son_action" → Use command name "boy" with phrases ["boy", "son", "atreus", "help"]
- For "spartan_rage" → Use command name "rage" with phrases ["rage", "fury", "power", "berserker"]
- For "weapons" → Use command name "weapons" with phrases ["weapons", "gear", "sword", "equipment"]
- For "map" → Use command name "map" with phrases ["map", "world", "location", "where"]

OUTPUT FORMAT - COPY THIS STRUCTURE EXACTLY:
{{
  "gesture_mappings": {{
    "block_parry": "Q",
    "axe_recall": "R"
  }},
  "voice_commands": {{
    "block": {{
      "key": "Q",
      "phrases": ["block", "shield", "defend", "guard", "parry", "protect"],
      "description": "Block and parry attacks"
    }},
    "axe": {{
      "key": "R", 
      "phrases": ["axe", "throw", "recall", "back", "return", "weapon"],
      "description": "Recall thrown axe"
    }},
    "boy": {{
      "key": "F",
      "phrases": ["boy", "son", "atreus", "help", "assist", "support"],
      "description": "Command Atreus"
    }},
    "rage": {{
      "key": "Q + Middle Mouse Button",
      "phrases": ["rage", "fury", "power", "berserker", "mode", "unleash"],
      "description": "Activate Spartan Rage"
    }},
    "dodge": {{
      "key": "Space",
      "phrases": ["dodge", "roll", "evade", "move", "avoid", "escape"],
      "description": "Dodge roll"
    }},
    "weapons": {{
      "key": "I",
      "phrases": ["weapons", "gear", "sword", "equipment", "inventory", "items"],
      "description": "Open weapons menu"
    }},
    "map": {{
      "key": "M",
      "phrases": ["map", "world", "location", "where", "navigate", "travel"],
      "description": "Open world map"
    }},
    "grab": {{
      "key": "Middle Mouse Button",
      "phrases": ["grab", "stun", "grapple", "hold", "catch", "seize"],
      "description": "Stun grab enemy"
    }},
    "heavy": {{
      "key": "Right Click",
      "phrases": ["heavy", "slam", "smash", "crush", "power", "strong"],
      "description": "Heavy attack"
    }},
    "aim": {{
      "key": "Left CTRL",
      "phrases": ["aim", "focus", "target", "precision", "steady", "lock"],
      "description": "Aim mode"
    }}
  }}
}}

🚨 CRITICAL: Use command names like "block", "axe", "boy" - NOT "block_parry", "axe_recall", "son_action"!
Generate 10+ voice commands with natural gaming language that players actually use in combat!
"""
        
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama3-8b-8192",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 1500
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Extract JSON from response
                start = content.find('{')
                end = content.rfind('}') + 1
                if start != -1 and end != -1:
                    json_str = content[start:end]
                    return json.loads(json_str)
                else:
                    print("Could not extract JSON from LLM response")
                    return self._fallback_mapping(all_actions, available_slots)
            else:
                print(f"LLM API error: {response.status_code}")
                return self._fallback_mapping(all_actions, available_slots)
                
        except Exception as e:
            print(f"LLM analysis failed: {e}")
            return self._fallback_mapping(all_actions, available_slots)
    
    def _fallback_mapping(self, all_actions: Dict, available_slots: int) -> Dict:
        """Fallback mapping if LLM fails"""
        gesture_actions = ["crouch", "reload", "melee", "interact"][:available_slots]
        
        gesture_mappings = {}
        voice_commands = {}
        
        for action, key in all_actions.items():
            if action in gesture_actions:
                gesture_mappings[action] = key.lower()
            else:
                voice_commands[action] = {
                    "key": key.lower(),
                    "phrases": [action, action.replace("_", " ")],
                    "description": f"Activates {action}"
                }
        
        return {
            "gesture_mappings": gesture_mappings,
            "voice_commands": voice_commands,
            "analysis": {
                "gesture_rationale": "Fallback: Selected most common actions",
                "voice_rationale": "Fallback: All other actions via voice",
                "strategy": "Basic fallback mapping"
            }
        }
    
    def apply_mappings(self, mappings: Dict, mlp_trainer, input_controller, voice_controller=None):
        """Apply the LLM-recommended mappings to the system"""
        print("\n🎮 Applying God of War control mappings...")
        print("IMPORTANT: Preserving all trained gesture data - only remapping keys")
        
        # Set game-specific camera mode first (God of War uses arrow keys)
        if hasattr(input_controller, 'set_game_mode'):
            game_name = getattr(self, 'current_game_name', 'god of war')
            input_controller.set_game_mode(game_name)
        
        # CLEAR ONLY VOICE COMMANDS (preserves gesture training data)
        if voice_controller and "voice_commands" in mappings:
            # Convert God of War formatted voice commands
            converted_commands = self._convert_voice_commands(mappings["voice_commands"])
            voice_controller.clear_and_add_game_commands(converted_commands)
        
        # Update gesture-to-key mappings (reuse trained hand shapes, change key outputs)
        if "gesture_mappings" in mappings:
            print("\n📋 Gesture Key Remappings for God of War:")
            for action, key_value in mappings["gesture_mappings"].items():
                # Convert God of War key format to simple key for MLP trainer
                simple_key = self._convert_key_to_simple_format(key_value)
                
                # Find which gesture class corresponds to this action
                if mlp_trainer and action in mlp_trainer.gesture_classes:
                    class_id = mlp_trainer.gesture_classes[action]
                    # Update ONLY the key mapping for this gesture (preserve training data)
                    mlp_trainer.default_gestures[action] = simple_key
                    print(f"   {action} hand shape (class {class_id}) → NOW outputs {key_value}")
                else:
                    print(f"   ⚠️ {action} not found in trained gestures - training needed")
        
        # Show analysis
        if "analysis" in mappings:
            analysis = mappings["analysis"]
            print(f"\n🧠 God of War Strategy: {analysis.get('strategy', 'N/A')}")
            print(f"📝 Gesture Logic: {analysis.get('gesture_rationale', 'N/A')}")
        
        print("\n✅ God of War mappings applied - gesture training preserved")
        return True
    
    def _convert_key_to_simple_format(self, key_value: str) -> str:
        """Convert God of War key format to simple format for MLP trainer"""
        # Handle common conversions
        conversions = {
            "Left Click": "click",
            "Right Click": "right_click", 
            "Middle Mouse Button": "middle_click",
            "Left Shift": "shift",
            "Left CTRL": "ctrl",
            "Space": "space"
        }
        
        # Check for direct conversion
        if key_value in conversions:
            return conversions[key_value]
        
        # For combo keys, just return the primary key (combos handled by voice controller)
        if "+" in key_value:
            return key_value  # Let voice controller handle combos
        
        # Single letter keys
        if len(key_value) == 1:
            return key_value.lower()
        
        return key_value.lower()
    
    def _convert_voice_commands(self, voice_commands: Dict) -> Dict:
        """Convert God of War voice commands to system format"""
        converted = {}
        
        for action_name, command_data in voice_commands.items():
            converted[action_name] = {
                "key": command_data["key"],  # Use exact key from God of War JSON
                "name": command_data.get("description", action_name),
                "intents": command_data.get("phrases", [action_name])
            }
        
        return converted

# Example usage
if __name__ == "__main__":
    # Test with dummy API key
    mapper = GameControllerMapper("test_key")
    controls = mapper.load_game_controls("last_of_us_2")
    print("Loaded controls:", json.dumps(controls, indent=2))