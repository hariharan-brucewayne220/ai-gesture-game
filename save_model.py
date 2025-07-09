#!/usr/bin/env python3
import sys
import os
sys.path.append('src')

try:
    from gesture_detector import GestureDetector
    
    detector = GestureDetector()
    if detector.mlp_trainer and detector.mlp_trainer.model:
        result = detector.mlp_trainer.manual_save()
        if result:
            print('✅ Model saved successfully!')
        else:
            print('❌ Save failed')
    else:
        print('❌ No model found in memory')
except Exception as e:
    print(f'❌ Error: {e}')