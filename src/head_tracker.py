import cv2
import mediapipe as mp
import numpy as np
import pydirectinput
import time
from typing import Optional, Tuple

class HeadTracker:
    def __init__(self, enabled=True):
        self.enabled = enabled
        if not self.enabled:
            return
            
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Configure pydirectinput for game compatibility
        pydirectinput.FAILSAFE = False  # Disable failsafe for games
        pydirectinput.PAUSE = 0  # Remove default pause between actions
        
        # Head tracking state
        self.reference_pose = None
        self.last_mouse_move = time.time()
        self.mouse_cooldown = 0.01  # 10ms between mouse moves
        
        # Sensitivity settings
        self.sensitivity_x = 8.0  # Horizontal sensitivity
        self.sensitivity_y = 6.0  # Vertical sensitivity
        self.deadzone = 5.0  # Degrees of deadzone
        
        # Visual settings
        self.show_face_mesh = False  # Set to True to show face landmarks
        
        print("🎯 Head Tracker initialized - ready for camera control!")
    
    def calculate_head_pose(self, landmarks, image_size):
        """Calculate head pose angles from face landmarks"""
        if not landmarks:
            return None, None, None
            
        # Key face landmarks for pose estimation
        face_3d = []
        face_2d = []
        
        # Specific landmark indices for pose calculation
        landmark_indices = [33, 263, 1, 61, 291, 199]  # Key face points
        
        for idx in landmark_indices:
            if idx < len(landmarks.landmark):
                lm = landmarks.landmark[idx]
                x, y = int(lm.x * image_size[1]), int(lm.y * image_size[0])
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])
        
        if len(face_2d) < 6:
            return None, None, None
            
        # Convert to numpy arrays
        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)
        
        # Camera matrix (simplified)
        focal_length = 1 * image_size[1]
        cam_matrix = np.array([[focal_length, 0, image_size[1] / 2],
                              [0, focal_length, image_size[0] / 2],
                              [0, 0, 1]])
        
        # Distortion parameters (assuming no distortion)
        dist_matrix = np.zeros((4, 1), dtype=np.float64)
        
        # Solve PnP to get rotation vector
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        
        if not success:
            return None, None, None
            
        # Convert rotation vector to rotation matrix
        rmat, jac = cv2.Rodrigues(rot_vec)
        
        # Get angles from rotation matrix
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        
        # Extract pitch, yaw, roll (in degrees)
        pitch = angles[0] * 360
        yaw = angles[1] * 360  
        roll = angles[2] * 360
        
        return pitch, yaw, roll
    
    def set_reference_pose(self, pitch, yaw, roll):
        """Set current pose as reference/center position"""
        self.reference_pose = (pitch, yaw, roll)
        print(f"📍 Head tracking reference set: P:{pitch:.1f}° Y:{yaw:.1f}° R:{roll:.1f}°")
    
    def process_head_movement(self, frame):
        """Process frame and return mouse movement if head tracking enabled"""
        if not self.enabled:
            return frame, 0, 0
            
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        mouse_dx, mouse_dy = 0, 0
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Calculate head pose
                pitch, yaw, roll = self.calculate_head_pose(face_landmarks, frame.shape)
                
                if pitch is not None and yaw is not None:
                    # Set reference pose if not set
                    if self.reference_pose is None:
                        self.set_reference_pose(pitch, yaw, roll)
                        return frame, 0, 0
                    
                    # Calculate relative movement from reference
                    ref_pitch, ref_yaw, ref_roll = self.reference_pose
                    
                    delta_pitch = pitch - ref_pitch  # Up/Down
                    delta_yaw = yaw - ref_yaw        # Left/Right
                    
                    # Apply deadzone and fix Y-axis inversion
                    if abs(delta_pitch) > self.deadzone:
                        mouse_dy = int(-delta_pitch * self.sensitivity_y)  # Negative to fix inversion
                    if abs(delta_yaw) > self.deadzone:
                        mouse_dx = int(delta_yaw * self.sensitivity_x)
                    
                    # Move mouse if movement detected and cooldown passed
                    current_time = time.time()
                    if (mouse_dx != 0 or mouse_dy != 0) and (current_time - self.last_mouse_move > self.mouse_cooldown):
                        try:
                            pydirectinput.moveRel(mouse_dx, mouse_dy)
                            self.last_mouse_move = current_time
                        except Exception as e:
                            print(f"⚠️ Mouse move error: {e}")
                    
                    # Draw face landmarks if enabled
                    if self.show_face_mesh:
                        self.mp_drawing.draw_landmarks(
                            frame, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                        )
                    
                    # Display head tracking info
                    cv2.putText(frame, f"Head: P:{delta_pitch:.1f}° Y:{delta_yaw:.1f}°", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                    
                    if abs(delta_pitch) > self.deadzone or abs(delta_yaw) > self.deadzone:
                        cv2.putText(frame, f"Mouse: ({mouse_dx}, {mouse_dy})", 
                                   (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        else:
            # No face detected
            cv2.putText(frame, "No face detected for head tracking", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
        return frame, mouse_dx, mouse_dy
    
    def reset_reference(self):
        """Reset reference pose - call this to recalibrate"""
        self.reference_pose = None
        print("🔄 Head tracking reference reset - move your head to set new center")
    
    def toggle(self):
        """Toggle head tracking on/off"""
        self.enabled = not self.enabled
        if self.enabled:
            self.reference_pose = None  # Reset reference when re-enabling
            print("✅ Head tracking ENABLED")
        else:
            print("❌ Head tracking DISABLED")
        return self.enabled
    
    def toggle_face_mesh(self):
        """Toggle face mesh visualization on/off"""
        self.show_face_mesh = not self.show_face_mesh
        status = "SHOWN" if self.show_face_mesh else "HIDDEN"
        print(f"👤 Face mesh {status}")
        return self.show_face_mesh