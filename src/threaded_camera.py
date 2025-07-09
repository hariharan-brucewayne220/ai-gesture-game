"""
Threaded Camera Processor - Optimized for gaming performance
Separates camera capture from gesture processing to prevent frame drops
"""

import cv2
import threading
import time
import queue
from typing import Optional, Tuple
import os

# Optional psutil import for performance monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

class ThreadedCameraProcessor:
    def __init__(self, camera_index=0, target_fps=30, buffer_size=2):
        self.camera_index = camera_index
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.buffer_size = buffer_size
        
        # Threading components
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.capture_thread = None
        self.running = False
        
        # Performance monitoring
        self.last_fps_check = time.time()
        self.frame_count = 0
        self.current_fps = 0
        self.adaptive_mode = True
        
        # System load monitoring
        self.cpu_threshold = 80  # Switch to low-performance mode if CPU > 80%
        self.low_performance_fps = 15
        self.check_interval = 2.0  # Check system load every 2 seconds
        self.last_system_check = time.time()
        
        # Camera object
        self.cap = None
        
        print("🎮 Threaded Camera Processor initialized")
        print(f"   Target FPS: {target_fps}")
        print(f"   Adaptive mode: {self.adaptive_mode}")
        print(f"   CPU threshold: {self.cpu_threshold}%")
    
    def start(self):
        """Start the threaded camera capture"""
        if self.running:
            return False
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print("❌ Failed to open camera")
            return False
        
        # Configure camera for performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        # Set thread priority
        self._set_thread_priority()
        
        # Start capture thread
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        print("✅ Threaded camera capture started")
        return True
    
    def stop(self):
        """Stop the threaded camera capture"""
        self.running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
        
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        print("🛑 Threaded camera capture stopped")
    
    def get_frame(self) -> Optional[Tuple[bool, any]]:
        """Get the latest frame (non-blocking)"""
        try:
            # Get most recent frame, discard older ones
            frame = None
            while not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
            
            if frame is not None:
                return True, frame
            else:
                return False, None
                
        except queue.Empty:
            return False, None
    
    def _capture_loop(self):
        """Main capture loop running in separate thread"""
        print("🎥 Camera capture thread started with HIGH priority")
        
        last_frame_time = time.time()
        
        while self.running:
            current_time = time.time()
            
            # Adaptive FPS based on system load
            target_interval = self._get_adaptive_interval()
            
            # Skip if too soon for next frame
            if current_time - last_frame_time < target_interval:
                time.sleep(0.001)  # Brief sleep to prevent CPU spinning
                continue
            
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                print("⚠️ Camera read failed")
                time.sleep(0.01)
                continue
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Add frame to queue (non-blocking)
            try:
                if self.frame_queue.full():
                    # Remove oldest frame if queue is full
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.frame_queue.put_nowait(frame)
                last_frame_time = current_time
                
                # Update FPS counter
                self._update_fps_counter()
                
            except queue.Full:
                # Queue full, skip this frame
                pass
    
    def _get_adaptive_interval(self) -> float:
        """Get adaptive frame interval based on system load"""
        current_time = time.time()
        
        # Check system load periodically
        if current_time - self.last_system_check > self.check_interval:
            self.last_system_check = current_time
            
            if self.adaptive_mode and PSUTIL_AVAILABLE:
                try:
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    
                    if cpu_percent > self.cpu_threshold:
                        # High CPU usage - reduce frame rate
                        target_fps = self.low_performance_fps
                        if hasattr(self, '_adaptive_warning_shown') == False:
                            print(f"🔄 Adaptive mode: CPU {cpu_percent:.1f}% > {self.cpu_threshold}% - reducing to {target_fps}fps")
                            self._adaptive_warning_shown = True
                    else:
                        # Normal CPU usage - use target frame rate
                        target_fps = self.target_fps
                        if hasattr(self, '_adaptive_warning_shown') and self._adaptive_warning_shown:
                            print(f"✅ Adaptive mode: CPU {cpu_percent:.1f}% - restored to {target_fps}fps")
                            self._adaptive_warning_shown = False
                    
                    return 1.0 / target_fps
                    
                except:
                    # Fallback to normal interval if psutil fails
                    return self.frame_interval
        
        return self.frame_interval
    
    def _update_fps_counter(self):
        """Update FPS counter for monitoring"""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_check >= 1.0:
            self.current_fps = self.frame_count / (current_time - self.last_fps_check)
            self.frame_count = 0
            self.last_fps_check = current_time
    
    def _set_thread_priority(self):
        """Set high priority for camera thread"""
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process(os.getpid())
                process.nice(psutil.HIGH_PRIORITY_CLASS if hasattr(psutil, 'HIGH_PRIORITY_CLASS') else -10)
                print("📈 Set camera process to HIGH priority")
            except:
                print("⚠️ Could not set process priority")
        else:
            print("⚠️ psutil not available - cannot set process priority")
    
    def get_stats(self) -> dict:
        """Get performance statistics"""
        return {
            "current_fps": self.current_fps,
            "target_fps": self.target_fps,
            "queue_size": self.frame_queue.qsize(),
            "adaptive_mode": self.adaptive_mode,
            "running": self.running
        }
    
    def set_adaptive_mode(self, enabled: bool):
        """Enable/disable adaptive FPS mode"""
        self.adaptive_mode = enabled
        mode = "ENABLED" if enabled else "DISABLED"
        print(f"🔄 Adaptive FPS mode: {mode}")
    
    def set_cpu_threshold(self, threshold: int):
        """Set CPU threshold for adaptive mode"""
        self.cpu_threshold = max(50, min(95, threshold))
        print(f"📊 CPU threshold set to {self.cpu_threshold}%")