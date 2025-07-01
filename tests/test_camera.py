"""
Quick camera test to verify webcam is working
Run this first to make sure your camera is accessible
"""
import cv2

def test_camera():
    print("🔍 Testing camera access...")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return False
    
    print("✅ Camera opened successfully!")
    print("📹 Press 'q' to quit camera test")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Error: Could not read frame")
            break
            
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Add test text
        cv2.putText(frame, "Camera Test - Press 'q' to quit", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Camera Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Camera test completed!")
    return True

if __name__ == "__main__":
    test_camera()