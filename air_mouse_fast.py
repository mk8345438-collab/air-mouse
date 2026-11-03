import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import os
import json

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0  # حذف تأخیر بین حرکات

# ==========================================
# تنظیمات بهینه‌شده
# ==========================================
cap = cv2.VideoCapture(0)
FRAME_W, FRAME_H = 1280, 720
cap.set(3, FRAME_W)
cap.set(4, FRAME_H)

SCREEN_W, SCREEN_H = pyautogui.size()

# تنظیمات MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "/home/mohammed/Desktop/virtual-wall/models/hand_landmarker.task"

# ⚡ تنظیمات سرعت و دقت
smoothing = 3  # کمتر = سریع‌تر (قبلی 7 بود)
margin = 50    # کمتر = دقیق‌تر (قبلی 100 بود)

prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

CALIB_FILE = "calibration.json"

# ==========================================
# متغیرهای تشخیص Tap (کلیک لمسی)
# ==========================================
# حالت‌ها: 'idle', 'pointing_one', 'pointing_two'
gesture_state = 'idle'
last_tap_time = 0
tap_cooldown = 0.4  # حداقل فاصله بین دو tap

# ==========================================
# توابع کالیبراسیون
# ==========================================
def calibrate():
    """کالیبراسیون ۴ گوشه صفحه"""
    print("\n" + "="*60)
    print("🎯 حالت کالیبراسیون")
    print("="*60)
    print("دستورالعمل:")
    print("1. انگشت اشاره رو بالا بگیر")
    print("2. انگشتت رو ببر به گوشه‌ای که روی صفحه نمایش داده میشه")
    print("3. وقتی آماده‌ای، دکمه SPACE رو بزن")
    print("="*60 + "\n")
    
    corners = []
    corner_names = ["بالا-چپ", "بالا-راست", "پایین-راست", "پایین-چپ"]
    
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1
    )
    landmarker = HandLandmarker.create_from_options(options)
    
    for i, name in enumerate(corner_names):
        print(f"👉 مرحله {i+1}/4: انگشتت رو ببر به گوشه {name} و SPACE بزن...")
        
        while True:
            success, frame = cap.read()
            if not success:
                continue
            
            frame = cv2.flip(frame, 1)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = landmarker.detect(mp_image)
            
            cv2.putText(frame, f"Step {i+1}/4: Point to {name}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, "Press SPACE when ready", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            for j, (cx, cy) in enumerate(corners):
                cv2.circle(frame, (cx, cy), 20, (0, 255, 0), -1)
                cv2.putText(frame, f"{j+1}", (cx-10, cy+10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    for idx, landmark in enumerate(hand_landmarks):
                        x = int(landmark.x * FRAME_W)
                        y = int(landmark.y * FRAME_H)
                        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                    
                    tip_x = int(hand_landmarks[8].x * FRAME_W)
                    tip_y = int(hand_landmarks[8].y * FRAME_H)
                    cv2.circle(frame, (tip_x, tip_y), 15, (0, 0, 255), -1)
            
            cv2.imshow("Calibration - Air Mouse", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("❌ کالیبراسیون لغو شد")
                landmarker.close()
                return None
            elif key == 32 and result.hand_landmarks:
                tip_x = int(result.hand_landmarks[0][8].x * FRAME_W)
                tip_y = int(result.hand_landmarks[0][8].y * FRAME_H)
                corners.append((tip_x, tip_y))
                print(f"✅ گوشه {name} ثبت شد: ({tip_x}, {tip_y})")
                time.sleep(0.5)
                break
        
        time.sleep(0.5)
    
    landmarker.close()
    cv2.destroyAllWindows()
    print("\n✅ کالیبراسیون کامل شد!")
    return corners

def save_calibration(corners):
    with open(CALIB_FILE, 'w') as f:
        json.dump(corners, f)
    print(f"💾 کالیبراسیون ذخیره شد در {CALIB_FILE}")

def load_calibration():
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE, 'r') as f:
            corners = json.load(f)
        print(f"✅ کالیبراسیون بارگذاری شد از {CALIB_FILE}")
        return corners
    return None

def get_perspective_matrix(corners):
    src_points = np.array(corners, dtype=np.float32)
    dst_points = np.array([
        [0, 0],
        [SCREEN_W, 0],
        [SCREEN_W, SCREEN_H],
        [0, SCREEN_H]
    ], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return matrix

def transform_point(matrix, point):
    point = np.array([point], dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(point, matrix)
    return int(transformed[0][0][0]), int(transformed[0][0][1])

# ==========================================
# تابع تشخیص وضعیت انگشت‌ها
# ==========================================
def get_finger_state(hand_landmarks):
    """تشخیص وضعیت انگشت‌ها"""
    # نوک و بند دوم انگشت‌ها
    fingers = []
    
    # انگشت شست (Landmark 4 و 3)
    thumb_tip = hand_landmarks[4]
    thumb_ip = hand_landmarks[3]
    if thumb_tip.x < thumb_ip.x:  # برای دست راست (بعد از flip)
        fingers.append(1)
    else:
        fingers.append(0)
    
    # ۴ انگشت دیگه
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    
    for tip_idx, pip_idx in zip(finger_tips, finger_pips):
        tip = hand_landmarks[tip_idx]
        pip = hand_landmarks[pip_idx]
        if tip.y < pip.y:  # انگشت بالاست
            fingers.append(1)
        else:
            fingers.append(0)
    
    return fingers

# ==========================================
# برنامه اصلی
# ==========================================
print("\n" + "="*60)
print("🖱️ Air Mouse System - Fast & Precise")
print("="*60)

corners = load_calibration()

if corners is None or len(corners) != 4:
    print("⚠️ کالیبراسیون قبلی پیدا نشد یا ناقص است")
    corners = calibrate()
    if corners is None:
        print("❌ برنامه بسته شد")
        exit()
    save_calibration(corners)
else:
    print("\nآیا می‌خواهی کالیبراسیون را تغییر دهی؟ (y/n)")
    if input().lower() == 'y':
        corners = calibrate()
        if corners is None:
            print("❌ برنامه بسته شد")
            exit()
        save_calibration(corners)

matrix = get_perspective_matrix(corners)

print("\n" + "="*60)
print("🚀 سیستم آماده است!")
print("="*60)
print("☝️ یک انگشت بالا = حرکت موس")
print("☝️ یک انگشت: بالا → پایین → بالا = کلیک چپ")
print("☝️🖖 دو انگشت: بالا → پایین → بالا = کلیک راست")
print("⌨️ دکمه R = کالیبراسیون مجدد")
print("⌨️ دکمه ESC = خروج")
print("="*60 + "\n")

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)
landmarker = HandLandmarker.create_from_options(options)

# ==========================================
# حلقه اصلی
# ==========================================
frame_timestamp = 0

while True:
    success, frame = cap.read()
    if not success:
        break
    
    frame = cv2.flip(frame, 1)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    frame_timestamp += 33
    result = landmarker.detect_for_video(mp_image, frame_timestamp)
    
    # رسم گوشه‌های کالیبراسیون
    for i, (cx, cy) in enumerate(corners):
        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
        cv2.putText(frame, f"{i+1}", (cx-10, cy+10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            # رسم landmarks
            for idx, landmark in enumerate(hand_landmarks):
                x = int(landmark.x * FRAME_W)
                y = int(landmark.y * FRAME_H)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (5, 9), (9, 10), (10, 11), (11, 12),
                (9, 13), (13, 14), (14, 15), (15, 16),
                (13, 17), (17, 18), (18, 19), (19, 20),
                (0, 17)
            ]
            
            for conn in connections:
                pt1 = (int(hand_landmarks[conn[0]].x * FRAME_W), int(hand_landmarks[conn[0]].y * FRAME_H))
                pt2 = (int(hand_landmarks[conn[1]].x * FRAME_W), int(hand_landmarks[conn[1]].y * FRAME_H))
                cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
            
            # تشخیص وضعیت انگشت‌ها
            fingers = get_finger_state(hand_landmarks)
            
            # نوک انگشت اشاره
            tip_x = hand_landmarks[8].x * FRAME_W
            tip_y = hand_landmarks[8].y * FRAME_H
            
            # بررسی حالت‌ها
            # fingers: [شست، اشاره، وسط، حلقه، کوچک]
            index_up = fingers[1] == 1
            middle_up = fingers[2] == 1
            
            current_time = time.time()
            
            # حالت ۱: فقط انگشت اشاره بالا (حرکت موس)
            if index_up and not middle_up:
                screen_x, screen_y = transform_point(matrix, (tip_x, tip_y))
                
                curr_x = prev_x + (screen_x - prev_x) / smoothing
                curr_y = prev_y + (screen_y - prev_y) / smoothing
                
                curr_x = max(0, min(SCREEN_W, curr_x))
                curr_y = max(0, min(SCREEN_H, curr_y))
                
                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y
                
                cv2.circle(frame, (int(tip_x), int(tip_y)), 15, (0, 255, 0), -1)
                
                # تشخیص Tap برای کلیک چپ
                if gesture_state == 'idle':
                    gesture_state = 'pointing_one'
                elif gesture_state == 'pointing_one_down':
                    # انگشت دوباره بالا اومد → کلیک چپ
                    if current_time - last_tap_time > tap_cooldown:
                        pyautogui.click(button='left')
                        last_tap_time = current_time
                        cv2.circle(frame, (int(tip_x), int(tip_y)), 25, (0, 255, 255), -1)
                        print("🖱️ کلیک چپ!")
                    gesture_state = 'pointing_one'
            
            # حالت ۲: انگشت اشاره و وسط بالا (حرکت موس + آماده برای کلیک راست)
            elif index_up and middle_up:
                screen_x, screen_y = transform_point(matrix, (tip_x, tip_y))
                
                curr_x = prev_x + (screen_x - prev_x) / smoothing
                curr_y = prev_y + (screen_y - prev_y) / smoothing
                
                curr_x = max(0, min(SCREEN_W, curr_x))
                curr_y = max(0, min(SCREEN_H, curr_y))
                
                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y
                
                cv2.circle(frame, (int(tip_x), int(tip_y)), 15, (255, 0, 255), -1)
                
                # تشخیص Tap برای کلیک راست
                if gesture_state == 'idle':
                    gesture_state = 'pointing_two'
                elif gesture_state == 'pointing_two_down':
                    # انگشت‌ها دوباره بالا اومدن → کلیک راست
                    if current_time - last_tap_time > tap_cooldown:
                        pyautogui.click(button='right')
                        last_tap_time = current_time
                        cv2.circle(frame, (int(tip_x), int(tip_y)), 25, (255, 0, 255), -1)
                        print("🖱️ کلیک راست!")
                    gesture_state = 'pointing_two'
            
            # حالت ۳: انگشت‌ها پایین (مشت یا استراحت)
            else:
                cv2.circle(frame, (int(tip_x), int(tip_y)), 15, (0, 0, 255), -1)
                
                # ثبت حالت "پایین" برای تشخیص tap بعدی
                if gesture_state == 'pointing_one':
                    gesture_state = 'pointing_one_down'
                elif gesture_state == 'pointing_two':
                    gesture_state = 'pointing_two_down'
                else:
                    gesture_state = 'idle'
    
    # نمایش اطلاعات
    cv2.putText(frame, "Air Mouse - Fast & Precise", (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"State: {gesture_state}", (50, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "R: Recalibrate | ESC: Exit", (50, 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Air Mouse - Fast Version", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('r'):
        print("\n🔄 شروع کالیبراسیون مجدد...")
        cv2.destroyAllWindows()
        corners = calibrate()
        if corners:
            save_calibration(corners)
            matrix = get_perspective_matrix(corners)
            print("✅ کالیبراسیون مجدد کامل شد!")
        else:
            print("❌ کالیبراسیون لغو شد")
            break

landmarker.close()
cap.release()
cv2.destroyAllWindows()
print("\n👋 برنامه بسته شد!")
