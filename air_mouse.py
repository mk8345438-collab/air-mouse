import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import os
import json

pyautogui.FAILSAFE = False

# ==========================================
# تنظیمات
# ==========================================
cap = cv2.VideoCapture(0)
FRAME_W, FRAME_H = 1280, 720
cap.set(3, FRAME_W)
cap.set(4, FRAME_H)

SCREEN_W, SCREEN_H = pyautogui.size()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# تنظیمات smoothing
smoothing = 7
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

# فایل ذخیره کالیبراسیون
CALIB_FILE = "calibration.json"

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
    
    for i, name in enumerate(corner_names):
        print(f"👉 مرحله {i+1}/4: انگشتت رو ببر به گوشه {name} و SPACE بزن...")
        
        while True:
            success, frame = cap.read()
            if not success:
                continue
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            # نمایش دستورالعمل روی تصویر
            cv2.putText(frame, f"Step {i+1}/4: Point to {name}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, "Press SPACE when ready", (50, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # رسم گوشه‌های قبلی
            for j, (cx, cy) in enumerate(corners):
                cv2.circle(frame, (cx, cy), 20, (0, 255, 0), -1)
                cv2.putText(frame, f"{j+1}", (cx-10, cy+10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                    
                    # نوک انگشت اشاره
                    tip_x = int(hand_lms.landmark[8].x * FRAME_W)
                    tip_y = int(hand_lms.landmark[8].y * FRAME_H)
                    
                    # رسم دایره روی نوک انگشت
                    cv2.circle(frame, (tip_x, tip_y), 15, (0, 0, 255), -1)
                    
                    # تشخیص کلیک (SPACE)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 32:  # SPACE
                        corners.append((tip_x, tip_y))
                        print(f"✅ گوشه {name} ثبت شد: ({tip_x}, {tip_y})")
                        break
            
            cv2.imshow("Calibration - Air Mouse", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("❌ کالیبراسیون لغو شد")
                return None
            elif key == 32 and len(corners) == i + 1:
                break
        
        time.sleep(0.5)
    
    cv2.destroyAllWindows()
    print("\n✅ کالیبراسیون کامل شد!")
    return corners

def save_calibration(corners):
    """ذخیره کالیبراسیون در فایل"""
    with open(CALIB_FILE, 'w') as f:
        json.dump(corners, f)
    print(f"💾 کالیبراسیون ذخیره شد در {CALIB_FILE}")

def load_calibration():
    """بارگذاری کالیبراسیون از فایل"""
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE, 'r') as f:
            corners = json.load(f)
        print(f"✅ کالیبراسیون بارگذاری شد از {CALIB_FILE}")
        return corners
    return None

def get_perspective_matrix(corners):
    """محاسبه ماتریس تبدیل perspective"""
    # ۴ نقطه از دوربین
    src_points = np.array(corners, dtype=np.float32)
    
    # ۴ نقطه مقصد (گوشه‌های صفحه نمایش)
    dst_points = np.array([
        [0, 0],
        [SCREEN_W, 0],
        [SCREEN_W, SCREEN_H],
        [0, SCREEN_H]
    ], dtype=np.float32)
    
    # محاسبه ماتریس
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return matrix

def transform_point(matrix, point):
    """تبدیل یک نقطه با استفاده از ماتریس"""
    point = np.array([point], dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(point, matrix)
    return int(transformed[0][0][0]), int(transformed[0][0][1])

# ==========================================
# برنامه اصلی
# ==========================================
print("\n" + "="*60)
print("🖱️ Air Mouse System - Mohammad's Project")
print("="*60)

# بررسی کالیبراسیون قبلی
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

# محاسبه ماتریس تبدیل
matrix = get_perspective_matrix(corners)

print("\n" + "="*60)
print("🚀 سیستم آماده است!")
print("="*60)
print("☝️ انگشت اشاره بالا = حرکت موس")
print("☝️🖖 انگشت اشاره + وسط = کلیک")
print("⌨️ دکمه R = کالیبراسیون مجدد")
print("⌨️ دکمه ESC = خروج")
print("="*60 + "\n")

# ==========================================
# حلقه اصلی
# ==========================================
while True:
    success, frame = cap.read()
    if not success:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    # رسم گوشه‌های کالیبراسیون روی تصویر (برای دیباگ)
    for i, (cx, cy) in enumerate(corners):
        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
        cv2.putText(frame, f"{i+1}", (cx-10, cy+10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            # نوک انگشت اشاره (Landmark 8)
            tip_x = hand_lms.landmark[8].x * FRAME_W
            tip_y = hand_lms.landmark[8].y * FRAME_H
            
            # بند دوم انگشت اشاره (Landmark 6)
            pip_x = hand_lms.landmark[6].x * FRAME_W
            pip_y = hand_lms.landmark[6].y * FRAME_H
            
            # تشخیص اینکه انگشت بالاست یا نه
            if tip_y < pip_y:
                # تبدیل مختصات با perspective transform
                screen_x, screen_y = transform_point(matrix, (tip_x, tip_y))
                
                # اعمال smoothing
                curr_x = prev_x + (screen_x - prev_x) / smoothing
                curr_y = prev_y + (screen_y - prev_y) / smoothing
                
                # محدود کردن به محدوده صفحه
                curr_x = max(0, min(SCREEN_W, curr_x))
                curr_y = max(0, min(SCREEN_H, curr_y))
                
                # حرکت موس
                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y
                
                # رسم دایره سبز روی نوک انگشت
                cv2.circle(frame, (int(tip_x), int(tip_y)), 15, (0, 255, 0), -1)
                
                # تشخیص کلیک (انگشت وسط هم بالا باشه)
                mid_tip_y = hand_lms.landmark[12].y * FRAME_H
                mid_pip_y = hand_lms.landmark[10].y * FRAME_H
                
                if mid_tip_y < mid_pip_y:
                    pyautogui.click()
                    cv2.circle(frame, (int(tip_x), int(tip_y)), 25, (0, 0, 255), -1)
                    time.sleep(0.3)  # جلوگیری از کلیک رگباری
            else:
                # انگشت پایین - دایره قرمز
                cv2.circle(frame, (int(tip_x), int(tip_y)), 15, (0, 0, 255), -1)
    
    # نمایش اطلاعات روی تصویر
    cv2.putText(frame, "Air Mouse Active", (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, "R: Recalibrate | ESC: Exit", (50, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Air Mouse - Press R to Recalibrate", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('r'):  # R - کالیبراسیون مجدد
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

# ==========================================
# پاکسازی
# ==========================================
cap.release()
cv2.destroyAllWindows()
print("\n👋 برنامه بسته شد!")
