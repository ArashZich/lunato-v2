import cv2
import numpy as np
import logging
import os
from typing import Dict, Any, Tuple, List, Optional
import mediapipe as mp

from app.config import settings

# تنظیمات لاگر
logger = logging.getLogger(__name__)

# متغیرهای سراسری برای کش مدل‌ها
_face_cascade = None
_landmark_detector = None


def load_face_detector():
    """
    بارگیری مدل تشخیص چهره.
    
    Returns:
        cv2.CascadeClassifier: مدل تشخیص چهره
    """
    global _face_cascade
    
    if _face_cascade is not None:
        return _face_cascade
    
    try:
        # مسیر فایل مدل haarcascade
        cascade_path = cv2.data.haarcascades + settings.FACE_DETECTION_MODEL
        
        # بررسی وجود فایل
        if not os.path.exists(cascade_path):
            logger.error(f"فایل مدل تشخیص چهره یافت نشد: {cascade_path}")
            raise FileNotFoundError(f"فایل مدل تشخیص چهره یافت نشد: {cascade_path}")
        
        # بارگیری مدل
        _face_cascade = cv2.CascadeClassifier(cascade_path)
        
        logger.info("مدل تشخیص چهره با موفقیت بارگیری شد")
        return _face_cascade
        
    except Exception as e:
        logger.error(f"خطا در بارگیری مدل تشخیص چهره: {str(e)}")
        raise


def load_landmark_detector():
    """
    بارگیری مدل تشخیص نقاط کلیدی چهره.
    
    Returns:
        Object: مدل تشخیص نقاط کلیدی چهره
    """
    global _landmark_detector
    
    if _landmark_detector is not None:
        return _landmark_detector
    
    try:
        # استفاده از مدل تشخیص نقاط کلیدی OpenCV's DNN Face Detector
        # یا در صورت امکان از dlib's facial landmark detector
        try:
            import dlib
            _landmark_detector = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")
            logger.info("مدل تشخیص نقاط کلیدی dlib با موفقیت بارگیری شد")
        except (ImportError, RuntimeError):
            logger.warning("امکان استفاده از مدل dlib وجود ندارد. استفاده از روش جایگزین...")
            # استفاده از OpenCV DNN
            _landmark_detector = cv2.face.createFacemarkLBF()
            _landmark_detector.loadModel("data/lbfmodel.yaml")
            logger.info("مدل تشخیص نقاط کلیدی OpenCV با موفقیت بارگیری شد")
        
        return _landmark_detector
        
    except Exception as e:
        logger.error(f"خطا در بارگیری مدل تشخیص نقاط کلیدی: {str(e)}")
        # برگرداندن None به جای raise، چون ممکن است نیاز به نقاط کلیدی نباشد
        return None


def detect_face(image: np.ndarray) -> Dict[str, Any]:
    """
    تشخیص چهره در تصویر.
    
    Args:
        image: تصویر OpenCV
        
    Returns:
        dict: نتیجه تشخیص چهره
    """
    try:
        # بارگیری مدل تشخیص چهره
        face_cascade = load_face_detector()
        
        # تبدیل تصویر به سیاه و سفید برای تشخیص بهتر
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # تشخیص چهره
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # اگر چهره‌ای تشخیص داده نشد
        if len(faces) == 0:
            return {
                "success": False,
                "message": "هیچ چهره‌ای در تصویر تشخیص داده نشد"
            }
        
        # اگر بیش از یک چهره تشخیص داده شد
        if len(faces) > 1:
            # انتخاب بزرگترین چهره (احتمالاً نزدیک‌ترین چهره)
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            logger.info(f"چندین چهره تشخیص داده شد ({len(faces)}). بزرگترین چهره انتخاب شد.")
        else:
            largest_face = faces[0]
        
        # استخراج مختصات چهره
        x, y, w, h = largest_face
        
        # محاسبه مرکز چهره
        center_x = x + w // 2
        center_y = y + h // 2
        
        # محاسبه نسبت عرض به ارتفاع
        aspect_ratio = w / h if h > 0 else 0
        
        # ساخت دیکشنری مختصات چهره
        face_coordinates = {
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h),
            "center_x": int(center_x),
            "center_y": int(center_y),
            "aspect_ratio": float(aspect_ratio)
        }
        
        return {
            "success": True,
            "message": "چهره با موفقیت تشخیص داده شد",
            "face": face_coordinates
        }
        
    except Exception as e:
        logger.error(f"خطا در تشخیص چهره: {str(e)}")
        return {
            "success": False,
            "message": f"خطا در تشخیص چهره: {str(e)}"
        }


def detect_face_landmarks(image: np.ndarray, face_coordinates: Dict[str, int]) -> Optional[np.ndarray]:
    """تشخیص نقاط کلیدی چهره"""
    try:
        logger.info("شروع تشخیص نقاط کلیدی چهره...")
        
        # استخراج مختصات چهره
        x, y, w, h = face_coordinates["x"], face_coordinates["y"], face_coordinates["width"], face_coordinates["height"]
        
        try:
            import mediapipe as mp
            mp_face_mesh = mp.solutions.face_mesh
            
            # برش چهره با حاشیه
            padding = int(w * 0.1)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            face_img = image[y1:y2, x1:x2]
            
            # تبدیل به RGB
            rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_mesh:
                
                results = face_mesh.process(rgb_face)
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    
                    # تبدیل مختصات نسبی به مختصات مطلق
                    face_height, face_width = face_img.shape[:2]
                    
                    # انتخاب نقاط کلیدی متناسب با مدل mediapipe
                    # نقاط اصلی برای تعیین شکل صورت
                    selected_landmarks = np.array([
                        # پیشانی
                        [(landmarks[10].x * face_width + x1), (landmarks[10].y * face_height + y1)],  # پیشانی چپ
                        [(landmarks[8].x * face_width + x1), (landmarks[8].y * face_height + y1 - h*0.1)],  # وسط پیشانی (بالاتر)
                        [(landmarks[338].x * face_width + x1), (landmarks[338].y * face_height + y1)],  # پیشانی راست
                        
                        # فک
                        [(landmarks[149].x * face_width + x1), (landmarks[149].y * face_height + y1)],  # فک چپ
                        [(landmarks[152].x * face_width + x1), (landmarks[152].y * face_height + y1)],  # چانه
                        [(landmarks[378].x * face_width + x1), (landmarks[378].y * face_height + y1)],  # فک راست
                        
                        # گونه‌ها
                        [(landmarks[116].x * face_width + x1), (landmarks[116].y * face_height + y1)],  # گونه چپ
                        [(landmarks[168].x * face_width + x1), (landmarks[168].y * face_height + y1)],  # بینی (وسط)
                        [(landmarks[345].x * face_width + x1), (landmarks[345].y * face_height + y1)]   # گونه راست
                    ])
                    
                    logger.info(f"تشخیص {len(selected_landmarks)} نقطه کلیدی با استفاده از mediapipe")
                    return selected_landmarks
                else:
                    logger.warning("نقاط کلیدی با mediapipe تشخیص داده نشد")
        except Exception as e:
            logger.warning(f"خطا در تشخیص با mediapipe: {str(e)}")
            
        # در صورت شکست، از تابع نقاط پویا استفاده کنیم
        return _generate_dynamic_landmarks(image, face_coordinates)
    except Exception as e:
        logger.error(f"خطا در تشخیص نقاط کلیدی چهره: {str(e)}")
        return _generate_dynamic_landmarks(image, face_coordinates)
    

def _generate_dynamic_landmarks(image: np.ndarray, face_coordinates: Dict[str, int]) -> np.ndarray:
    """تولید نقاط کلیدی پویا"""
    x, y, w, h = face_coordinates["x"], face_coordinates["y"], face_coordinates["width"], face_coordinates["height"]
    
    # موقعیت‌های عمودی
    forehead_y = y + h * 0.15
    middle_y = y + h * 0.5
    jawline_y = y + h * 0.75
    chin_y = y + h * 0.9
    
    # تصویر چهره را برش بزنیم
    face_img = image[y:y+h, x:x+w]
    
    # تبدیل به سیاه و سفید
    if len(face_img.shape) == 3:
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_face = face_img
    
    # میانگین روشنایی قسمت‌های مختلف صورت
    height, width = gray_face.shape
    top_third = gray_face[:int(height/3), :]
    middle_third = gray_face[int(height/3):int(2*height/3), :]
    bottom_third = gray_face[int(2*height/3):, :]
    
    top_brightness = np.mean(top_third)
    middle_brightness = np.mean(middle_third)
    bottom_brightness = np.mean(bottom_third)
    
    # نسبت روشنایی به کار می‌رود برای تخمین شکل
    top_ratio = top_brightness / middle_brightness
    bottom_ratio = bottom_brightness / middle_brightness
    
    # تنظیم پهنای بخش‌های مختلف صورت براساس روشنایی
    aspect_ratio = w / h if h > 0 else 1.0
    
    # تنظیم پویای مقادیر
    forehead_width_factor = 0.7 + (top_ratio - 1) * 0.1
    forehead_width_factor = max(0.5, min(0.9, forehead_width_factor))
    
    cheekbone_width_factor = 0.75
    
    jaw_width_factor = 0.7 + (bottom_ratio - 1) * 0.1
    jaw_width_factor = max(0.55, min(0.85, jaw_width_factor))
    
    # محاسبه پهنای بخش‌های مختلف چهره
    forehead_width = w * forehead_width_factor
    cheekbone_width = w * cheekbone_width_factor
    jaw_width = w * jaw_width_factor
    
    # موقعیت افقی
    forehead_left = x + (w - forehead_width) / 2
    forehead_right = forehead_left + forehead_width
    
    cheek_left = x + (w - cheekbone_width) / 2
    cheek_right = cheek_left + cheekbone_width
    
    jaw_left = x + (w - jaw_width) / 2
    jaw_right = jaw_left + jaw_width
    
    # ساخت نقاط کلیدی
    landmarks = np.array([
        [forehead_left, forehead_y],    # 0: گوشه بالا چپ پیشانی
        [x + w/2, y + h * 0.05],        # 1: وسط پیشانی
        [forehead_right, forehead_y],   # 2: گوشه بالا راست پیشانی
        [jaw_left, jawline_y],          # 3: گوشه چپ فک
        [x + w/2, chin_y],              # 4: چانه
        [jaw_right, jawline_y],         # 5: گوشه راست فک
        [cheek_left, middle_y],         # 6: گونه چپ
        [x + w/2, middle_y],            # 7: وسط صورت
        [cheek_right, middle_y]         # 8: گونه راست
    ])
    
    # لاگ مقادیر محاسبه شده برای اشکال‌زدایی
    forehead_to_cheekbone = forehead_width / cheekbone_width
    cheekbone_to_jaw = cheekbone_width / jaw_width
    
    logger.debug(f"نقاط کلیدی پویا: نسبت پیشانی به گونه: {forehead_to_cheekbone:.2f}")
    logger.debug(f"نقاط کلیدی پویا: نسبت گونه به فک: {cheekbone_to_jaw:.2f}")
    
    return landmarks




def get_face_image(image: np.ndarray) -> Tuple[bool, Dict[str, Any], Optional[np.ndarray]]:
    """
    تشخیص چهره و برش تصویر چهره.
    
    Args:
        image: تصویر OpenCV
        
    Returns:
        tuple: (موفقیت، نتیجه، تصویر_برش_خورده)
    """
    try:
        logger.info("شروع تشخیص چهره در تصویر...")

        # بررسی کیفیت و وضوح تصویر
        height, width = image.shape[:2]
        logger.info(f"ابعاد تصویر: {width}x{height}")
        
        if width < 100 or height < 100:
            logger.warning(f"تصویر با ابعاد {width}x{height} کیفیت مناسبی برای تشخیص چهره ندارد")
            return False, {
                "success": False, 
                "message": "تصویر با کیفیت مناسبی برای تشخیص چهره ندارد (ابعاد خیلی کوچک)"
            }, None
            
        # تشخیص چهره
        detection_result = detect_face(image)
        
        if not detection_result.get("success", False):
            logger.warning(f"تشخیص چهره ناموفق بود: {detection_result.get('message', 'دلیل نامشخص')}")
            return False, detection_result, None
        
        # استخراج مختصات چهره
        face = detection_result.get("face")
        x, y, w, h = face["x"], face["y"], face["width"], face["height"]
        
        # بررسی نسبت ابعاد چهره برای اطمینان از تشخیص درست
        aspect_ratio = w / h if h > 0 else 0
        logger.info(f"نسبت ابعاد چهره تشخیص داده شده: {aspect_ratio:.2f}")
        
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            logger.warning(f"نسبت ابعاد چهره ({aspect_ratio:.2f}) خارج از محدوده منطقی است")
            return False, {
                "success": False, 
                "message": f"چهره تشخیص داده شده نسبت ابعاد غیرطبیعی دارد ({aspect_ratio:.2f})"
            }, None
        
        # بررسی اندازه چهره در مقایسه با کل تصویر
        face_area = w * h
        image_area = width * height
        face_percentage = (face_area / image_area) * 100
        logger.info(f"درصد اشغال تصویر توسط چهره: {face_percentage:.2f}%")
        
        if face_percentage < 5:
            logger.warning(f"چهره تشخیص داده شده خیلی کوچک است (فقط {face_percentage:.2f}% از تصویر)")
            return False, {
                "success": False, 
                "message": f"چهره تشخیص داده شده خیلی کوچک است"
            }, None
        
        # اضافه کردن مقداری حاشیه اطراف چهره (20%)
        padding_x = int(w * 0.2)
        padding_y = int(h * 0.2)
        
        # مختصات جدید با حاشیه
        x1 = max(0, x - padding_x)
        y1 = max(0, y - padding_y)
        x2 = min(image.shape[1], x + w + padding_x)
        y2 = min(image.shape[0], y + h + padding_y)
        
        # برش تصویر چهره
        face_image = image[y1:y2, x1:x2]
        
        # به‌روزرسانی مختصات چهره با حاشیه
        detection_result["face"]["x"] = x1
        detection_result["face"]["y"] = y1
        detection_result["face"]["width"] = x2 - x1
        detection_result["face"]["height"] = y2 - y1
        
        logger.info(f"چهره با موفقیت تشخیص داده شد. ابعاد تصویر برش خورده: {x2-x1}x{y2-y1}")
        
        return True, detection_result, face_image
        
    except Exception as e:
        logger.error(f"خطا در استخراج تصویر چهره: {str(e)}")
        return False, {"success": False, "message": f"خطا در استخراج تصویر چهره: {str(e)}"}, None