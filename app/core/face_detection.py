import cv2
import numpy as np
import logging
import os
from typing import Dict, Any, Tuple, List, Optional

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
    """
    تشخیص نقاط کلیدی چهره.
    
    Args:
        image: تصویر OpenCV
        face_coordinates: مختصات چهره
        
    Returns:
        numpy.ndarray: نقاط کلیدی چهره یا None در صورت خطا
    """
    try:
        # بارگیری مدل تشخیص نقاط کلیدی
        landmark_detector = load_landmark_detector()
        
        if landmark_detector is None:
            # اگر مدل در دسترس نباشد، از نقاط پیش‌فرض استفاده می‌کنیم
            return _generate_default_landmarks(face_coordinates)
        
        # استخراج مختصات چهره
        x, y, w, h = face_coordinates["x"], face_coordinates["y"], face_coordinates["width"], face_coordinates["height"]
        
        # تبدیل تصویر به سیاه و سفید
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # تشخیص نقاط کلیدی بسته به نوع مدل
        if str(type(landmark_detector)).find("dlib") != -1:
            # استفاده از dlib
            import dlib
            rect = dlib.rectangle(x, y, x + w, y + h)
            shape = landmark_detector(gray, rect)
            
            # تبدیل به آرایه numpy
            landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        else:
            # استفاده از OpenCV
            faces = [np.array([x, y, x + w, y + h])]
            success, landmarks = landmark_detector.fit(gray, faces)
            
            if not success or len(landmarks) == 0:
                return _generate_default_landmarks(face_coordinates)
                
            # استخراج نقاط
            landmarks = landmarks[0][0]
        
        return landmarks
        
    except Exception as e:
        logger.error(f"خطا در تشخیص نقاط کلیدی چهره: {str(e)}")
        return _generate_default_landmarks(face_coordinates)


def _generate_default_landmarks(face_coordinates: Dict[str, int]) -> np.ndarray:
    """
    تولید نقاط کلیدی پیش‌فرض براساس مختصات چهره.
    
    Args:
        face_coordinates: مختصات چهره
        
    Returns:
        numpy.ndarray: نقاط کلیدی پیش‌فرض
    """
    x, y, w, h = face_coordinates["x"], face_coordinates["y"], face_coordinates["width"], face_coordinates["height"]
    
    # تعریف ۹ نقطه کلیدی پیش‌فرض برای تحلیل شکل چهره
    # 0: بالا چپ، 1: بالا وسط، 2: بالا راست
    # 3: وسط چپ، 4: وسط (چانه)، 5: وسط راست
    # 6: پایین چپ، 7: پایین وسط، 8: پایین راست
    landmarks = np.array([
        [x, y],                  # 0: گوشه بالا چپ پیشانی
        [x + w // 2, y],         # 1: وسط پیشانی
        [x + w, y],              # 2: گوشه بالا راست پیشانی
        [x, y + h // 2],         # 3: گوشه چپ فک
        [x + w // 2, y + h],     # 4: چانه
        [x + w, y + h // 2],     # 5: گوشه راست فک
        [x + w // 4, y + h // 2],# 6: گونه چپ
        [x + w // 2, y + h // 2],# 7: وسط صورت
        [x + 3 * w // 4, y + h // 2] # 8: گونه راست
    ])
    
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
        # تشخیص چهره
        detection_result = detect_face(image)
        
        if not detection_result.get("success", False):
            return False, detection_result, None
        
        # استخراج مختصات چهره
        face = detection_result.get("face")
        x, y, w, h = face["x"], face["y"], face["width"], face["height"]
        
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
        
        return True, detection_result, face_image
        
    except Exception as e:
        logger.error(f"خطا در استخراج تصویر چهره: {str(e)}")
        return False, {"success": False, "message": f"خطا در استخراج تصویر چهره: {str(e)}"}, None