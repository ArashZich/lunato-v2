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
        logger.info("شروع تشخیص نقاط کلیدی چهره...")
        
        # بارگیری مدل تشخیص نقاط کلیدی
        landmark_detector = load_landmark_detector()
        
        if landmark_detector is None:
            logger.warning("مدل تشخیص نقاط کلیدی در دسترس نیست. استفاده از نقاط پیش‌فرض...")
            # اگر مدل در دسترس نباشد، از نقاط پیش‌فرض استفاده می‌کنیم
            return _generate_default_landmarks(face_coordinates)
        
        # استخراج مختصات چهره
        x, y, w, h = face_coordinates["x"], face_coordinates["y"], face_coordinates["width"], face_coordinates["height"]
        
        # تبدیل تصویر به سیاه و سفید برای تشخیص بهتر
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # بهبود کنتراست تصویر برای تشخیص بهتر نقاط کلیدی
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)
        except Exception as e:
            logger.warning(f"خطا در بهبود کنتراست تصویر: {str(e)}")
            enhanced_gray = gray
        
        # تشخیص نقاط کلیدی بسته به نوع مدل
        if str(type(landmark_detector)).find("dlib") != -1:
            # استفاده از dlib
            try:
                import dlib
                rect = dlib.rectangle(x, y, x + w, y + h)
                shape = landmark_detector(enhanced_gray, rect)
                
                # تبدیل به آرایه numpy
                landmarks = np.array([[p.x, p.y] for p in shape.parts()])
                
                # لاگ کردن اطلاعات برای اشکال‌زدایی
                logger.info(f"تشخیص {len(landmarks)} نقطه کلیدی با استفاده از dlib")
                
                # انتخاب نقاط کلیدی مورد نیاز برای تشخیص شکل چهره
                # اگر تعداد نقاط کافی باشد
                if len(landmarks) >= 68:  # نقاط کلیدی استاندارد dlib
                    selected_landmarks = np.array([
                        landmarks[16],   # گوشه راست پیشانی
                        landmarks[27],   # وسط پیشانی
                        landmarks[0],    # گوشه چپ پیشانی
                        landmarks[4],    # گوشه چپ فک
                        landmarks[8],    # چانه
                        landmarks[12],   # گوشه راست فک
                        landmarks[1],    # گونه چپ
                        landmarks[30],   # وسط صورت (بینی)
                        landmarks[15]    # گونه راست
                    ])
                    return selected_landmarks
                
                # اگر تعداد نقاط کمتر از انتظار باشد، همه را برگردان
                return landmarks
                
            except Exception as dlib_error:
                logger.warning(f"خطا در استفاده از dlib برای تشخیص نقاط کلیدی: {str(dlib_error)}")
                
        else:
            # استفاده از OpenCV
            try:
                faces = [np.array([x, y, x + w, y + h])]
                success, landmarks = landmark_detector.fit(enhanced_gray, faces)
                
                if not success or len(landmarks) == 0:
                    logger.warning("خطا در تشخیص نقاط کلیدی با OpenCV")
                    return _generate_default_landmarks(face_coordinates)
                    
                # استخراج نقاط
                landmarks = landmarks[0][0]
                
                # لاگ کردن اطلاعات برای اشکال‌زدایی
                logger.info(f"تشخیص {len(landmarks)} نقطه کلیدی با استفاده از OpenCV")
                
                # انتخاب نقاط کلیدی مورد نیاز برای تشخیص شکل چهره
                if len(landmarks) >= 20:  # نقاط کلیدی استاندارد OpenCV LBF
                    selected_landmarks = np.array([
                        landmarks[16],   # گوشه راست پیشانی
                        landmarks[19],   # وسط پیشانی
                        landmarks[0],    # گوشه چپ پیشانی
                        landmarks[3],    # گوشه چپ فک
                        landmarks[8],    # چانه
                        landmarks[13],   # گوشه راست فک
                        landmarks[5],    # گونه چپ
                        landmarks[10],   # وسط صورت 
                        landmarks[11]    # گونه راست
                    ])
                    return selected_landmarks
                
                # اگر تعداد نقاط کمتر از انتظار باشد، همه را برگردان
                return landmarks
                
            except Exception as cv_error:
                logger.warning(f"خطا در استفاده از OpenCV برای تشخیص نقاط کلیدی: {str(cv_error)}")
        
        # در صورت بروز خطا، از نقاط پیش‌فرض استفاده می‌کنیم
        logger.info("استفاده از نقاط کلیدی پیش‌فرض به دلیل خطا در تشخیص...")
        return _generate_default_landmarks(face_coordinates)
        
    except Exception as e:
        logger.error(f"خطا در تشخیص نقاط کلیدی چهره: {str(e)}")
        # استفاده از نقاط کلیدی پیش‌فرض در صورت بروز خطا
        return _generate_default_landmarks(face_coordinates)


def _generate_default_landmarks(face_coordinates: Dict[str, int]) -> np.ndarray:
    """
    تولید نقاط کلیدی پیش‌فرض براساس مختصات چهره.
    
    این تابع در صورت عدم موفقیت در تشخیص نقاط کلیدی با استفاده از مدل، 
    نقاط کلیدی اصلی را براساس مختصات چهره تولید می‌کند.
    
    Args:
        face_coordinates: مختصات چهره
        
    Returns:
        numpy.ndarray: نقاط کلیدی پیش‌فرض
    """
    x, y, w, h = face_coordinates["x"], face_coordinates["y"], face_coordinates["width"], face_coordinates["height"]
    
    # لاگ کردن محدوده چهره برای اشکال‌زدایی
    logger.debug(f"تولید نقاط کلیدی پیش‌فرض برای چهره در محدوده: x={x}, y={y}, w={w}, h={h}")
    
    # محاسبه نسبت ابعاد چهره برای بهبود تشخیص
    aspect_ratio = face_coordinates.get("aspect_ratio", w / h if h > 0 else 1.0)
    logger.debug(f"نسبت ابعاد چهره: {aspect_ratio:.2f}")
    
    # تنظیم نقاط کلیدی بر اساس نسبت ابعاد چهره
    # برای چهره‌های مختلف، نقاط پیش‌فرض متفاوتی تولید می‌کنیم
    
    # موقعیت‌های عمودی استاندارد
    forehead_y = y + h * 0.15    # خط پیشانی (15% از بالای چهره)
    eyes_y = y + h * 0.3         # خط چشم‌ها (30% از بالای چهره)
    middle_y = y + h * 0.5       # خط میانی صورت (50% از بالای چهره)
    jawline_y = y + h * 0.75     # خط فک (75% از بالای چهره)
    chin_y = y + h * 0.9         # خط چانه (90% از بالای چهره)
    
    # برای چهره‌های با نسبت ابعاد بالا (چهره‌های پهن‌تر)
    if aspect_ratio > 1.1:
        # تنظیم موقعیت‌های افقی برای چهره‌های پهن
        forehead_left = x + w * 0.15
        forehead_right = x + w * 0.85
        jaw_left = x + w * 0.2
        jaw_right = x + w * 0.8
        cheek_left = x + w * 0.25
        cheek_right = x + w * 0.75
        
        landmarks = np.array([
            [forehead_left, forehead_y],    # 0: گوشه بالا چپ پیشانی
            [x + w * 0.5, y + h * 0.05],    # 1: وسط پیشانی (کمی بالاتر)
            [forehead_right, forehead_y],   # 2: گوشه بالا راست پیشانی
            [jaw_left, jawline_y],          # 3: گوشه چپ فک
            [x + w * 0.5, chin_y],          # 4: چانه
            [jaw_right, jawline_y],         # 5: گوشه راست فک
            [cheek_left, middle_y],         # 6: گونه چپ
            [x + w * 0.5, middle_y],        # 7: وسط صورت
            [cheek_right, middle_y]         # 8: گونه راست
        ])
    # برای چهره‌های با نسبت ابعاد پایین (چهره‌های کشیده‌تر)
    elif aspect_ratio < 0.9:
        # تنظیم موقعیت‌های افقی برای چهره‌های کشیده
        forehead_left = x + w * 0.25
        forehead_right = x + w * 0.75
        jaw_left = x + w * 0.3
        jaw_right = x + w * 0.7
        cheek_left = x + w * 0.35
        cheek_right = x + w * 0.65
        
        landmarks = np.array([
            [forehead_left, forehead_y],    # 0: گوشه بالا چپ پیشانی
            [x + w * 0.5, y + h * 0.05],    # 1: وسط پیشانی (کمی بالاتر)
            [forehead_right, forehead_y],   # 2: گوشه بالا راست پیشانی
            [jaw_left, jawline_y],          # 3: گوشه چپ فک
            [x + w * 0.5, chin_y],          # 4: چانه
            [jaw_right, jawline_y],         # 5: گوشه راست فک
            [cheek_left, middle_y],         # 6: گونه چپ
            [x + w * 0.5, middle_y],        # 7: وسط صورت
            [cheek_right, middle_y]         # 8: گونه راست
        ])
    # برای چهره‌های با نسبت ابعاد متوسط (بیضی یا گرد)
    else:
        # تنظیم موقعیت‌های افقی برای چهره‌های متوسط
        forehead_left = x + w * 0.2
        forehead_right = x + w * 0.8
        jaw_left = x + w * 0.25
        jaw_right = x + w * 0.75
        cheek_left = x + w * 0.3
        cheek_right = x + w * 0.7
        
        landmarks = np.array([
            [forehead_left, forehead_y],    # 0: گوشه بالا چپ پیشانی
            [x + w * 0.5, y + h * 0.05],    # 1: وسط پیشانی (کمی بالاتر)
            [forehead_right, forehead_y],   # 2: گوشه بالا راست پیشانی
            [jaw_left, jawline_y],          # 3: گوشه چپ فک
            [x + w * 0.5, chin_y],          # 4: چانه
            [jaw_right, jawline_y],         # 5: گوشه راست فک
            [cheek_left, middle_y],         # 6: گونه چپ
            [x + w * 0.5, middle_y],        # 7: وسط صورت
            [cheek_right, middle_y]         # 8: گونه راست
        ])
    
    # لاگ کردن نقاط کلیدی جدید برای اشکال‌زدایی
    logger.debug(f"نقاط کلیدی پیش‌فرض تولید شده بر اساس نسبت ابعاد {aspect_ratio:.2f}")
    
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