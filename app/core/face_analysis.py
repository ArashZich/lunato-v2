import cv2
import numpy as np
import json
import logging
import os
from typing import Dict, Any, Tuple, List, Optional
import math

from app.config import settings
from app.core.face_detection import detect_face_landmarks
from app.services.classifier import predict_face_shape

# تنظیمات لاگر
logger = logging.getLogger(__name__)

# دیکشنری اطلاعات شکل‌های چهره
_face_shape_info = None


def load_face_shape_data() -> Dict[str, Any]:
    """
    بارگیری اطلاعات مربوط به شکل‌های چهره و فریم‌های مناسب.
    
    Returns:
        dict: اطلاعات شکل‌های چهره و فریم‌های مناسب
    """
    global _face_shape_info
    
    if _face_shape_info is not None:
        return _face_shape_info
    
    try:
        # بررسی وجود فایل داده
        if not os.path.exists(settings.FACE_SHAPE_DATA_PATH):
            logger.warning(f"فایل داده‌های شکل چهره یافت نشد: {settings.FACE_SHAPE_DATA_PATH}")
            # استفاده از داده‌های پیش‌فرض
            return _get_default_face_shape_info()
        
        # خواندن فایل JSON
        with open(settings.FACE_SHAPE_DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        _face_shape_info = data
        logger.info("اطلاعات شکل چهره با موفقیت بارگیری شد")
        return _face_shape_info
        
    except Exception as e:
        logger.error(f"خطا در بارگیری اطلاعات شکل چهره: {str(e)}")
        # استفاده از داده‌های پیش‌فرض
        return _get_default_face_shape_info()


def _get_default_face_shape_info() -> Dict[str, Any]:
    """
    ایجاد داده‌های پیش‌فرض شکل چهره.
    
    Returns:
        dict: داده‌های پیش‌فرض شکل چهره
    """
    return {
        "face_shapes": {
            "OVAL": {
                "name": "بیضی",
                "description": "صورت بیضی متعادل‌ترین شکل صورت است. پهنای گونه‌ها با پهنای پیشانی و فک متناسب است.",
                "recommendation": "اکثر فریم‌ها برای این نوع صورت مناسب هستند، اما فریم‌های مستطیلی و مربعی بهترین گزینه هستند."
            },
            "ROUND": {
                "name": "گرد",
                "description": "صورت گرد دارای عرض و طول تقریباً یکسان است و فاقد زوایای مشخص است.",
                "recommendation": "فریم‌های مستطیلی و مربعی که باعث ایجاد زاویه می‌شوند، مناسب هستند."
            },
            "SQUARE": {
                "name": "مربعی",
                "description": "صورت مربعی دارای فک زاویه‌دار و پهنای پیشانی و فک تقریباً یکسان است.",
                "recommendation": "فریم‌های گرد و بیضی که خطوط صورت را نرم‌تر می‌کنند، مناسب هستند."
            },
            "HEART": {
                "name": "قلبی",
                "description": "صورت قلبی دارای پیشانی پهن و فک باریک است.",
                "recommendation": "فریم‌های گرد و بیضی که در قسمت پایین پهن‌تر هستند، مناسب هستند."
            },
            "OBLONG": {
                "name": "کشیده",
                "description": "صورت کشیده دارای طول بیشتر نسبت به عرض است.",
                "recommendation": "فریم‌های گرد و مربعی با عمق بیشتر مناسب هستند."
            },
            "DIAMOND": {
                "name": "لوزی",
                "description": "صورت لوزی دارای گونه‌های برجسته و پیشانی و فک باریک است.",
                "recommendation": "فریم‌های گربه‌ای و بیضی که خط ابرو را برجسته می‌کنند، مناسب هستند."
            },
            "TRIANGLE": {
                "name": "مثلثی",
                "description": "صورت مثلثی دارای پیشانی باریک و فک پهن است.",
                "recommendation": "فریم‌های که در قسمت بالا پررنگ‌تر هستند، مناسب هستند."
            }
        },
        "frame_types": {
            "OVAL": ["مستطیلی", "مربعی", "هشت‌ضلعی", "گربه‌ای", "بیضی"],
            "ROUND": ["مستطیلی", "مربعی", "هشت‌ضلعی", "هاوایی"],
            "SQUARE": ["گرد", "بیضی", "گربه‌ای", "هاوایی"],
            "HEART": ["گرد", "بیضی", "هاوایی", "پایین‌بدون‌فریم"],
            "OBLONG": ["مربعی", "گرد", "گربه‌ای", "هاوایی"],
            "DIAMOND": ["گربه‌ای", "هاوایی", "بیضی", "بدون‌فریم"],
            "TRIANGLE": ["گربه‌ای", "مستطیلی", "هاوایی", "بالا‌پررنگ"]
        }
    }


def analyze_face_shape(image: np.ndarray, face_coordinates: Dict[str, int]) -> Dict[str, Any]:
    """
    تحلیل شکل چهره با استفاده از نسبت‌های هندسی.
    
    این تابع از روش‌های هندسی برای تحلیل شکل چهره استفاده می‌کند.
    برای نتایج دقیق‌تر می‌توان از تابع generate_full_analysis استفاده کرد
    که از مدل یادگیری ماشین نیز استفاده می‌کند.
    
    Args:
        image: تصویر OpenCV
        face_coordinates: مختصات چهره
        
    Returns:
        dict: نتیجه تحلیل شکل چهره
    """
    try:
        # دریافت نقاط کلیدی چهره
        landmarks = detect_face_landmarks(image, face_coordinates)
        
        if landmarks is None:
            return {
                "success": False,
                "message": "امکان تشخیص نقاط کلیدی چهره وجود ندارد"
            }
        
        # استخراج مقیاس‌های مهم چهره
        # عرض پیشانی
        forehead_width = np.linalg.norm(landmarks[0] - landmarks[2])
        
        # عرض گونه‌ها
        cheekbone_width = np.linalg.norm(landmarks[6] - landmarks[8])
        
        # عرض فک
        jawline_width = np.linalg.norm(landmarks[3] - landmarks[5])
        
        # طول صورت
        face_length = np.linalg.norm(landmarks[4] - landmarks[1])
        
        # محاسبه نسبت‌های کلیدی
        width_to_length_ratio = cheekbone_width / face_length if face_length > 0 else 0
        cheekbone_to_jaw_ratio = cheekbone_width / jawline_width if jawline_width > 0 else 0
        forehead_to_cheekbone_ratio = forehead_width / cheekbone_width if cheekbone_width > 0 else 0
        
        # محاسبه زاویه فک
        chin_to_jaw_left_vec = landmarks[3] - landmarks[4]
        chin_to_jaw_right_vec = landmarks[5] - landmarks[4]
        
        # محاسبه شباهت کسینوسی بین بردارها
        cos_angle = np.dot(chin_to_jaw_left_vec, chin_to_jaw_right_vec) / (
            np.linalg.norm(chin_to_jaw_left_vec) * np.linalg.norm(chin_to_jaw_right_vec)
        )
        # اطمینان از اینکه مقدار در محدوده معتبر برای arccos قرار دارد
        cos_angle = min(1.0, max(-1.0, cos_angle))
        jaw_angle = np.arccos(cos_angle) * 180 / np.pi  # تبدیل به درجه
        
        # تعیین شکل چهره براساس نسبت‌های هندسی
        shape_metrics = {
            "width_to_length_ratio": float(width_to_length_ratio),
            "cheekbone_to_jaw_ratio": float(cheekbone_to_jaw_ratio),
            "forehead_to_cheekbone_ratio": float(forehead_to_cheekbone_ratio),
            "jaw_angle": float(jaw_angle)
        }
        
        # تعیین شکل چهره
        face_shape = _determine_face_shape(shape_metrics)
        confidence = _calculate_confidence(shape_metrics, face_shape)
        
        # دریافت توضیحات و توصیه‌ها
        face_shape_data = load_face_shape_data()
        face_shape_info = face_shape_data.get("face_shapes", {}).get(face_shape, {})
        
        description = face_shape_info.get("description", "")
        recommendation = face_shape_info.get("recommendation", "")
        
        return {
            "success": True,
            "face_shape": face_shape,
            "confidence": confidence,
            "shape_metrics": shape_metrics,
            "description": description,
            "recommendation": recommendation
        }
        
    except Exception as e:
        logger.error(f"خطا در تحلیل شکل چهره: {str(e)}")
        return {
            "success": False,
            "message": f"خطا در تحلیل شکل چهره: {str(e)}"
        }


def _calculate_confidence(metrics: Dict[str, float], face_shape: str) -> float:
    """
    محاسبه میزان اطمینان تشخیص شکل چهره.
    
    Args:
        metrics: نسبت‌های هندسی
        face_shape: شکل چهره تشخیص داده شده
        
    Returns:
        float: میزان اطمینان (0-100)
    """
    # تعریف مقادیر ایده‌آل برای هر شکل چهره
    ideal_metrics = {
        "OVAL": {
            "width_to_length_ratio": 0.8,
            "cheekbone_to_jaw_ratio": 1.0,
            "forehead_to_cheekbone_ratio": 1.0,
            "jaw_angle": 160
        },
        "ROUND": {
            "width_to_length_ratio": 0.9,
            "cheekbone_to_jaw_ratio": 1.05,
            "forehead_to_cheekbone_ratio": 1.0,
            "jaw_angle": 170
        },
        "SQUARE": {
            "width_to_length_ratio": 0.9,
            "cheekbone_to_jaw_ratio": 1.0,
            "forehead_to_cheekbone_ratio": 1.0,
            "jaw_angle": 140
        },
        "HEART": {
            "width_to_length_ratio": 0.8,
            "cheekbone_to_jaw_ratio": 1.0,
            "forehead_to_cheekbone_ratio": 1.15,
            "jaw_angle": 160
        },
        "OBLONG": {
            "width_to_length_ratio": 0.7,
            "cheekbone_to_jaw_ratio": 1.0,
            "forehead_to_cheekbone_ratio": 1.0,
            "jaw_angle": 160
        },
        "DIAMOND": {
            "width_to_length_ratio": 0.8,
            "cheekbone_to_jaw_ratio": 1.2,
            "forehead_to_cheekbone_ratio": 0.9,
            "jaw_angle": 160
        },
        "TRIANGLE": {
            "width_to_length_ratio": 0.8,
            "cheekbone_to_jaw_ratio": 0.8,
            "forehead_to_cheekbone_ratio": 0.9,
            "jaw_angle": 150
        }
    }
    
    # دریافت مقادیر ایده‌آل برای شکل چهره تشخیص داده شده
    ideal = ideal_metrics.get(face_shape, ideal_metrics["OVAL"])
    
    # محاسبه میزان انحراف از مقادیر ایده‌آل
    deviations = [
        1 - min(abs(metrics["width_to_length_ratio"] - ideal["width_to_length_ratio"]) / ideal["width_to_length_ratio"], 1),
        1 - min(abs(metrics["cheekbone_to_jaw_ratio"] - ideal["cheekbone_to_jaw_ratio"]) / ideal["cheekbone_to_jaw_ratio"], 1),
        1 - min(abs(metrics["forehead_to_cheekbone_ratio"] - ideal["forehead_to_cheekbone_ratio"]) / ideal["forehead_to_cheekbone_ratio"], 1),
        1 - min(abs(metrics["jaw_angle"] - ideal["jaw_angle"]) / ideal["jaw_angle"], 1)
    ]
    
    # وزن‌دهی به انحرافات
    weights = [0.3, 0.3, 0.2, 0.2]
    weighted_avg = sum(d * w for d, w in zip(deviations, weights))
    
    # تبدیل به درصد
    confidence = weighted_avg * 100
    
    # محدود کردن به بازه 60-95
    confidence = max(60, min(95, confidence))
    
    return round(confidence, 1)


def generate_full_analysis(image: np.ndarray, face_coordinates: Dict[str, int]) -> Dict[str, Any]:
    """
    تحلیل کامل شکل چهره با استفاده از ترکیب روش‌های هندسی و یادگیری ماشین.
    
    Args:
        image: تصویر OpenCV
        face_coordinates: مختصات چهره
        
    Returns:
        dict: نتیجه تحلیل شکل چهره
    """
    try:
        # ابتدا تلاش برای استفاده از مدل یادگیری ماشین
        try:
            face_shape, confidence, shape_details = predict_face_shape(image, face_coordinates)
            logger.info(f"تشخیص شکل چهره با مدل ML: {face_shape} با میزان اطمینان {confidence:.1f}%")
            ml_success = True
        except Exception as model_error:
            logger.warning(f"خطا در استفاده از مدل ML: {str(model_error)}. استفاده از روش هندسی...")
            ml_success = False
        
        # تحلیل هندسی به عنوان پشتیبان یا مکمل
        geometric_result = analyze_face_shape(image, face_coordinates)
        
        if not geometric_result.get("success", False):
            # اگر تحلیل هندسی هم شکست خورد
            if ml_success:
                # اگر مدل ML موفق بوده، از نتایج آن استفاده می‌کنیم
                pass
            else:
                # هر دو روش شکست خوردند
                return {
                    "success": False,
                    "message": geometric_result.get("message", "خطا در تحلیل شکل چهره")
                }
        
        # تصمیم‌گیری نهایی
        if ml_success:
            # اگر مدل ML استفاده شده است، از نتایج آن استفاده می‌کنیم
            # اما اطلاعات توصیفی را از تحلیل هندسی می‌گیریم
            result_face_shape = face_shape
            result_confidence = confidence
        else:
            # استفاده از نتایج تحلیل هندسی
            result_face_shape = geometric_result.get("face_shape")
            result_confidence = geometric_result.get("confidence")
        
        # دریافت توضیحات و توصیه‌ها از فایل داده
        face_shape_data = load_face_shape_data()
        face_shape_info = face_shape_data.get("face_shapes", {}).get(result_face_shape, {})
        
        description = face_shape_info.get("description", "")
        recommendation = face_shape_info.get("recommendation", "")
        
        # دریافت انواع فریم پیشنهادی
        recommended_frame_types = get_recommended_frame_types(result_face_shape)
        
        return {
            "success": True,
            "face_shape": result_face_shape,
            "confidence": result_confidence,
            "description": description,
            "recommendation": recommendation,
            "recommended_frame_types": recommended_frame_types
        }
        
    except Exception as e:
        logger.error(f"خطا در تحلیل کامل شکل چهره: {str(e)}")
        return {
            "success": False,
            "message": f"خطا در تحلیل شکل چهره: {str(e)}"
        }


def get_recommended_frame_types(face_shape: str) -> List[str]:
    """
    دریافت انواع فریم پیشنهادی براساس شکل چهره.
    
    Args:
        face_shape: شکل چهره
        
    Returns:
        list: انواع فریم پیشنهادی
    """
    # بارگیری اطلاعات شکل چهره
    face_shape_data = load_face_shape_data()
    
    # دریافت فریم‌های پیشنهادی
    frame_types = face_shape_data.get("frame_types", {}).get(face_shape, [])
    
    # اگر هیچ پیشنهادی نبود، یک لیست پیش‌فرض برگردان
    if not frame_types:
        logger.warning(f"هیچ نوع فریمی برای شکل چهره {face_shape} پیدا نشد. استفاده از مقادیر پیش‌فرض.")
        frame_types = ["مستطیلی", "گرد", "گربه‌ای", "هاوایی"]
    
    return frame_types


def _determine_face_shape(metrics: Dict[str, float]) -> str:
    """
    تعیین شکل چهره براساس نسبت‌های هندسی.
    
    Args:
        metrics: مقادیر محاسبه شده نسبت‌های هندسی
        
    Returns:
        str: شکل چهره تشخیص داده شده
    """
    width_to_length = metrics["width_to_length_ratio"]
    cheekbone_to_jaw = metrics["cheekbone_to_jaw_ratio"]
    forehead_to_cheekbone = metrics["forehead_to_cheekbone_ratio"]
    jaw_angle = metrics["jaw_angle"]
    
    # تعیین شکل چهره براساس قوانین هندسی
    # مقادیر آستانه براساس تحقیقات و بررسی‌های انجام شده تعیین شده‌اند
    
    # صورت بیضی (OVAL)
    # نسبت عرض به طول متوسط، نسبت گونه به فک نزدیک به 1
    if 0.75 <= width_to_length <= 0.85 and 0.9 <= cheekbone_to_jaw <= 1.1:
        return "OVAL"
    
    # صورت گرد (ROUND)
    # نسبت عرض به طول بالا، زاویه فک بزرگ
    if width_to_length > 0.85 and jaw_angle > 150:
        return "ROUND"
    
    # صورت مربعی (SQUARE)
    # نسبت عرض به طول بالا، نسبت گونه به فک نزدیک به 1، زاویه فک کوچک
    if width_to_length > 0.85 and 0.9 <= cheekbone_to_jaw <= 1.1 and jaw_angle < 150:
        return "SQUARE"
    
    # صورت قلبی (HEART)
    # نسبت پیشانی به گونه بزرگتر از 1
    if forehead_to_cheekbone > 1.05:
        return "HEART"
    
    # صورت کشیده (OBLONG)
    # نسبت عرض به طول پایین
    if width_to_length < 0.75:
        return "OBLONG"
    
    # صورت لوزی (DIAMOND)
    # نسبت گونه به فک بالا
    if cheekbone_to_jaw > 1.1:
        return "DIAMOND"
    
    # صورت مثلثی (TRIANGLE)
    # نسبت گونه به فک پایین، نسبت پیشانی به گونه پایین
    if cheekbone_to_jaw < 0.9 and forehead_to_cheekbone < 0.95:
        return "TRIANGLE"
    
    # اگر هیچ کدام از شرایط بالا برقرار نبود، پیش‌فرض بیضی
    return "OVAL"