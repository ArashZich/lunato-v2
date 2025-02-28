import cv2
import numpy as np
import logging
import os
from typing import Dict, Any, Tuple, List, Optional
import math

from app.config import settings
from app.core.face_detection import detect_face_landmarks
from app.services.classifier import predict_face_shape
# واردسازی از فایل جدید
from app.core.face_shape_data import load_face_shape_data, get_recommended_frame_types

# تنظیمات لاگر
logger = logging.getLogger(__name__)


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
        logger.info("شروع تحلیل شکل چهره با روش هندسی...")
        # دریافت نقاط کلیدی چهره
        landmarks = detect_face_landmarks(image, face_coordinates)
        
        if landmarks is None:
            logger.error("امکان تشخیص نقاط کلیدی چهره وجود ندارد")
            return {
                "success": False,
                "message": "امکان تشخیص نقاط کلیدی چهره وجود ندارد"
            }
        
        # استخراج مقیاس‌های مهم چهره
        logger.info("محاسبه نسبت‌های هندسی چهره...")
        
        # عرض پیشانی
        forehead_width = np.linalg.norm(landmarks[0] - landmarks[2])
        logger.debug(f"عرض پیشانی: {forehead_width:.2f}")
        
        # عرض گونه‌ها
        cheekbone_width = np.linalg.norm(landmarks[6] - landmarks[8])
        logger.debug(f"عرض گونه‌ها: {cheekbone_width:.2f}")
        
        # عرض فک
        jawline_width = np.linalg.norm(landmarks[3] - landmarks[5])
        logger.debug(f"عرض فک: {jawline_width:.2f}")
        
        # طول صورت
        face_length = np.linalg.norm(landmarks[4] - landmarks[1])
        logger.debug(f"طول صورت: {face_length:.2f}")
        
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
        
        logger.info(f"نسبت‌های محاسبه شده: {shape_metrics}")
        
        # تعیین شکل چهره
        face_shape = _determine_face_shape(shape_metrics)
        confidence = _calculate_confidence(shape_metrics, face_shape)
        
        # دریافت توضیحات و توصیه‌ها
        face_shape_data = load_face_shape_data()
        face_shape_info = face_shape_data.get("face_shapes", {}).get(face_shape, {})
        
        description = face_shape_info.get("description", "")
        recommendation = face_shape_info.get("recommendation", "")
        
        logger.info(f"تحلیل شکل چهره با روش هندسی انجام شد: {face_shape} با اطمینان {confidence:.1f}%")
        
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
        logger.info("شروع تحلیل کامل شکل چهره...")
        
        # ابتدا تلاش برای استفاده از مدل یادگیری ماشین
        try:
            face_shape, confidence, shape_details = predict_face_shape(image, face_coordinates)
            logger.info(f"تشخیص شکل چهره با مدل ML: {face_shape} با میزان اطمینان {confidence:.1f}%")
            for shape, score in shape_details.items():
                logger.debug(f"امتیاز {shape}: {score:.1f}%")
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
                logger.info("تحلیل هندسی با شکست مواجه شد، استفاده از نتایج مدل ML")
                pass
            else:
                # هر دو روش شکست خوردند
                logger.error("هر دو روش تحلیل (ML و هندسی) با شکست مواجه شدند")
                return {
                    "success": False,
                    "message": geometric_result.get("message", "خطا در تحلیل شکل چهره")
                }
        
        # تصمیم‌گیری نهایی - اولویت با روش هندسی
        # این تغییر برای بهبود دقت تشخیص است
        if geometric_result.get("success", False):
            result_face_shape = geometric_result.get("face_shape")
            result_confidence = geometric_result.get("confidence")
            logger.info(f"استفاده از نتایج تحلیل هندسی: {result_face_shape} با اطمینان {result_confidence:.1f}%")
        elif ml_success:
            # استفاده از نتایج مدل ML در صورتی که تحلیل هندسی موفق نبوده است
            result_face_shape = face_shape
            result_confidence = confidence
            logger.info(f"استفاده از نتایج مدل ML: {result_face_shape} با اطمینان {result_confidence:.1f}%")
        else:
            # این حالت نباید رخ دهد (قبلاً بررسی شده است)
            return {
                "success": False,
                "message": "خطای غیرمنتظره در تحلیل شکل چهره"
            }
        
        # دریافت توضیحات و توصیه‌ها از فایل داده
        face_shape_data = load_face_shape_data()
        face_shape_info = face_shape_data.get("face_shapes", {}).get(result_face_shape, {})
        
        description = face_shape_info.get("description", "")
        recommendation = face_shape_info.get("recommendation", "")
        
        # دریافت انواع فریم پیشنهادی
        recommended_frame_types = get_recommended_frame_types(result_face_shape)
        
        logger.info(f"تحلیل کامل شکل چهره انجام شد. نتیجه نهایی: {result_face_shape}")
        
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
    
    # لاگ کردن مقادیر محاسبه شده برای اشکال‌زدایی
    logger.info(f"نسبت‌های محاسبه شده برای تشخیص شکل چهره:")
    logger.info(f"نسبت عرض به طول: {width_to_length:.2f}")
    logger.info(f"نسبت گونه به فک: {cheekbone_to_jaw:.2f}")
    logger.info(f"نسبت پیشانی به گونه: {forehead_to_cheekbone:.2f}")
    logger.info(f"زاویه فک: {jaw_angle:.2f} درجه")
    
    # تشخیص استفاده از نقاط پیش‌فرض
    is_using_default_landmarks = (abs(forehead_to_cheekbone - 1.5) < 0.01 and 
                                abs(cheekbone_to_jaw - 0.57) < 0.01)
    
    if is_using_default_landmarks:
        logger.info("استفاده از نقاط پیش‌فرض تشخیص داده شد. استفاده از الگوریتم جایگزین...")
        # در حالت استفاده از نقاط پیش‌فرض، بر اساس نسبت عرض به طول تصمیم می‌گیریم
        
        # نسبت عرض به طول برای انواع مختلف صورت
        # گرد: ~0.90+
        # بیضی: ~0.75-0.85
        # مربعی: ~0.80-0.90 با زاویه فک کمتر
        # کشیده: ~0.65 یا کمتر
        
        if width_to_length >= 0.85:
            if jaw_angle >= 150:
                logger.info("تشخیص شکل چهره: گرد (ROUND)")
                return "ROUND"
            else:
                logger.info("تشخیص شکل چهره: مربعی (SQUARE)")
                return "SQUARE"
        elif width_to_length >= 0.75:
            logger.info("تشخیص شکل چهره: بیضی (OVAL)")
            return "OVAL"
        elif width_to_length < 0.65:
            logger.info("تشخیص شکل چهره: کشیده (OBLONG)")
            return "OBLONG"
        else:
            # برای مقادیر میانی، بیضی را ترجیح می‌دهیم
            logger.info("تشخیص شکل چهره: بیضی (OVAL) - حالت پیش‌فرض")
            return "OVAL"
    
    # اگر از نقاط پیش‌فرض استفاده نمی‌شود، از منطق استاندارد استفاده می‌کنیم
    
    # صورت گرد (ROUND)
    # شرط اصلی: نسبت عرض به طول نزدیک به 1، زاویه فک بزرگ، نسبت گونه به فک نزدیک به 1
    if width_to_length >= 0.85 and width_to_length <= 1.1 and jaw_angle >= 160 and 0.9 <= cheekbone_to_jaw <= 1.1:
        logger.info("تشخیص شکل چهره: گرد (ROUND)")
        return "ROUND"
    
    # صورت مربعی (SQUARE)
    # شرط اصلی: نسبت عرض به طول نزدیک به 1، زاویه فک کوچک‌تر، نسبت گونه به فک نزدیک به 1
    if width_to_length >= 0.85 and width_to_length <= 1.1 and jaw_angle <= 150 and 0.9 <= cheekbone_to_jaw <= 1.1:
        logger.info("تشخیص شکل چهره: مربعی (SQUARE)")
        return "SQUARE"
    
    # صورت قلبی (HEART)
    # شرط اصلی: پیشانی پهن‌تر از گونه‌ها، فک باریک
    if forehead_to_cheekbone >= 1.05 and cheekbone_to_jaw >= 1.1:
        logger.info("تشخیص شکل چهره: قلبی (HEART)")
        return "HEART"
    
    # صورت مثلثی (TRIANGLE)
    # شرط اصلی: پیشانی باریک‌تر از گونه‌ها، فک پهن‌تر از گونه‌ها
    if forehead_to_cheekbone <= 0.95 and cheekbone_to_jaw <= 0.9:
        logger.info("تشخیص شکل چهره: مثلثی (TRIANGLE)")
        return "TRIANGLE"
    
    # صورت لوزی (DIAMOND)
    # شرط اصلی: گونه‌های برجسته، پیشانی و فک باریک
    if cheekbone_to_jaw >= 1.1 and forehead_to_cheekbone <= 0.95:
        logger.info("تشخیص شکل چهره: لوزی (DIAMOND)")
        return "DIAMOND"
    
    # صورت کشیده (OBLONG)
    # شرط اصلی: نسبت عرض به طول کوچک (چهره طویل)
    if width_to_length < 0.70:
        logger.info("تشخیص شکل چهره: کشیده (OBLONG)")
        return "OBLONG"
    
    # صورت بیضی (OVAL) - حالت پیش‌فرض
    # شرط اصلی: نسبت عرض به طول متوسط، نسبت‌های متعادل
    logger.info("تشخیص شکل چهره: بیضی (OVAL) - پیش‌فرض")
    return "OVAL"