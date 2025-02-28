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
    """تحلیل شکل چهره با استفاده از نسبت‌های هندسی دقیق‌تر"""
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
        
        # اضافه کردن لاگ برای نقاط کلیدی
        logger.info(f"نقاط کلیدی شناسایی شده: {landmarks}")
        
        # محاسبه نسبت‌های کلیدی
        if face_length > 0 and cheekbone_width > 0 and jawline_width > 0:
            width_to_length_ratio = cheekbone_width / face_length
            cheekbone_to_jaw_ratio = cheekbone_width / jawline_width
            forehead_to_cheekbone_ratio = forehead_width / cheekbone_width
            
            # لاگ کردن مقادیر واقعی محاسبه شده
            logger.info(f"مقادیر واقعی محاسبه شده:")
            logger.info(f"عرض پیشانی: {forehead_width}")
            logger.info(f"عرض گونه‌ها: {cheekbone_width}")
            logger.info(f"عرض فک: {jawline_width}")
            logger.info(f"طول صورت: {face_length}")
            logger.info(f"نسبت عرض به طول: {width_to_length_ratio}")
            logger.info(f"نسبت گونه به فک: {cheekbone_to_jaw_ratio}")
            logger.info(f"نسبت پیشانی به گونه: {forehead_to_cheekbone_ratio}")
        else:
            width_to_length_ratio = 0.8
            cheekbone_to_jaw_ratio = 1.0
            forehead_to_cheekbone_ratio = 1.0
        
        # محاسبه زاویه فک
        chin_to_jaw_left_vec = landmarks[3] - landmarks[4]
        chin_to_jaw_right_vec = landmarks[5] - landmarks[4]
        
        # محاسبه شباهت کسینوسی بین بردارها
        if np.linalg.norm(chin_to_jaw_left_vec) > 0 and np.linalg.norm(chin_to_jaw_right_vec) > 0:
            dot_product = np.dot(chin_to_jaw_left_vec, chin_to_jaw_right_vec)
            norm_left = np.linalg.norm(chin_to_jaw_left_vec)
            norm_right = np.linalg.norm(chin_to_jaw_right_vec)
            
            cos_angle = dot_product / (norm_left * norm_right)
            # اطمینان از اینکه مقدار در محدوده معتبر برای arccos قرار دارد
            cos_angle = min(1.0, max(-1.0, cos_angle))
            jaw_angle = np.arccos(cos_angle) * 180 / np.pi  # تبدیل به درجه
        else:
            jaw_angle = 150.0  # مقدار پیش‌فرض
        
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
            "cheekbone_to_jaw_ratio": 1.15,
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
        
        # ابتدا تلاش برای استفاده از روش هندسی
        geometric_result = analyze_face_shape(image, face_coordinates)
        
        # اگر تحلیل هندسی موفق نبود، تلاش برای استفاده از مدل ML
        if not geometric_result.get("success", False):
            logger.warning("تحلیل هندسی با شکست مواجه شد، تلاش برای استفاده از مدل ML")
            try:
                face_shape, confidence, shape_details = predict_face_shape(image, face_coordinates)
                ml_success = True
                logger.info(f"تشخیص شکل چهره با مدل ML: {face_shape} با میزان اطمینان {confidence:.1f}%")
            except Exception as model_error:
                logger.error(f"خطا در استفاده از مدل ML: {str(model_error)}")
                return {
                    "success": False,
                    "message": "هر دو روش تحلیل (هندسی و ML) با شکست مواجه شدند"
                }
                
            result_face_shape = face_shape
            result_confidence = confidence
        else:
            # استفاده از نتیجه تحلیل هندسی
            result_face_shape = geometric_result.get("face_shape")
            result_confidence = geometric_result.get("confidence")
            logger.info(f"استفاده از نتایج تحلیل هندسی: {result_face_shape} با اطمینان {result_confidence:.1f}%")
        
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
    تعیین شکل چهره براساس نسبت‌های هندسی با معیارهای دقیق‌تر.
    
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
    
    # یک سیستم امتیازدهی برای تشخیص بهتر استفاده می‌کنیم
    scores = {
        "OVAL": 0,
        "ROUND": 0,
        "SQUARE": 0,
        "HEART": 0,
        "OBLONG": 0,
        "DIAMOND": 0,
        "TRIANGLE": 0
    }
    
    # امتیازدهی برای صورت کشیده (OBLONG)
    if width_to_length < 0.8:
        scores["OBLONG"] += 3
    if 0.8 <= width_to_length < 0.85:
        scores["OBLONG"] += 1
    
    # امتیازدهی برای صورت گرد (ROUND)
    if width_to_length > 0.95:
        scores["ROUND"] += 2
    if 0.9 <= width_to_length <= 1.1 and jaw_angle >= 150:
        scores["ROUND"] += 2
    if abs(forehead_to_cheekbone - 1.0) < 0.2 and abs(cheekbone_to_jaw - 1.0) < 0.2:
        scores["ROUND"] += 1
        
    # امتیازدهی برای صورت مربعی (SQUARE)
    if width_to_length >= 0.85 and width_to_length <= 1.0 and jaw_angle < 145:
        scores["SQUARE"] += 2
    if abs(forehead_to_cheekbone - 1.0) < 0.2 and abs(cheekbone_to_jaw - 1.0) < 0.2:
        scores["SQUARE"] += 1
    if jaw_angle < 140:
        scores["SQUARE"] += 1
        
    # امتیازدهی برای صورت قلبی (HEART)
    if forehead_to_cheekbone > 1.0:
        scores["HEART"] += 2
    if forehead_to_cheekbone > 1.05 and cheekbone_to_jaw > 1.0:
        scores["HEART"] += 2
    
    # امتیازدهی برای صورت مثلثی (TRIANGLE)
    if forehead_to_cheekbone < 0.9 and cheekbone_to_jaw < 0.9:
        scores["TRIANGLE"] += 3
    if forehead_to_cheekbone < 0.8:
        scores["TRIANGLE"] += 1
        
    # امتیازدهی برای صورت لوزی (DIAMOND)
    # محدودتر کردن شرط‌های لوزی
    if 0.15 < forehead_to_cheekbone < 0.4 and cheekbone_to_jaw > 1.9:
        scores["DIAMOND"] += 1
    if 0.1 < forehead_to_cheekbone < 0.25 and cheekbone_to_jaw > 2.0:
        scores["DIAMOND"] += 2
    
    # امتیازدهی برای صورت بیضی (OVAL)
    if 0.8 <= width_to_length <= 0.9:
        scores["OVAL"] += 1
    if 0.85 <= width_to_length <= 0.95:
        scores["OVAL"] += 1
    if 0.9 < cheekbone_to_jaw < 1.9 and 0.4 < forehead_to_cheekbone < 0.9:
        scores["OVAL"] += 1
    if 145 <= jaw_angle <= 155:
        scores["OVAL"] += 1
        
    # اگر نسبت گونه به فک بین 1.85 و 2.0 است، امتیاز لوزی را کم کنیم
    if 1.85 <= cheekbone_to_jaw <= 2.0 and forehead_to_cheekbone > 0.3:
        scores["DIAMOND"] -= 1
        scores["OVAL"] += 1
        
    # اگر پیشانی خیلی باریک است، امتیاز لوزی را افزایش دهیم
    if forehead_to_cheekbone < 0.2:
        scores["DIAMOND"] += 1
        
    # شرایط اضافی برای لوزی و بیضی
    if forehead_to_cheekbone < 0.25 and cheekbone_to_jaw > 1.8 and cheekbone_to_jaw < 2.2:
        if jaw_angle > 145:
            scores["OVAL"] += 1
        else:
            scores["DIAMOND"] += 1
    
    # انتخاب شکل چهره با بیشترین امتیاز
    face_shape = max(scores, key=scores.get)
    
    # لاگ امتیازات برای اشکال‌زدایی
    logger.info(f"امتیازات تشخیص شکل چهره: {scores}")
    logger.info(f"تشخیص شکل چهره: {face_shape}")
    
    return face_shape