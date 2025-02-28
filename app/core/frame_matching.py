import logging
import json
from typing import Dict, Any, List, Optional, Tuple
import asyncio

from app.config import settings
from app.core.face_analysis import load_face_shape_data, get_recommended_frame_types
from app.services.woocommerce import get_recommended_frames
from app.models.responses import RecommendedFrame


# تنظیمات لاگر
logger = logging.getLogger(__name__)


async def match_frames_to_face_shape(
    face_shape: str,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """
    یافتن فریم‌های مناسب برای شکل چهره.
    
    Args:
        face_shape: شکل چهره
        min_price: حداقل قیمت (اختیاری)
        max_price: حداکثر قیمت (اختیاری)
        limit: حداکثر تعداد فریم‌های پیشنهادی
        
    Returns:
        dict: نتیجه جستجوی فریم‌های مناسب
    """
    try:
        logger.info(f"تطبیق فریم‌های مناسب برای شکل چهره {face_shape}")
        
        # دریافت فریم‌های پیشنهادی از WooCommerce
        frames_result = await get_recommended_frames(
            face_shape=face_shape,
            min_price=min_price,
            max_price=max_price,
            limit=limit
        )
        
        if not frames_result.get("success", False):
            logger.warning(f"خطا در دریافت فریم‌های پیشنهادی: {frames_result.get('message')}")
            return {
                "success": False,
                "message": frames_result.get("message", "خطا در دریافت فریم‌های پیشنهادی")
            }
        
        return frames_result
        
    except Exception as e:
        logger.error(f"خطا در تطبیق فریم‌ها: {str(e)}")
        return {
            "success": False,
            "message": f"خطا در تطبیق فریم‌ها: {str(e)}"
        }


async def get_combined_result(
    face_shape: str,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """
    دریافت نتیجه ترکیبی شامل اطلاعات شکل چهره و فریم‌های پیشنهادی.
    
    Args:
        face_shape: شکل چهره
        min_price: حداقل قیمت (اختیاری)
        max_price: حداکثر قیمت (اختیاری)
        limit: حداکثر تعداد فریم‌های پیشنهادی
        
    Returns:
        dict: نتیجه ترکیبی
    """
    try:
        logger.info(f"دریافت نتیجه ترکیبی برای شکل چهره {face_shape}")
        
        # دریافت اطلاعات شکل چهره
        face_shape_data = load_face_shape_data()
        face_shape_info = face_shape_data.get("face_shapes", {}).get(face_shape, {})
        
        # دریافت انواع فریم پیشنهادی
        recommended_frame_types = get_recommended_frame_types(face_shape)
        
        # دریافت فریم‌های پیشنهادی
        frames_result = await match_frames_to_face_shape(
            face_shape=face_shape,
            min_price=min_price,
            max_price=max_price,
            limit=limit
        )
        
        # ساخت نتیجه نهایی
        result = {
            "success": frames_result.get("success", False),
            "face_shape": face_shape,
            "description": face_shape_info.get("description", ""),
            "recommendation": face_shape_info.get("recommendation", ""),
            "recommended_frame_types": recommended_frame_types
        }
        
        # اگر دریافت فریم‌ها موفقیت‌آمیز بود، افزودن فریم‌ها به نتیجه
        if frames_result.get("success", False):
            result["recommended_frames"] = frames_result.get("recommended_frames", [])
            result["total_matches"] = frames_result.get("total_matches", 0)
        else:
            # در صورت خطا، افزودن پیام خطا
            result["message"] = frames_result.get("message", "خطا در دریافت فریم‌های پیشنهادی")
        
        return result
        
    except Exception as e:
        logger.error(f"خطا در دریافت نتیجه ترکیبی: {str(e)}")
        return {
            "success": False,
            "message": f"خطا در پردازش درخواست: {str(e)}"
        }