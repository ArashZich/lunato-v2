import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from motor.motor_asyncio import AsyncIOMotorDatabase
from bson.objectid import ObjectId
import uuid

from app.db.connection import get_database
from app.config import settings

# تنظیمات لاگر
logger = logging.getLogger(__name__)


async def save_request_info(
    path: str,
    method: str,
    client_info: Dict[str, Any],
    status_code: int,
    process_time: float,
    request_id: str
) -> bool:
    """
    ذخیره اطلاعات درخواست در دیتابیس.
    
    Args:
        path: مسیر درخواست
        method: متد HTTP
        client_info: اطلاعات کاربر
        status_code: کد وضعیت پاسخ
        process_time: زمان پردازش بر حسب ثانیه
        request_id: شناسه منحصر به فرد درخواست
        
    Returns:
        bool: نتیجه عملیات ذخیره‌سازی
    """
    # بررسی وضعیت ذخیره‌سازی اطلاعات تحلیلی
    if not settings.STORE_ANALYTICS:
        return False
    
    try:
        db = get_database()
        
        # ساخت داده درخواست
        request_data = {
            "request_id": request_id,
            "path": path,
            "method": method,
            "client_info": client_info,
            "status_code": status_code,
            "process_time": process_time,
            "created_at": datetime.utcnow()
        }
        
        # ذخیره در کالکشن درخواست‌ها
        await db.requests.insert_one(request_data)
        
        return True
        
    except Exception as e:
        logger.error(f"خطا در ذخیره اطلاعات درخواست: {e}")
        return False


async def save_analysis_result(
    user_id: str,
    request_id: str,
    face_shape: str,
    confidence: float,
    client_info: Dict[str, Any],
    task_id: Optional[str] = None
) -> str:
    """
    ذخیره نتیجه تحلیل چهره در دیتابیس.
    
    Args:
        user_id: شناسه کاربر
        request_id: شناسه درخواست
        face_shape: شکل چهره
        confidence: میزان اطمینان
        client_info: اطلاعات کاربر
        task_id: شناسه وظیفه Celery (اختیاری)
        
    Returns:
        str: شناسه رکورد ایجاد شده
    """
    # بررسی وضعیت ذخیره‌سازی اطلاعات تحلیلی
    if not settings.STORE_ANALYTICS:
        return str(uuid.uuid4())
    
    try:
        db = get_database()
        
        # ساخت داده تحلیل
        analysis_data = {
            "user_id": user_id,
            "request_id": request_id,
            "face_shape": face_shape,
            "confidence": confidence,
            "client_info": client_info,
            "task_id": task_id,
            "created_at": datetime.utcnow()
        }
        
        # ذخیره در کالکشن نتایج تحلیل
        result = await db.analysis_results.insert_one(analysis_data)
        
        return str(result.inserted_id)
        
    except Exception as e:
        logger.error(f"خطا در ذخیره نتیجه تحلیل: {e}")
        return str(uuid.uuid4())


async def save_recommendation(
    user_id: str,
    face_shape: str,
    recommended_frame_types: List[str],
    recommended_frames: List[Dict[str, Any]],
    client_info: Dict[str, Any],
    analysis_id: Optional[str] = None
) -> str:
    """
    ذخیره پیشنهادات فریم در دیتابیس.
    
    Args:
        user_id: شناسه کاربر
        face_shape: شکل چهره
        recommended_frame_types: انواع فریم پیشنهادی
        recommended_frames: فریم‌های پیشنهادی
        client_info: اطلاعات کاربر
        analysis_id: شناسه تحلیل (اختیاری)
        
    Returns:
        str: شناسه رکورد ایجاد شده
    """
    # بررسی وضعیت ذخیره‌سازی اطلاعات تحلیلی
    if not settings.STORE_ANALYTICS:
        return str(uuid.uuid4())
    
    try:
        db = get_database()
        
        # ساخت داده پیشنهاد
        recommendation_data = {
            "user_id": user_id,
            "face_shape": face_shape,
            "recommended_frame_types": recommended_frame_types,
            "recommended_frames": [
                {
                    "id": frame.get("id"),
                    "name": frame.get("name"),
                    "frame_type": frame.get("frame_type"),
                    "match_score": frame.get("match_score")
                }
                for frame in recommended_frames
            ],
            "client_info": client_info,
            "analysis_id": analysis_id,
            "created_at": datetime.utcnow()
        }
        
        # ذخیره در کالکشن پیشنهادات
        result = await db.recommendations.insert_one(recommendation_data)
        
        return str(result.inserted_id)
        
    except Exception as e:
        logger.error(f"خطا در ذخیره پیشنهادات: {e}")
        return str(uuid.uuid4())


async def get_analytics_summary() -> Dict[str, Any]:
    """
    دریافت خلاصه اطلاعات تحلیلی.
    
    Returns:
        dict: خلاصه اطلاعات تحلیلی
    """
    try:
        db = get_database()
        
        # تعداد کل درخواست‌ها
        total_requests = await db.requests.count_documents({})
        
        # تعداد تحلیل‌ها به تفکیک شکل چهره
        face_shapes = {}
        pipeline = [
            {"$group": {"_id": "$face_shape", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        
        async for doc in db.analysis_results.aggregate(pipeline):
            face_shapes[doc["_id"]] = doc["count"]
        
        # تعداد به تفکیک دستگاه
        devices = {}
        pipeline = [
            {"$group": {"_id": "$client_info.device_type", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        
        async for doc in db.requests.aggregate(pipeline):
            devices[doc["_id"]] = doc["count"]
        
        # تعداد به تفکیک مرورگر
        browsers = {}
        pipeline = [
            {"$group": {"_id": "$client_info.browser_name", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        
        async for doc in db.requests.aggregate(pipeline):
            browsers[doc["_id"]] = doc["count"]
        
        # انواع فریم پیشنهادی
        frame_types = {}
        pipeline = [
            {"$unwind": "$recommended_frame_types"},
            {"$group": {"_id": "$recommended_frame_types", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        
        async for doc in db.recommendations.aggregate(pipeline):
            frame_types[doc["_id"]] = doc["count"]
        
        return {
            "total_requests": total_requests,
            "face_shapes": face_shapes,
            "devices": devices,
            "browsers": browsers,
            "frame_types": frame_types
        }
        
    except Exception as e:
        logger.error(f"خطا در دریافت خلاصه اطلاعات تحلیلی: {e}")
        return {
            "total_requests": 0,
            "face_shapes": {},
            "devices": {},
            "browsers": {},
            "frame_types": {}
        }