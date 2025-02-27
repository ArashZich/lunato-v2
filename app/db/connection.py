import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from typing import Optional
import asyncio

from app.config import settings

# تنظیمات لاگر
logger = logging.getLogger(__name__)

# متغیرهای سراسری
_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None


async def connect_to_mongo() -> AsyncIOMotorDatabase:
    """
    اتصال به MongoDB و بازگرداندن دیتابیس.
    
    Returns:
        AsyncIOMotorDatabase: دیتابیس MongoDB
    """
    global _client, _db
    
    if _client is None:
        try:
            # اتصال به سرور MongoDB
            logger.info(f"اتصال به MongoDB: {settings.MONGODB_URI}")
            
            # ایجاد کلاینت با تنظیمات پیشرفته
            _client = AsyncIOMotorClient(
                settings.MONGODB_URI,
                serverSelectionTimeoutMS=10000,  # 10 ثانیه تایم‌اوت برای انتخاب سرور
                connectTimeoutMS=5000,  # 5 ثانیه تایم‌اوت برای اتصال
                socketTimeoutMS=30000,  # 30 ثانیه تایم‌اوت برای عملیات سوکت
                maxPoolSize=10,  # حداکثر تعداد کانکشن‌های هم‌زمان
                minPoolSize=1,  # حداقل تعداد کانکشن‌ها
                maxIdleTimeMS=60000,  # حداکثر زمان بیکاری کانکشن‌ها (1 دقیقه)
                retryWrites=True,  # تلاش مجدد برای عملیات نوشتن
                retryReads=True     # تلاش مجدد برای عملیات خواندن
            )
            
            # بررسی اتصال با زمان انتظار بیشتر
            await _client.admin.command("ping")
            
            logger.info("اتصال به MongoDB با موفقیت برقرار شد")
            
            # انتخاب دیتابیس
            _db = _client[settings.MONGODB_DB_NAME]
            
            # ایجاد ایندکس‌ها
            await _create_indices()
            
            return _db
            
        except Exception as e:
            logger.error(f"خطا در اتصال به MongoDB: {str(e)}")
            if _client:
                _client.close()
                _client = None
            raise
    else:
        return _db


async def _create_indices():
    """ایجاد ایندکس‌های مورد نیاز در کالکشن‌ها"""
    if _db:
        try:
            # ایندکس برای کالکشن درخواست‌ها
            await _db.requests.create_index("request_id", unique=True)
            await _db.requests.create_index("created_at")
            
            # ایندکس برای کالکشن نتایج تحلیل
            await _db.analysis_results.create_index("user_id")
            await _db.analysis_results.create_index("request_id")
            await _db.analysis_results.create_index("face_shape")
            await _db.analysis_results.create_index("created_at")
            
            # ایندکس برای کالکشن پیشنهادات
            await _db.recommendations.create_index("user_id")
            await _db.recommendations.create_index("face_shape")
            await _db.recommendations.create_index("analysis_id")
            await _db.recommendations.create_index("created_at")
            
            logger.info("ایندکس‌های MongoDB با موفقیت ایجاد شدند")
        except Exception as e:
            logger.warning(f"خطا در ایجاد ایندکس‌های MongoDB: {str(e)}")


def get_database() -> AsyncIOMotorDatabase:
    """
    دسترسی به دیتابیس MongoDB.
    
    Returns:
        AsyncIOMotorDatabase: دیتابیس MongoDB
    """
    if _db is None:
        raise RuntimeError("اتصال به MongoDB برقرار نشده است. ابتدا تابع connect_to_mongo را فراخوانی کنید.")
    return _db


async def close_mongo_connection():
    """بستن اتصال به MongoDB"""
    global _client, _db
    
    if _client:
        logger.info("در حال بستن اتصال MongoDB...")
        _client.close()
        _client = None
        _db = None
        logger.info("اتصال MongoDB با موفقیت بسته شد")