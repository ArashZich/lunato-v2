from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os

from app.config import settings
from app.api.face_analysis import router as face_analysis_router
from app.api.health import router as health_router
from app.api.analytics import router as analytics_router
from app.middleware import client_info_middleware
from app.db.connection import connect_to_mongo, close_mongo_connection

# تنظیمات لاگینگ
logging.basicConfig(
    level=logging.INFO if settings.DEBUG else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# بررسی وجود داشتن فایل داده‌های مرجع
if not os.path.exists(settings.FACE_SHAPE_DATA_PATH):
    logging.warning(f"فایل داده‌های مرجع شکل صورت در مسیر {settings.FACE_SHAPE_DATA_PATH} یافت نشد!")


# ایجاد نمونه برنامه FastAPI
app = FastAPI(
    title="سیستم تشخیص چهره و پیشنهاد فریم عینک",
    description="API برای تحلیل شکل صورت و پیشنهاد فریم عینک مناسب",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# افزودن middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # در محیط تولید، دامنه‌های خاص را مشخص کنید
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# افزودن میدلور برای استخراج اطلاعات کاربر
app.middleware("http")(client_info_middleware)

# افزودن روترها
app.include_router(face_analysis_router, prefix="/api/v1", tags=["تحلیل چهره"])
app.include_router(health_router, prefix="/api/v1", tags=["سلامت سیستم"])
app.include_router(analytics_router, prefix="/api/v1", tags=["آمار و تحلیل"])


@app.on_event("startup")
async def startup_event():
    """
    رویداد راه‌اندازی برنامه
    """
    logging.info("سیستم تشخیص چهره و پیشنهاد فریم عینک در حال راه‌اندازی...")
    
    # بررسی و ایجاد دایرکتوری داده در صورت نیاز
    os.makedirs(os.path.dirname(settings.FACE_SHAPE_DATA_PATH), exist_ok=True)
    
    # اتصال به MongoDB
    await connect_to_mongo()


@app.on_event("shutdown")
async def shutdown_event():
    """
    رویداد خاموش شدن برنامه
    """
    logging.info("سیستم تشخیص چهره و پیشنهاد فریم عینک در حال خاموش شدن...")
    
    # بستن اتصال MongoDB
    await close_mongo_connection()


# راه‌اندازی برنامه با uvicorn در صورت اجرای مستقیم این فایل
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )