from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """تنظیمات اصلی برنامه"""
    
    # تنظیمات API
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")
    DEBUG: bool = Field(default=False, env="DEBUG")
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    
    # تنظیمات MongoDB
    MONGODB_HOST: str = Field(default="mongo", env="MONGODB_HOST")
    MONGODB_PORT: int = Field(default=27017, env="MONGODB_PORT")
    MONGODB_USERNAME: str = Field(default="eyeglass_admin", env="MONGODB_USERNAME")
    MONGODB_PASSWORD: str = Field(default="secure_password123", env="MONGODB_PASSWORD")
    MONGODB_DB_NAME: str = Field(default="eyeglass_recommendation", env="MONGODB_DB_NAME")
    MONGODB_URI: str = Field(
        default="mongodb://eyeglass_admin:secure_password123@mongo:27017/eyeglass_recommendation?authSource=admin", 
        env="MONGODB_URI"
    )
    
    # تنظیمات Celery
    CELERY_BROKER_URL: str = Field(default="redis://redis:6379/0", env="CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: str = Field(default="redis://redis:6379/0", env="CELERY_RESULT_BACKEND")
    
    # تنظیمات WooCommerce API
    WOOCOMMERCE_API_URL: str = Field(
        default="https://lunato.shop/wp-json/wc/v3/products", 
        env="WOOCOMMERCE_API_URL"
    )
    WOOCOMMERCE_CONSUMER_KEY: str = Field(
        default="ck_818f6ea310b3712583afc0d2f12657ae78440b38", 
        env="WOOCOMMERCE_CONSUMER_KEY"
    )
    WOOCOMMERCE_CONSUMER_SECRET: str = Field(
        default="cs_b9e90f2f44c1f262049c7acda1933610fb182571", 
        env="WOOCOMMERCE_CONSUMER_SECRET"
    )
    WOOCOMMERCE_PER_PAGE: int = Field(default=100, env="WOOCOMMERCE_PER_PAGE")
    
    # تنظیمات تشخیص چهره
    FACE_DETECTION_MODEL: str = Field(
        default="haarcascade_frontalface_default.xml",
        env="FACE_DETECTION_MODEL"
    )
    CONFIDENCE_THRESHOLD: float = Field(default=0.5, env="CONFIDENCE_THRESHOLD")
    
    # مسیر فایل داده‌های مرجع
    FACE_SHAPE_DATA_PATH: str = Field(
        default="data/face_shape_frames.json",
        env="FACE_SHAPE_DATA_PATH"
    )
    
    # مسیر مدل آموزش‌دیده
    FACE_SHAPE_MODEL_PATH: str = Field(
        default="data/face_shape_model.pkl",
        env="FACE_SHAPE_MODEL_PATH"
    )
    
    # تنظیمات ذخیره‌سازی
    STORE_ANALYTICS: bool = Field(default=True, env="STORE_ANALYTICS")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


# تابع برای دریافت تنظیمات
def get_settings() -> Settings:
    """دریافت تنظیمات برنامه"""
    return Settings()


# نمونه تنظیمات برای استفاده در کل برنامه
settings = get_settings()