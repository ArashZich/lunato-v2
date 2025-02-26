from fastapi import APIRouter, Depends, HTTPException, Request, Query
import logging
from typing import Optional
from datetime import datetime, timedelta

from app.db.repository import get_analytics_summary
from app.models.database import AnalyticsSummary

# تنظیمات لاگر
logger = logging.getLogger(__name__)

# تعریف روتر
router = APIRouter()


@router.get("/analytics/summary", response_model=AnalyticsSummary)
async def get_analytics_summary_api():
    """
    دریافت خلاصه اطلاعات تحلیلی سیستم.
    
    این API اطلاعات کلی در مورد تحلیل‌های انجام شده، شکل‌های چهره، و دستگاه‌های کاربران را ارائه می‌دهد.
    """
    try:
        # دریافت خلاصه اطلاعات تحلیلی
        summary = await get_analytics_summary()
        
        # بازگرداندن نتیجه
        return AnalyticsSummary(**summary)
        
    except Exception as e:
        logger.error(f"خطا در دریافت خلاصه اطلاعات تحلیلی: {str(e)}")
        raise HTTPException(status_code=500, detail=f"خطا در دریافت اطلاعات تحلیلی: {str(e)}")