import logging
import json
import aiohttp
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from app.config import settings
# تغییر واردسازی به فایل جدید
from app.core.face_shape_data import get_recommended_frame_types

# تنظیمات لاگر
logger = logging.getLogger(__name__)

# کش برای محصولات
product_cache = None
last_cache_update = None


async def get_all_products() -> List[Dict[str, Any]]:
    """
    دریافت تمام محصولات از WooCommerce API با استفاده از کش.
    
    Returns:
        list: لیست محصولات
    """
    global product_cache, last_cache_update
    
    # استفاده از کش اگر معتبر باشد (کمتر از 1 ساعت)
    if (product_cache is not None and 
        last_cache_update is not None and 
        datetime.utcnow() - last_cache_update < timedelta(hours=1)):
        logger.info("استفاده از کش محصولات WooCommerce")
        return product_cache
    
    try:
        logger.info("دریافت محصولات از WooCommerce API")
        
        # API پارامترها
        api_url = settings.WOOCOMMERCE_API_URL
        consumer_key = settings.WOOCOMMERCE_CONSUMER_KEY
        consumer_secret = settings.WOOCOMMERCE_CONSUMER_SECRET
        per_page = settings.WOOCOMMERCE_PER_PAGE
        
        # مقداردهی اولیه لیست محصولات
        all_products = []
        page = 1
        
        async with aiohttp.ClientSession() as session:
            while True:
                # پارامترهای درخواست
                params = {
                    "consumer_key": consumer_key,
                    "consumer_secret": consumer_secret,
                    "per_page": per_page,
                    "page": page
                }
                
                # ارسال درخواست
                async with session.get(api_url, params=params, timeout=30) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"خطا در WooCommerce API: {response.status} - {error_text}")
                        break
                        
                    products = await response.json()
                    
                    if not products:
                        break
                        
                    all_products.extend(products)
                    
                    # بررسی تعداد محصولات دریافتی
                    if len(products) < per_page:
                        break
                        
                    page += 1
        
        # به‌روزرسانی کش
        product_cache = all_products
        last_cache_update = datetime.utcnow()
        
        logger.info(f"دریافت {len(all_products)} محصول از WooCommerce API")
        return all_products
        
    except Exception as e:
        logger.error(f"خطا در دریافت محصولات از WooCommerce API: {str(e)}")
        return []


def is_eyeglass_frame(product: Dict[str, Any]) -> bool:
    """
    بررسی اینکه آیا محصول یک فریم عینک است.
    
    Args:
        product: محصول WooCommerce
        
    Returns:
        bool: True اگر محصول فریم عینک باشد
    """
    # بررسی دسته‌بندی‌های محصول
    categories = product.get("categories", [])
    for category in categories:
        category_name = category.get("name", "").lower()
        if "عینک" in category_name or "frame" in category_name or "eyeglass" in category_name:
            return True
    
    # بررسی ویژگی‌های محصول
    attributes = product.get("attributes", [])
    for attribute in attributes:
        attr_name = attribute.get("name", "").lower()
        if "frame" in attr_name or "شکل" in attr_name or "فریم" in attr_name or "نوع" in attr_name:
            return True
    
    # بررسی نام محصول
    name = product.get("name", "").lower()
    keywords = ["عینک", "فریم", "eyeglass", "glasses", "frame"]
    for keyword in keywords:
        if keyword in name:
            return True
    
    return False


def get_frame_type(product: Dict[str, Any]) -> str:
    """
    استخراج نوع فریم از محصول.
    
    Args:
        product: محصول WooCommerce
        
    Returns:
        str: نوع فریم
    """
    # بررسی ویژگی‌های محصول
    attributes = product.get("attributes", [])
    
    frame_type_attrs = ["شکل فریم", "نوع فریم", "فرم فریم", "frame type", "frame shape"]
    
    for attribute in attributes:
        attr_name = attribute.get("name", "").lower()
        
        # بررسی اینکه آیا این ویژگی مربوط به نوع فریم است
        is_frame_type_attr = any(frame_type in attr_name for frame_type in frame_type_attrs)
        
        if is_frame_type_attr:
            # دریافت مقدار ویژگی
            if "options" in attribute and attribute["options"]:
                # برگرداندن اولین گزینه
                return attribute["options"][0]
    
    # اگر نوع فریم خاصی پیدا نشد، سعی در استنباط از نام محصول
    name = product.get("name", "").lower()
    
    # نقشه نوع فریم به کلمات کلیدی
    try:
        # بارگیری نقشه از فایل داده
        with open(settings.FACE_SHAPE_DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            frame_type_mappings = data.get('frame_type_mappings', {})
    except Exception as e:
        logger.error(f"خطا در بارگیری نقشه نوع فریم: {str(e)}")
        # نقشه پیش‌فرض
        frame_type_mappings = {
            "مستطیلی": ["مستطیل", "rectangular", "rectangle"],
            "مربعی": ["مربع", "square"],
            "گرد": ["گرد", "round", "circular"],
            "بیضی": ["بیضی", "oval"],
            "گربه‌ای": ["گربه", "cat eye", "cat-eye"],
            "هشت‌ضلعی": ["هشت", "octagonal", "octagon"],
            "هاوایی": ["هاوایی", "aviator"],
            "بدون‌فریم": ["بدون فریم", "rimless"]
        }
    
    for frame_type, keywords in frame_type_mappings.items():
        for keyword in keywords:
            if keyword in name:
                return frame_type
    
    # پیش‌فرض به یک نوع رایج
    return "مستطیلی"


def calculate_match_score(face_shape: str, frame_type: str) -> float:
    """
    محاسبه امتیاز تطابق بین شکل چهره و نوع فریم.
    
    Args:
        face_shape: شکل چهره
        frame_type: نوع فریم
        
    Returns:
        float: امتیاز تطابق (0-100)
    """
    # دریافت انواع فریم توصیه شده برای این شکل چهره
    recommended_types = get_recommended_frame_types(face_shape)
    
    if not recommended_types:
        return 50.0  # امتیاز متوسط پیش‌فرض
    
    # اگر نوع فریم در 2 نوع توصیه شده برتر باشد، امتیاز بالا
    if frame_type in recommended_types[:2]:
        return 90.0 + (recommended_types.index(frame_type) * -5.0)
    
    # اگر در لیست توصیه شده باشد اما نه در 2 نوع برتر، امتیاز متوسط
    if frame_type in recommended_types:
        position = recommended_types.index(frame_type)
        return 80.0 - (position * 5.0)
    
    # اگر در لیست توصیه شده نباشد، امتیاز پایین
    return 40.0


def filter_products_by_price(products: List[Dict[str, Any]], min_price: Optional[float] = None, max_price: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    فیلتر محصولات بر اساس قیمت.
    
    Args:
        products: لیست محصولات
        min_price: حداقل قیمت (اختیاری)
        max_price: حداکثر قیمت (اختیاری)
        
    Returns:
        list: محصولات فیلتر شده
    """
    if min_price is None and max_price is None:
        return products
        
    filtered_products = []
    
    for product in products:
        try:
            price = float(product.get("price", 0))
            
            # بررسی حداقل قیمت
            if min_price is not None and price < min_price:
                continue
                
            # بررسی حداکثر قیمت
            if max_price is not None and price > max_price:
                continue
                
            filtered_products.append(product)
        except (ValueError, TypeError):
            # در صورت خطا در تبدیل قیمت، محصول را نادیده می‌گیریم
            logger.warning(f"خطا در تبدیل قیمت برای محصول: {product.get('id', '')}")
            continue
    
    return filtered_products


def sort_products_by_match_score(products: List[Dict[str, Any]], face_shape: str) -> List[Dict[str, Any]]:
    """
    مرتب‌سازی محصولات بر اساس امتیاز تطابق با شکل چهره.
    
    Args:
        products: لیست محصولات
        face_shape: شکل چهره
        
    Returns:
        list: محصولات مرتب شده
    """
    # محاسبه امتیاز تطابق برای هر محصول
    for product in products:
        frame_type = get_frame_type(product)
        product["match_score"] = calculate_match_score(face_shape, frame_type)
    
    # مرتب‌سازی بر اساس امتیاز تطابق (نزولی)
    return sorted(products, key=lambda x: x.get("match_score", 0), reverse=True)


async def get_recommended_frames(face_shape: str, min_price: Optional[float] = None, max_price: Optional[float] = None, limit: int = 10) -> Dict[str, Any]:
    """
    دریافت فریم‌های پیشنهادی بر اساس شکل چهره.
    
    Args:
        face_shape: شکل چهره
        min_price: حداقل قیمت (اختیاری)
        max_price: حداکثر قیمت (اختیاری)
        limit: حداکثر تعداد توصیه‌ها
        
    Returns:
        dict: نتیجه عملیات شامل فریم‌های توصیه شده
    """
    try:
        logger.info(f"دریافت فریم‌های پیشنهادی برای شکل چهره {face_shape}")
        
        # دریافت انواع فریم توصیه شده برای این شکل چهره
        recommended_frame_types = get_recommended_frame_types(face_shape)
        
        if not recommended_frame_types:
            logger.warning(f"هیچ نوع فریمی برای شکل چهره {face_shape} توصیه نشده است")
            return {
                "success": False,
                "message": f"هیچ توصیه فریمی برای شکل چهره {face_shape} موجود نیست"
            }
        
        # دریافت محصولات از WooCommerce API
        products = await get_all_products()
        
        if not products:
            logger.error("خطا در دریافت محصولات از WooCommerce API")
            return {
                "success": False,
                "message": "خطا در دریافت فریم‌های موجود"
            }
        
        # فیلتر کردن فریم‌های عینک
        eyeglass_frames = [product for product in products if is_eyeglass_frame(product)]
        
        # فیلتر بر اساس قیمت (اگر درخواست شده باشد)
        if min_price is not None or max_price is not None:
            eyeglass_frames = filter_products_by_price(eyeglass_frames, min_price, max_price)
        
        # مرتب‌سازی بر اساس امتیاز تطابق
        sorted_frames = sort_products_by_match_score(eyeglass_frames, face_shape)
        
        # تبدیل به فرمت پاسخ مورد نظر
        recommended_frames = []
        for product in sorted_frames[:limit]:
            frame_type = get_frame_type(product)
            match_score = product.get("match_score", 0)
            
            recommended_frames.append({
                "id": product["id"],
                "name": product["name"],
                "permalink": product["permalink"],
                "price": product.get("price", ""),
                "regular_price": product.get("regular_price", ""),
                "frame_type": frame_type,
                "images": [img["src"] for img in product.get("images", [])[:3]],
                "match_score": match_score
            })
        
        logger.info(f"تطبیق فریم کامل شد: {len(recommended_frames)} توصیه پیدا شد")
        
        return {
            "success": True,
            "face_shape": face_shape,
            "recommended_frame_types": recommended_frame_types,
            "recommended_frames": recommended_frames,
            "total_matches": len(recommended_frames)
        }
        
    except Exception as e:
        logger.error(f"خطا در تطبیق فریم: {str(e)}")
        return {
            "success": False,
            "message": f"خطا در تطبیق فریم: {str(e)}"
        }


async def get_product_by_id(product_id: int) -> Optional[Dict[str, Any]]:
    """
    دریافت یک محصول خاص با شناسه.
    
    Args:
        product_id: شناسه محصول
        
    Returns:
        dict: اطلاعات محصول یا None اگر پیدا نشود
    """
    try:
        logger.info(f"دریافت اطلاعات محصول با شناسه {product_id}")
        
        # API پارامترها
        api_url = f"{settings.WOOCOMMERCE_API_URL}/{product_id}"
        consumer_key = settings.WOOCOMMERCE_CONSUMER_KEY
        consumer_secret = settings.WOOCOMMERCE_CONSUMER_SECRET
        
        # ارسال درخواست
        async with aiohttp.ClientSession() as session:
            params = {
                "consumer_key": consumer_key,
                "consumer_secret": consumer_secret
            }
            
            async with session.get(api_url, params=params, timeout=30) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"خطا در WooCommerce API: {response.status} - {error_text}")
                    return None
                    
                product = await response.json()
                return product
                
    except Exception as e:
        logger.error(f"خطا در دریافت محصول با شناسه {product_id}: {str(e)}")
        return None


async def get_products_by_category(category_id: int) -> List[Dict[str, Any]]:
    """
    دریافت محصولات یک دسته‌بندی خاص.
    
    Args:
        category_id: شناسه دسته‌بندی
        
    Returns:
        list: لیست محصولات
    """
    try:
        logger.info(f"دریافت محصولات دسته‌بندی با شناسه {category_id}")
        
        # API پارامترها
        api_url = settings.WOOCOMMERCE_API_URL
        consumer_key = settings.WOOCOMMERCE_CONSUMER_KEY
        consumer_secret = settings.WOOCOMMERCE_CONSUMER_SECRET
        per_page = settings.WOOCOMMERCE_PER_PAGE
        
        # مقداردهی اولیه لیست محصولات
        category_products = []
        page = 1
        
        async with aiohttp.ClientSession() as session:
            while True:
                # پارامترهای درخواست
                params = {
                    "consumer_key": consumer_key,
                    "consumer_secret": consumer_secret,
                    "category": category_id,
                    "per_page": per_page,
                    "page": page
                }
                
                # ارسال درخواست
                async with session.get(api_url, params=params, timeout=30) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"خطا در WooCommerce API: {response.status} - {error_text}")
                        break
                        
                    products = await response.json()
                    
                    if not products:
                        break
                        
                    category_products.extend(products)
                    
                    # بررسی تعداد محصولات دریافتی
                    if len(products) < per_page:
                        break
                        
                    page += 1
        
        logger.info(f"دریافت {len(category_products)} محصول از دسته‌بندی {category_id}")
        return category_products
        
    except Exception as e:
        logger.error(f"خطا در دریافت محصولات دسته‌بندی {category_id}: {str(e)}")
        return []
                                           