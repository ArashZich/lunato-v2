from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime


class RequestRecord(BaseModel):
    """مدل رکورد درخواست‌ها"""
    request_id: str
    path: str
    method: str
    client_info: Dict[str, Any]
    status_code: int
    process_time: float
    created_at: datetime


class AnalysisRecord(BaseModel):
    """مدل رکورد نتایج تحلیل چهره"""
    user_id: str
    request_id: str
    face_shape: str
    confidence: float
    client_info: Dict[str, Any]
    task_id: Optional[str] = None
    created_at: datetime


class RecommendationRecord(BaseModel):
    """مدل رکورد پیشنهادات فریم"""
    user_id: str
    face_shape: str
    recommended_frame_types: List[str]
    recommended_frames: List[Dict[str, Any]]
    client_info: Dict[str, Any]
    analysis_id: Optional[str] = None
    created_at: datetime


class AnalyticsSummary(BaseModel):
    """مدل خلاصه اطلاعات تحلیلی"""
    total_requests: int = 0
    face_shapes: Dict[str, int] = {}
    devices: Dict[str, int] = {}
    browsers: Dict[str, int] = {}
    frame_types: Dict[str, int] = {}