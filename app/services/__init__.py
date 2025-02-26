# Import service modules
from app.services.tasks import detect_face_task, analyze_face_shape_task, match_frames_task
from app.services.classifier import predict_face_shape, load_model, train_model
from app.services.woocommerce import get_recommended_frames