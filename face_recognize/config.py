from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FACE_DIR = DATA_DIR / "faces"
PERSONS_FILE = DATA_DIR / "persons.json"
DATABASE_PATH = DATA_DIR / "faces.db"
SNAPSHOT_FILE = DATA_DIR / "last_snapshot.jpg"

# Model pack InspireFace
MODEL_PACK_PATH = Path("/home/orangepi/.inspireface/ms/tunmxy/InspireFace/Pikachu")

# ============================================================
# WEB INTEGRATION
# ============================================================
FRAME_URL = "http://127.0.0.1:8080/frame.jpg"
FACE_API_URL = "http://127.0.0.1:8080/api/face_result"
SHM_FACE_CROP = Path("/dev/shm/face_crop.jpg")

# ============================================================
# THRESHOLDS & SETTINGS
# ============================================================
SIMILARITY_THRESHOLD = 0.48
RECOGNIZE_INTERVAL = 0.12  # ~8 FPS (Tối ưu tải CPU, giải phóng tài nguyên cho MediaPipe và Video)

GENDER_MAP = {
    0: "Nữ",
    1: "Nam",
    -1: "Không xác định"
}

AGE_BRACKET_MAP = {
    0: "0 - 2 tuổi",
    1: "3 - 9 tuổi",
    2: "10 - 19 tuổi",
    3: "20 - 29 tuổi",
    4: "30 - 39 tuổi",
    5: "40 - 49 tuổi",
    6: "50 - 59 tuổi",
    7: "60+ tuổi",
    -1: "Không xác định"
}

EMOTION_MAP = {
    0: "Bình thường",
    1: "Tức giận",
    2: "Ghê sợ",
    3: "Vui vẻ",
    4: "Buồn bã",
    5: "Ngạc nhiên",
    6: "Điềm tĩnh",
    -1: "Bình thường"
}
