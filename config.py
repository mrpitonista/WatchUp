import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Folders where the downloads will be stored
DOWNLOAD_ROOT = "/Users/Admin/Media/Other"

DOWNLOAD_FOLDERS = {
    "Videos": DOWNLOAD_ROOT,
    "Music": DOWNLOAD_ROOT,
    "Clips": DOWNLOAD_ROOT
}

# Ensure the folder exists
import os
os.makedirs(DOWNLOAD_ROOT, exist_ok=True)

# Optional: disable Celery/Redis if not using yet
CELERY_BROKER_URL = 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'

