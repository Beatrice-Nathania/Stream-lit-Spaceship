from pathlib import Path

BASE_DIR      = Path(__file__).resolve().parent.parent
DATA_RAW_DIR  = BASE_DIR / "data" / "raw"
DATA_ING_DIR  = BASE_DIR / "data" / "ingested"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

ARTIFACT_PIPELINE     = ARTIFACTS_DIR / "model_logreg.pkl"

TARGET_COL = "Transported"
DROP_COLS  = ["PassengerId"]

NUM_FEATURES = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
CAT_FEATURES = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP','Name']

LG_C = 0.316735
LG_MAXITER = 200
RANDOM_STATE = 42
TEST_SIZE    = 0.2

MLFLOW_TRACKING_URI = f"sqlite:///{BASE_DIR.parent / 'mlflow.db'}"
MLFLOW_EXP_PIPELINE = "Transported Prediction"

ACCURACY_THRESHOLD = 0.7
