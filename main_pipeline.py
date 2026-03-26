from config.config import ACCURACY_THRESHOLD
from src.data.data_ingestion import ingest_data, split_data
from src.pipelines.sklearn_pipeline import model_pipeline
from src.models.modeltrain import train
from src.models.evaluation import evaluate
from src.feature_engineering.feat_eng import feature_engineering
from config.config import DATA_ING_DIR
import pandas as pd

def main():
    print("=" * 50)
    print("Approach B – sklearn Pipeline")
    print("=" * 50)

    print("\nStep 1: Data Ingestion")
    ingest_data()

    print("\nStep 2: Load and Feature Engineering")
    df = pd.read_csv(DATA_ING_DIR / "train.csv")
    print(df.head(5))
    df = feature_engineering(df)

    print("\nStep 3: Split the data")
    train_scaled, test_scaled = split_data(df)

    print("\nStep 4: Build and Train Pipeline")
    pipeline_train = model_pipeline(train_scaled)
    pipeline_test = model_pipeline(test_scaled)
    
    run_id = train(pipeline_train, train_scaled)

    print("\nStep 5: Evaluation")
    accuracy, precision, recall = evaluate(test_scaled, run_id)

    print("\n" + "=" * 50)
    if accuracy >= ACCURACY_THRESHOLD:
        print(f"Model APPROVED (accuracy={accuracy:.3f})")
    else:
        print(f"Model REJECTED (accuracy={accuracy:.3f} < {ACCURACY_THRESHOLD})")


if __name__ == "__main__":
    main()
