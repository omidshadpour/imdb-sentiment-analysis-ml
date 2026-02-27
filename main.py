from src.data_preprocessing import DataPreprocessor
from src.text_preprocessing import TextPreprocessor
from src.train_models import ModelTrainer
from src.evaluation import evaluate_model


def main():
    # Load and split the dataset
    print("Loading Dataset...")
    pre = DataPreprocessor("data/raw/IMDB Dataset.csv")
    df = pre.load_data()
    train_df , test_df = pre.split_data()

    print("Dataset loaded")
    print("Train size:" , train_df.shape)
    print("Test size:" , test_df.shape)

    # Text preprocessing
    print("Cleaning and Lemmatizing text...")

    tp = TextPreprocessor()
    x_train = tp.preprocess_series(train_df["review"])
    x_test = tp.preprocess_series(test_df["review"])

    print("Text preprocessing completed")

    # Labels
    y_train = train_df["sentiment"].map({"positive": 1, "negative": 0})
    y_test = test_df["sentiment"].map({"positive": 1, "negative": 0})

    # Train models with GridSearchCV + Pipeline
    print("\nTraining Models...")
    trainer = ModelTrainer()

    results , best_model_name = trainer.train_and_evaluate(x_train, y_train , x_test , y_test)

    print("\nAll model results:")
    for model_name , metrics in results.items():
        print(f"{model_name} : Accuracy = {metrics['accuracy']:.4f} , F1 = {metrics['f1_score']:.4f}")

    print(f"\nBest model selected: {best_model_name}")

    # Evaluate best model
    print("\nRunning Evaluation...")
    best_model = trainer.best_model
    evaluate_model(best_model, x_test , y_test , model_name = best_model_name)


    # Save best model
    trainer.save_best_model(path = "models/best_model.pkl")



if __name__ == "__main__":
    main()

