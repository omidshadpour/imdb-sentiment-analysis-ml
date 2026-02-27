import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score , f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

class ModelTrainer:
    def __init__(self):

        self.best_model = None
        self.best_model_name = None

        self.models = {
            "Logistic Regression": LogisticRegression(solver = "liblinear"),
            "Linear SVM": LinearSVC(),
            "Naive Bayes": MultinomialNB()
        }


        self.param_grid = {
            "Logistic Regression": {
                "clf__C" : [0.1 , 1 , 3],
                "clf__max_iter" : [1000 , 2000]
            },
            "Linear SVM": {
                "clf__C" : [0.1 , 1 , 3]
            },
            "Naive Bayes": {
                "clf__alpha" : [0.5 , 1.0 , 1.5]
            }
        }


    def train_and_evaluate(self , x_train , y_train , x_valid , y_valid ):
        results = {}
        best_f1 = -1
        best_model = None
        best_name = None

        for name , model in self.models.items():
            print(f"\nTraining {name} with GridSearchCV...")

            pipeline = Pipeline([
                ("tfidf" , TfidfVectorizer(
                    max_features = 20000,
                    ngram_range=(1,2),
                    stop_words = "english"
                )),
                ("clf" , model),
            ])

            grid = GridSearchCV(
                estimator = pipeline,
                param_grid = self.param_grid[name],
                cv = 3 ,
                scoring = "f1",
                n_jobs = -1
            )

            grid.fit(x_train , y_train)

            best_estimator = grid.best_estimator_

            y_pred = best_estimator.predict(x_valid)

            accuracy = accuracy_score(y_valid , y_pred)
            f1 = f1_score(y_valid , y_pred)

            results[name] = {"accuracy" : accuracy , "f1_score" : f1}

            print(f"{name} - Accuracy : {accuracy:.4f} , F1 : {f1:.4f}")
            print("Best params:" , grid.best_params_)

            if f1 > best_f1:
                best_f1 = f1
                best_model = best_estimator
                best_name = name

        # Select best model
        self.best_model = best_model
        self.best_model_name = best_name

        print(f"\nBest model: {self.best_model_name}")

        return results , self.best_model_name

    def save_best_model(self     , path = "models/best_model.pkl"):
        if self.best_model is None:
            raise ValueError("No model trained yet.")

        joblib.dump(self.best_model , path)
        print(f"model saved to {path}")

