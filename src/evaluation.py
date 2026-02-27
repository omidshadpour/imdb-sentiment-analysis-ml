import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.metrics import confusion_matrix , classification_report ,roc_curve ,auc


def evaluate_model(model, x_test, y_test , model_name = "Best Model"):
    # Create folder if not exists
    os.makedirs("models/reports", exist_ok= True)


    print("\n===== Evaluation Report =====")
    y_pred = model.predict(x_test)

    # 1) Classification Report
    print("\n--- Classification Report ---")
    report = classification_report(y_test, y_pred , target_names = ["Negative", "Positive"])
    print(report)


    # 2) Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize = (6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues" , xticklabels= ["Negative", "Positive"] , yticklabels = ["Negative", "Positive"])
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("reports/plots/confusion_matrix.png")
    plt.show()

    # 3) Normalized Confusion Matrix
    cm_norm = confusion_matrix(y_test, y_pred, normalize="true")
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens", xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.title(f"{model_name} - Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("reports/plots/confusion_matrix_normalized.png")
    plt.show()


    # 4) ROC Curve
    try:
        # Some models (like LinearSVC) don't have predict_proba
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(x_test)[: , 1]

        elif hasattr(model , "decision_function"):
            # fallback using decision_function
            y_prob = model.decision_function(x_test)
        else:
            raise AttributeError("Model does not support probability scoring.")


        fpr , tpr , _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize = (6,5))
        plt.plot(fpr, tpr, label = f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.title(f"{model_name} - ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig("reports/plots/figroc_curve.png")
        plt.show()

    except Exception as e:
        print("\nROC Curve could not be generated for this model.")
        print("Reason:", e)

    return {"classification_report" : report , "confusion_matrix" :cm}

