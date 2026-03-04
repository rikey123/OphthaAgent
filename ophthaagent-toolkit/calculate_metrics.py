import json
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score

def calculate_detailed_metrics(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    y_true = []
    y_pred = []
    
    for item in data['details']:
        y_true.append(item['gt'])
        y_pred.append(item['pred'])
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    # 2. Per-class metrics
    # Sensitivity (Recall), Specificity, Precision, F1
    metrics = {}
    for i in range(3):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        metrics[f"Class {i}"] = {
            "Sensitivity (Recall)": sensitivity,
            "Specificity": specificity,
            "Precision": precision,
            "F1-score": f1,
            "Support": int(tp + fn)
        }
    
    # 3. Global metrics
    report = classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=["Level 0", "Level 1", "Level 2"], output_dict=True)
    kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    # Balanced Accuracy
    balanced_acc = np.mean([metrics[f"Class {i}"]["Sensitivity (Recall)"] for i in range(3)])

    results = {
        "Confusion Matrix": cm.tolist(),
        "Per-class Metrics": metrics,
        "Macro F1": report['macro avg']['f1-score'],
        "Weighted F1": report['weighted avg']['f1-score'],
        "Quadratic Weighted Kappa": kappa,
        "Balanced Accuracy": balanced_acc,
        "Overall Accuracy": data['accuracy']
    }
    
    return results

if __name__ == "__main__":
    path = "<ANON_ABS_PATH>"
    res = calculate_detailed_metrics(path)
    print(json.dumps(res, indent=4, ensure_ascii=False))
