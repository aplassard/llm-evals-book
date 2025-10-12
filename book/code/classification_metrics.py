"""Classification evaluation metrics."""
from typing import List, Dict


def compute_accuracy(
    predictions: List[str], 
    ground_truth: List[str]
) -> float:
    """Compute overall classification accuracy.
    
    Args:
        predictions: List of predicted labels
        ground_truth: List of true labels
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    assert len(predictions) == len(ground_truth), \
        "Predictions and ground truth must have same length"
    
    correct = sum(p == g for p, g in zip(predictions, ground_truth))
    return correct / len(predictions) if predictions else 0.0


def confusion_matrix(
    predictions: List[str],
    ground_truth: List[str],
    labels: List[str]
) -> Dict[str, Dict[str, int]]:
    """Build confusion matrix showing prediction patterns.
    
    Args:
        predictions: List of predicted labels
        ground_truth: List of true labels
        labels: List of all possible labels
        
    Returns:
        Nested dict where matrix[true_label][pred_label] = count
    """
    matrix = {
        label: {label: 0 for label in labels} 
        for label in labels
    }
    
    for pred, true in zip(predictions, ground_truth):
        if true in matrix and pred in matrix[true]:
            matrix[true][pred] += 1
    
    return matrix


def print_confusion_matrix(
    matrix: Dict[str, Dict[str, int]],
    labels: List[str]
):
    """Pretty print confusion matrix.
    
    Args:
        matrix: Confusion matrix from confusion_matrix()
        labels: List of labels in desired order
    """
    # Header
    print("\nConfusion Matrix:")
    print("True \\ Pred", end="")
    for label in labels:
        print(f"  {label:>8}", end="")
    print()
    
    # Separator
    print("-" * (12 + 10 * len(labels)))
    
    # Rows
    for true_label in labels:
        print(f"{true_label:>12}", end="")
        for pred_label in labels:
            count = matrix.get(true_label, {}).get(pred_label, 0)
            print(f"  {count:>8}", end="")
        print()


def per_class_metrics(
    predictions: List[str],
    ground_truth: List[str],
    target_class: str
) -> Dict[str, float]:
    """Compute precision, recall, and F1 for a specific class.
    
    Args:
        predictions: List of predicted labels
        ground_truth: List of true labels
        target_class: Class to compute metrics for
        
    Returns:
        Dict with precision, recall, and f1_score
    """
    tp = sum(1 for p, g in zip(predictions, ground_truth)
             if p == target_class and g == target_class)
    fp = sum(1 for p, g in zip(predictions, ground_truth)
             if p == target_class and g != target_class)
    fn = sum(1 for p, g in zip(predictions, ground_truth)
             if p != target_class and g == target_class)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "support": tp + fn  # Total true examples of this class
    }


def classification_report(
    predictions: List[str],
    ground_truth: List[str],
    labels: List[str]
) -> Dict[str, Dict[str, float]]:
    """Generate full classification report with per-class metrics.
    
    Args:
        predictions: List of predicted labels
        ground_truth: List of true labels
        labels: List of all classes to evaluate
        
    Returns:
        Dict mapping each class to its metrics dict
    """
    report = {}
    
    for label in labels:
        report[label] = per_class_metrics(
            predictions, 
            ground_truth, 
            label
        )
    
    # Add overall metrics
    report["overall"] = {
        "accuracy": compute_accuracy(predictions, ground_truth),
        "total_samples": len(predictions)
    }
    
    return report


def print_classification_report(
    report: Dict[str, Dict[str, float]]
):
    """Pretty print classification report.
    
    Args:
        report: Report from classification_report()
    """
    print("\nClassification Report:")
    print("-" * 65)
    print(f"{'Class':>12}  {'Precision':>10}  {'Recall':>10}  "
          f"{'F1-Score':>10}  {'Support':>8}")
    print("-" * 65)
    
    for class_name, metrics in report.items():
        if class_name == "overall":
            continue
        
        print(f"{class_name:>12}  "
              f"{metrics['precision']:>10.3f}  "
              f"{metrics['recall']:>10.3f}  "
              f"{metrics['f1_score']:>10.3f}  "
              f"{metrics['support']:>8.0f}")
    
    print("-" * 65)
    if "overall" in report:
        print(f"{'Accuracy':>12}  {report['overall']['accuracy']:>10.3f}  "
              f"{'':>10}  {'':>10}  "
              f"{report['overall']['total_samples']:>8.0f}")
    print()


if __name__ == "__main__":
    # Example usage
    predictions = ["urgent", "low", "normal", "urgent", "low", "normal"]
    ground_truth = ["urgent", "low", "urgent", "urgent", "low", "normal"]
    labels = ["urgent", "normal", "low"]
    
    # Accuracy
    acc = compute_accuracy(predictions, ground_truth)
    print(f"Accuracy: {acc:.1%}")
    
    # Confusion matrix
    cm = confusion_matrix(predictions, ground_truth, labels)
    print_confusion_matrix(cm, labels)
    
    # Full report
    report = classification_report(predictions, ground_truth, labels)
    print_classification_report(report)
