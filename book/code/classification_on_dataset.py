"""Sentiment classification evaluation on Amazon Reviews 2023 dataset."""
from datasets import load_dataset
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal, List, Dict
import json
from classification_metrics import (
    compute_accuracy,
    confusion_matrix,
    print_confusion_matrix,
    classification_report,
    print_classification_report
)


client = OpenAI()


class SentimentClassification(BaseModel):
    """Sentiment classification result."""
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="Brief explanation for classification")


def rating_to_sentiment(rating: float) -> str:
    """Convert star rating to ground truth sentiment.
    
    Args:
        rating: Star rating from 1.0 to 5.0
        
    Returns:
        Sentiment label: positive (4-5), neutral (3), negative (1-2)
    """
    if rating >= 4.0:
        return "positive"
    elif rating >= 3.0:
        return "neutral"
    else:
        return "negative"


def classify_review_sentiment(
    title: str,
    text: str,
    max_length: int = 500
) -> SentimentClassification:
    """Classify review sentiment using LLM.
    
    Args:
        title: Review title/headline
        text: Full review text
        max_length: Maximum characters to include from text
        
    Returns:
        SentimentClassification with sentiment, confidence, and reasoning
    """
    # Truncate text to reduce token cost
    truncated_text = text[:max_length]
    if len(text) > max_length:
        truncated_text += "..."
    
    prompt = f"""Classify the sentiment of this product review.

Categories:
- positive: Customer is satisfied, recommends product, highlights benefits
- neutral: Mixed feelings, balanced pros/cons, "it's okay"
- negative: Customer is dissatisfied, complains, would not recommend

Return JSON with:
1. sentiment: one of the three categories
2. confidence: float 0.0-1.0 indicating classification confidence
3. reasoning: one sentence explaining the classification

Title: {title}
Review: {truncated_text}

Return ONLY valid JSON with sentiment, confidence, and reasoning fields."""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    output = response.choices[0].message.content
    
    # Extract JSON (handle markdown fences)
    if "```" in output:
        output = output.split("```")[1]
        if output.startswith("json"):
            output = output[4:]
    
    data = json.loads(output.strip())
    return SentimentClassification(**data)


def evaluate_classification(
    num_samples: int = 100,
    seed: int = 42
) -> Dict:
    """Evaluate sentiment classification on Amazon Reviews dataset.
    
    Args:
        num_samples: Number of reviews to evaluate
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with predictions, ground truth, and metrics
    """
    
    print(f"Loading {num_samples} reviews from Amazon Reviews 2023...")
    
    # Load Beauty category reviews
    dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_review_All_Beauty",
        split="full",
        trust_remote_code=True
    )
    
    # Sample reviews with diverse ratings
    sampled = dataset.shuffle(seed=seed).select(range(num_samples))
    
    print(f"\nProcessing {len(sampled)} reviews...")
    print("=" * 70)
    
    predictions = []
    ground_truth = []
    confidences = []
    details = []
    
    for i, review in enumerate(sampled, 1):
        # Get ground truth from rating
        true_sentiment = rating_to_sentiment(review['rating'])
        
        # Get LLM prediction
        try:
            result = classify_review_sentiment(
                review['title'],
                review['text']
            )
            pred_sentiment = result.sentiment
            confidence = result.confidence
            reasoning = result.reasoning
        except Exception as e:
            print(f"Error on review {i}: {e}")
            # Default to neutral on errors
            pred_sentiment = "neutral"
            confidence = 0.0
            reasoning = f"Error: {str(e)}"
        
        predictions.append(pred_sentiment)
        ground_truth.append(true_sentiment)
        confidences.append(confidence)
        
        details.append({
            "review_id": i,
            "rating": review['rating'],
            "title": review['title'],
            "true_sentiment": true_sentiment,
            "predicted_sentiment": pred_sentiment,
            "confidence": confidence,
            "reasoning": reasoning,
            "correct": pred_sentiment == true_sentiment
        })
        
        # Progress indicator
        if i % 20 == 0:
            print(f"Processed {i}/{len(sampled)} reviews...")
    
    print("\n" + "=" * 70)
    
    return {
        "predictions": predictions,
        "ground_truth": ground_truth,
        "confidences": confidences,
        "details": details
    }


def analyze_results(results: Dict):
    """Analyze and print classification results.
    
    Args:
        results: Dictionary from evaluate_classification()
    """
    predictions = results["predictions"]
    ground_truth = results["ground_truth"]
    confidences = results["confidences"]
    details = results["details"]
    
    labels = ["positive", "neutral", "negative"]
    
    print("\n📊 CLASSIFICATION RESULTS")
    print("=" * 70)
    
    # Overall accuracy
    accuracy = compute_accuracy(predictions, ground_truth)
    print(f"\n✓ Overall Accuracy: {accuracy:.1%}")
    print(f"  Correctly classified: {sum(d['correct'] for d in details)}/{len(details)}")
    
    # Average confidence
    avg_confidence = sum(confidences) / len(confidences)
    print(f"\n✓ Average Confidence: {avg_confidence:.2f}")
    
    # Confidence by correctness
    correct_confidences = [c for c, d in zip(confidences, details) if d['correct']]
    incorrect_confidences = [c for c, d in zip(confidences, details) if not d['correct']]
    
    if correct_confidences:
        print(f"  Correct predictions: {sum(correct_confidences)/len(correct_confidences):.2f}")
    if incorrect_confidences:
        print(f"  Incorrect predictions: {sum(incorrect_confidences)/len(incorrect_confidences):.2f}")
    
    # Confusion matrix
    cm = confusion_matrix(predictions, ground_truth, labels)
    print_confusion_matrix(cm, labels)
    
    # Per-class metrics
    report = classification_report(predictions, ground_truth, labels)
    print_classification_report(report)
    
    # Error analysis
    print("\n" + "=" * 70)
    print("\n🔍 ERROR ANALYSIS")
    print("=" * 70)
    
    errors = [d for d in details if not d['correct']]
    
    if errors:
        print(f"\nTotal errors: {len(errors)}")
        
        # Group errors by type
        error_types = {}
        for error in errors:
            key = f"{error['true_sentiment']} → {error['predicted_sentiment']}"
            if key not in error_types:
                error_types[key] = []
            error_types[key].append(error)
        
        print("\nError breakdown:")
        for error_type, error_list in sorted(
            error_types.items(), 
            key=lambda x: len(x[1]), 
            reverse=True
        ):
            print(f"  {error_type}: {len(error_list)} cases")
        
        # Show sample errors
        print("\nSample errors:")
        for i, error in enumerate(errors[:3], 1):
            print(f"\n[Error {i}]")
            print(f"  Rating: {error['rating']}/5.0")
            print(f"  Title: {error['title'][:60]}...")
            print(f"  True: {error['true_sentiment']}")
            print(f"  Predicted: {error['predicted_sentiment']} "
                  f"(confidence: {error['confidence']:.2f})")
            print(f"  Reasoning: {error['reasoning'][:80]}...")
    else:
        print("\n✓ Perfect classification! No errors found.")
    
    # Confidence calibration analysis
    print("\n" + "=" * 70)
    print("\n📈 CONFIDENCE CALIBRATION")
    print("=" * 70)
    
    # Bin predictions by confidence
    confidence_bins = {
        "high (>0.8)": [d for d in details if d['confidence'] > 0.8],
        "medium (0.5-0.8)": [d for d in details if 0.5 <= d['confidence'] <= 0.8],
        "low (<0.5)": [d for d in details if d['confidence'] < 0.5]
    }
    
    print("\nAccuracy by confidence level:")
    for bin_name, bin_details in confidence_bins.items():
        if bin_details:
            bin_accuracy = sum(d['correct'] for d in bin_details) / len(bin_details)
            print(f"  {bin_name}: {bin_accuracy:.1%} ({len(bin_details)} samples)")


def main():
    """Run sentiment classification evaluation."""
    
    # Run evaluation
    results = evaluate_classification(num_samples=100)
    
    # Analyze results
    analyze_results(results)
    
    # Save detailed results
    output_file = "classification_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "accuracy": compute_accuracy(
                results["predictions"], 
                results["ground_truth"]
            ),
            "num_samples": len(results["predictions"]),
            "details": results["details"]
        }, f, indent=2)
    
    print(f"\n\nDetailed results saved to {output_file}")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("\n💡 INTERPRETATION")
    print("=" * 70)
    
    accuracy = compute_accuracy(results["predictions"], results["ground_truth"])
    
    if accuracy >= 0.85:
        print("\n✓ Excellent performance!")
        print("  The LLM accurately classifies sentiment across diverse reviews.")
        print("  Consider: Can you use a smaller/cheaper model with similar accuracy?")
    elif accuracy >= 0.70:
        print("\n✓ Good performance with room for improvement.")
        print("  Check the error analysis above for patterns:")
        print("    - Are neutral reviews being misclassified?")
        print("    - Do errors concentrate in specific rating ranges?")
        print("  Consider: Add few-shot examples for problem cases.")
    else:
        print("\n⚠ Performance below expectations.")
        print("  Review the confusion matrix to identify systematic issues:")
        print("    - Is one class consistently misclassified?")
        print("    - Are predictions biased toward certain categories?")
        print("  Consider:")
        print("    - Refine category definitions in the prompt")
        print("    - Add explicit examples of each sentiment type")
        print("    - Use a more capable model")
    
    # Confidence analysis
    avg_conf = sum(results["confidences"]) / len(results["confidences"])
    errors = [d for d in results["details"] if not d['correct']]
    
    if errors:
        error_conf = sum(d['confidence'] for d in errors) / len(errors)
        
        if error_conf > 0.7:
            print("\n⚠ Model is overconfident on errors!")
            print(f"  Average confidence on incorrect predictions: {error_conf:.2f}")
            print("  This suggests the model doesn't know when it's uncertain.")
            print("  Consider: Implement confidence thresholding to reject low-confidence predictions.")


if __name__ == "__main__":
    main()
