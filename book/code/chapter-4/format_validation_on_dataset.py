"""Format validation on Amazon Reviews 2023 dataset."""
from datasets import load_dataset
from openai import OpenAI
from product_review_schema import ReviewAnalysis
from format_evaluation import (
    validate_output, 
    compute_conformance,
    ConformanceMetrics
)
import json
from typing import List, Tuple


client = OpenAI()


def analyze_review_from_dataset(
    rating: float, 
    title: str, 
    text: str
) -> str:
    """Analyze a review from the dataset.
    
    Args:
        rating: Star rating (1.0-5.0)
        title: Review title/headline
        text: Full review text
        
    Returns:
        Raw LLM output (may or may not be valid JSON)
    """
    prompt = f"""Analyze this product review and return JSON:
    
    - sentiment: "positive", "negative", "neutral", or "mixed"
    - confidence: float 0.0-1.0
    - key_themes: list of 1-5 main themes
    - product_issues: optional list of problems mentioned
    - recommendations: list of actionable recommendations
    
    Rating: {rating}/5.0
    Title: {title}
    Review: {text}
    
    Return ONLY valid JSON."""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    return response.choices[0].message.content


def evaluate_on_dataset(
    num_samples: int = 50
) -> Tuple[ConformanceMetrics, List]:
    """Evaluate format conformance on Amazon Reviews dataset.
    
    Args:
        num_samples: Number of reviews to sample and evaluate
        
    Returns:
        Tuple of (ConformanceMetrics, list of ValidationResults)
    """
    
    print(f"Loading {num_samples} reviews from dataset...")
    
    # Load Beauty category reviews (smaller subset for faster testing)
    dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_review_All_Beauty",
        split="full",
        trust_remote_code=True
    )
    
    # Sample reviews randomly but deterministically
    sampled = dataset.shuffle(seed=42).select(range(num_samples))
    
    print(f"\nProcessing {len(sampled)} reviews...\n")
    print("=" * 70)
    
    results = []
    for i, review in enumerate(sampled, 1):
        # Get LLM analysis
        raw_output = analyze_review_from_dataset(
            review['rating'],
            review['title'],
            review['text'][:500]  # Limit length to reduce cost
        )
        
        # Validate against schema
        result = validate_output(raw_output, ReviewAnalysis)
        results.append(result)
        
        # Progress indicator
        if i % 10 == 0:
            print(f"Processed {i}/{len(sampled)} reviews...")
    
    # Compute conformance metrics
    metrics = compute_conformance(results)
    
    print("\n" + "=" * 70)
    print("\nConformance Metrics:")
    print(f"  Total reviews: {metrics.total}")
    print(f"  Valid outputs: {metrics.successful}")
    print(f"  JSON parse failures: {metrics.json_parse_failures}")
    print(f"  Schema validation failures: {metrics.schema_validation_failures}")
    print(f"\n  Conformance rate: {metrics.conformance_rate:.1%}")
    print(f"  Parse rate: {metrics.parse_rate:.1%}")
    
    # Analyze failures
    print("\n" + "=" * 70)
    print("\nFailure Analysis:")
    
    failures = [r for r in results if not r.success]
    if failures:
        print(f"\nSample failures ({min(3, len(failures))} shown):")
        for i, failure in enumerate(failures[:3], 1):
            print(f"\n[Failure {i}]")
            print(f"Error: {failure.validation_error[:100]}...")
            print(f"Output preview: {failure.raw_output[:100]}...")
    else:
        print("  No failures! Perfect conformance.")
    
    return metrics, results


def main():
    """Run format validation evaluation on dataset."""
    
    # Run evaluation on sample
    metrics, results = evaluate_on_dataset(num_samples=50)
    
    # Save results for further analysis
    output_file = "conformance_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "conformance_rate": metrics.conformance_rate,
            "parse_rate": metrics.parse_rate,
            "total": metrics.total,
            "successful": metrics.successful,
            "json_parse_failures": metrics.json_parse_failures,
            "schema_validation_failures": metrics.schema_validation_failures,
            "failure_count": len([r for r in results if not r.success])
        }, f, indent=2)
    
    print(f"\n\nResults saved to {output_file}")
    print("\nInterpretation:")
    
    if metrics.conformance_rate >= 0.9:
        print("  ✓ High conformance - prompt and schema well-aligned")
    elif metrics.parse_rate < 0.7:
        print("  ⚠ Low parse rate - model not producing valid JSON")
        print("    Recommendation: Improve prompt format instructions")
    elif metrics.conformance_rate < metrics.parse_rate:
        print("  ⚠ Schema misalignment - valid JSON but wrong structure")
        print("    Recommendation: Clarify field requirements in prompt")
    else:
        print("  ⚠ Inconsistent conformance - review failure patterns")


if __name__ == "__main__":
    main()
