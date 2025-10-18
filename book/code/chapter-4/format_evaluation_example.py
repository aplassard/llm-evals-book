"""Example: Evaluating format conformance for product review analysis."""
# start snippet format_evaluation_example
from openai import OpenAI
from product_review_schema import ReviewAnalysis
from format_evaluation import validate_output, compute_conformance


client = OpenAI()


def analyze_review(review_text: str) -> str:
    """Get structured analysis from LLM.
    
    Args:
        review_text: Product review text to analyze
        
    Returns:
        Raw LLM output (may or may not be valid JSON)
    """
    prompt = f"""Analyze this product review and return a JSON object with:
    - sentiment: "positive", "negative", "neutral", or "mixed"
    - confidence: float between 0.0 and 1.0
    - key_themes: list of 1-5 main themes (strings)
    - product_issues: optional list of specific problems
    - recommendations: list of actionable recommendations
    
    Review: {review_text}
    
    Return ONLY the JSON object, no additional text."""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    return response.choices[0].message.content
# end snippet format_evaluation_example


def main():
    """Run format evaluation example."""
    # Test with diverse reviews
    test_reviews = [
        "Great product! Fast delivery and exactly as described.",
        "Disappointed. Cheap material, broke after a week.",
        "It's okay. Does the job but nothing special.",
        "Amazing! Exceeded expectations. Great service.",
        "Waste of money. Doesn't work as advertised."
    ]
    
    print("Evaluating format conformance for product review analysis\n")
    print("=" * 70)
    
    results = []
    for i, review in enumerate(test_reviews, 1):
        print(f"\n[Review {i}]: {review[:60]}...")
        
        # Get LLM analysis
        raw_output = analyze_review(review)
        
        # Validate against schema
        result = validate_output(raw_output, ReviewAnalysis)
        results.append(result)
        
        # Report result
        if result.success:
            print("✓ Valid - conforms to schema")
            print(f"  Sentiment: {result.parsed_data['sentiment']}")
            print(f"  Confidence: {result.parsed_data['confidence']:.2f}")
            print(f"  Themes: {', '.join(result.parsed_data['key_themes'])}")
        else:
            print("✗ Invalid - validation failed")
            print(f"  Error: {result.validation_error}")
    
    # Compute aggregate metrics
    print("\n" + "=" * 70)
    metrics = compute_conformance(results)
    print("\nAggregate Conformance Metrics:")
    print(f"  Total outputs: {metrics.total}")
    print(f"  Successful validations: {metrics.successful}")
    print(f"  JSON parse failures: {metrics.json_parse_failures}")
    print(f"  Schema validation failures: {metrics.schema_validation_failures}")
    print(f"\n  Conformance rate: {metrics.conformance_rate:.1%}")
    print(f"  Parse rate: {metrics.parse_rate:.1%}")
    
    # Interpretation
    print("\nInterpretation:")
    if metrics.conformance_rate >= 0.9:
        print("  ✓ High conformance - prompt and schema are well-aligned")
    elif metrics.parse_rate < 0.7:
        print("  ⚠ Low parse rate - model isn't producing valid JSON")
        print("    Recommendation: Improve prompt to explicitly require JSON format")
    elif metrics.conformance_rate < metrics.parse_rate:
        print("  ⚠ Schema misalignment - valid JSON but wrong structure")
        print("    Recommendation: Clarify field requirements in prompt")
    else:
        print("  ⚠ Inconsistent conformance - review failure patterns")


if __name__ == "__main__":
    main()
