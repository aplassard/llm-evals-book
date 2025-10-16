"""Practical example: Evaluating format conformance and classification on real data."""
from datasets import load_dataset
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from typing import Literal, Optional, List
import json

client = OpenAI()

# start snippet schema
class ReviewAnalysis(BaseModel):
    """Structured analysis of a product review."""
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    key_themes: List[str] = Field(min_items=1, max_items=5)
    would_recommend: bool
# end snippet schema

# start snippet analyze_review
def analyze_review(rating: float, title: str, text: str) -> str:
    """Analyze a product review using an LLM.
    
    Args:
        rating: Star rating (1.0-5.0)
        title: Review title
        text: Review text (will be truncated if too long)
        
    Returns:
        Raw LLM output (may or may not be valid JSON)
    """
    # Truncate long reviews to manage token costs
    max_length = 500
    truncated_text = text[:max_length]
    if len(text) > max_length:
        truncated_text += "..."
    
    prompt = f"""Analyze this product review and return JSON with:
- sentiment: "positive", "negative", or "neutral"
- confidence: float between 0.0 and 1.0
- key_themes: list of 1-5 main themes (strings)
- would_recommend: boolean indicating if reviewer recommends product

Rating: {rating}/5.0
Title: {title}
Review: {truncated_text}

Return ONLY valid JSON, no additional text."""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    return response.choices[0].message.content
# end snippet analyze_review

# start snippet validate_and_parse
def validate_and_parse(raw_output: str) -> tuple[bool, Optional[ReviewAnalysis], Optional[str]]:
    """Validate LLM output against schema.
    
    Args:
        raw_output: Raw text output from LLM
        
    Returns:
        Tuple of (success, parsed_data, error_message)
    """
    try:
        # Handle markdown code fences
        cleaned = raw_output
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0]
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0]
        
        # Parse JSON
        data = json.loads(cleaned.strip())
        
        # Validate against schema
        validated = ReviewAnalysis(**data)
        
        return True, validated, None
        
    except json.JSONDecodeError as e:
        return False, None, f"JSON parsing failed: {str(e)}"
    except ValidationError as e:
        return False, None, f"Schema validation failed: {str(e)}"
# end snippet validate_and_parse

# start snippet ground_truth
def rating_to_sentiment(rating: float) -> str:
    """Convert star rating to expected sentiment.
    
    Args:
        rating: Star rating from 1.0 to 5.0
        
    Returns:
        Expected sentiment based on rating
    """
    if rating >= 4.0:
        return "positive"
    elif rating >= 3.0:
        return "neutral"
    else:
        return "negative"
# end snippet ground_truth

# start snippet evaluate_single
def evaluate_single_review(review: dict, show_details: bool = True) -> dict:
    """Evaluate a single review end-to-end.
    
    Args:
        review: Dictionary with 'rating', 'title', 'text' keys
        show_details: Whether to print detailed output
        
    Returns:
        Dictionary with evaluation results
    """
    # Get LLM analysis
    raw_output = analyze_review(
        review['rating'],
        review['title'], 
        review['text']
    )
    
    # Validate format
    success, parsed, error = validate_and_parse(raw_output)
    
    # Get ground truth
    expected_sentiment = rating_to_sentiment(review['rating'])
    
    # Compute results
    result = {
        'format_valid': success,
        'parse_error': error,
        'predicted_sentiment': parsed.sentiment if parsed else None,
        'expected_sentiment': expected_sentiment,
        'classification_correct': parsed.sentiment == expected_sentiment if parsed else False,
        'confidence': parsed.confidence if parsed else None,
        'raw_output': raw_output
    }
    
    if show_details:
        print(f"\n{'='*70}")
        print(f"Rating: {review['rating']}/5.0")
        print(f"Title: {review['title'][:60]}...")
        print(f"Review: {review['text'][:100]}...")
        print(f"\nExpected sentiment: {expected_sentiment}")
        
        if success:
            print(f"✓ Format valid")
            print(f"  Predicted: {parsed.sentiment}")
            print(f"  Confidence: {parsed.confidence:.2f}")
            print(f"  Themes: {', '.join(parsed.key_themes)}")
            print(f"  Would recommend: {parsed.would_recommend}")
            
            if result['classification_correct']:
                print(f"✓ Classification correct")
            else:
                print(f"✗ Classification incorrect (expected {expected_sentiment})")
        else:
            print(f"✗ Format invalid: {error}")
            print(f"  Raw output preview: {raw_output[:100]}...")
    
    return result
# end snippet evaluate_single


def main():
    """Run practical example with real dataset."""
    print("="*70)
    print("Practical Example: Format Validation & Classification")
    print("="*70)
    
    # Load dataset
    print("\nLoading Amazon Reviews dataset...")
    dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_review_All_Beauty",
        split="full",
        trust_remote_code=True
    )
    
    # Select diverse examples (one of each rating)
    print("Selecting example reviews...")
    examples = []
    for rating in [5.0, 4.0, 3.0, 2.0, 1.0]:
        # Find a review with this rating
        for review in dataset.shuffle(seed=42):
            if review['rating'] == rating:
                examples.append(review)
                break
    
    print(f"\nEvaluating {len(examples)} reviews with diverse ratings...\n")
    
    # Evaluate each example
    results = []
    for i, review in enumerate(examples, 1):
        print(f"\n[Example {i}/{len(examples)}]")
        result = evaluate_single_review(review)
        results.append(result)
    
    # Aggregate metrics
    print("\n" + "="*70)
    print("AGGREGATE RESULTS")
    print("="*70)
    
    total = len(results)
    format_valid = sum(1 for r in results if r['format_valid'])
    classification_correct = sum(1 for r in results if r['classification_correct'])
    
    print(f"\nFormat Conformance:")
    print(f"  Valid outputs: {format_valid}/{total} ({format_valid/total:.1%})")
    print(f"  Parse failures: {total - format_valid}")
    
    print(f"\nClassification Accuracy:")
    print(f"  Correct predictions: {classification_correct}/{total} ({classification_correct/total:.1%})")
    
    # Show error cases
    errors = [r for r in results if not r['format_valid'] or not r['classification_correct']]
    if errors:
        print(f"\nError Analysis:")
        print(f"  Total issues: {len(errors)}")
        format_errors = [r for r in errors if not r['format_valid']]
        if format_errors:
            print(f"  Format failures: {len(format_errors)}")
        classification_errors = [r for r in errors if r['format_valid'] and not r['classification_correct']]
        if classification_errors:
            print(f"  Misclassifications: {len(classification_errors)}")
    else:
        print(f"\n✓ Perfect performance on all examples!")
    
    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    if format_valid == total and classification_correct == total:
        print("\n✓ Excellent results!")
        print("  Both format conformance and classification accuracy are perfect.")
        print("  This prompt and schema are well-aligned for this task.")
    elif format_valid < total * 0.9:
        print("\n⚠ Low format conformance")
        print("  The model is not consistently producing valid JSON.")
        print("  Consider: More explicit format instructions in prompt.")
    elif classification_correct < total * 0.8:
        print("\n⚠ Classification issues")
        print("  Format is good but predictions are inconsistent.")
        print("  Consider: Add few-shot examples for edge cases.")
    else:
        print("\n✓ Good performance with minor issues")
        print("  Review the error cases above to identify patterns.")


if __name__ == "__main__":
    main()
