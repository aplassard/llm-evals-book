"""Format evaluation with LLM healing for malformed outputs."""
from openai import OpenAI
from product_review_schema import ReviewAnalysis
from format_evaluation import extract_json, ValidationResult
import json
from pydantic import ValidationError
from typing import List
from dataclasses import dataclass


client = OpenAI()


def create_healing_prompt(
    schema_json: str,
    malformed_output: str,
    error_message: str
) -> str:
    """Generate a prompt for healing malformed output.
    
    Args:
        schema_json: JSON representation of the expected schema
        malformed_output: The output that failed validation
        error_message: The validation error message
        
    Returns:
        Prompt string for the healing LLM call
    """
    return f"""The following output failed validation.

Expected schema:
{schema_json}

Actual output that failed:
{malformed_output}

Validation error:
{error_message}

Return ONLY the corrected JSON that conforms to the schema.
Do not include explanations or additional text."""


def heal_output(
    raw_output: str,
    schema_class,
    max_attempts: int = 2
) -> ValidationResult:
    """Attempt to heal malformed output using LLM.
    
    Args:
        raw_output: Raw output from initial LLM call
        schema_class: Pydantic model class for validation
        max_attempts: Maximum number of healing attempts
        
    Returns:
        ValidationResult indicating success or failure
    """
    
    # First attempt: direct validation
    try:
        cleaned = extract_json(raw_output)
        data = json.loads(cleaned)
        schema_class(**data)
        
        return ValidationResult(
            success=True,
            parsed_data=data,
            validation_error=None,
            raw_output=raw_output
        )
    except (json.JSONDecodeError, ValidationError) as e:
        error_message = str(e)
        current_output = raw_output
    
    # Healing attempts
    for attempt in range(max_attempts):
        healing_prompt = create_healing_prompt(
            schema_json=schema_class.schema_json(indent=2),
            malformed_output=current_output,
            error_message=error_message
        )
        
        # Call LLM for repair
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": healing_prompt}],
            temperature=0.0  # Deterministic repairs
        )
        
        healed_output = response.choices[0].message.content
        
        # Try validating the healed output
        try:
            cleaned = extract_json(healed_output)
            data = json.loads(cleaned)
            schema_class(**data)
            
            # Success!
            return ValidationResult(
                success=True,
                parsed_data=data,
                validation_error=f"Healed on attempt {attempt + 1}",
                raw_output=healed_output
            )
        except (json.JSONDecodeError, ValidationError) as e:
            error_message = str(e)
            current_output = healed_output
    
    # All healing attempts failed
    return ValidationResult(
        success=False,
        parsed_data=None,
        validation_error=(
            f"Healing failed after {max_attempts} attempts: "
            f"{error_message}"
        ),
        raw_output=current_output
    )


def main():
    """Demonstrate LLM healing with intentionally malformed outputs."""
    
    # Example malformed outputs for testing
    test_cases = [
        # Missing quotes around value
        '''{
    "sentiment": positive,
    "confidence": 0.85,
    "key_themes": ["quality", "price"],
    "recommendations": ["Improve documentation"]
}''',
        # Missing required field
        '''{
    "sentiment": "negative",
    "confidence": 0.9,
    "key_themes": ["durability"]
}''',
        # Wrong type for confidence
        '''{
    "sentiment": "neutral",
    "confidence": "medium",
    "key_themes": ["shipping", "packaging"],
    "recommendations": ["Faster delivery"]
}'''
    ]
    
    print("Testing LLM Healing\n")
    print("=" * 70)
    
    results = []
    for i, malformed in enumerate(test_cases, 1):
        print(f"\n[Test Case {i}]")
        print("Malformed input:")
        print(malformed[:60] + "...")
        
        result = heal_output(malformed, ReviewAnalysis)
        results.append(result)
        
        if result.success:
            status = result.validation_error or "Valid initially"
            print(f"✓ {status}")
            print(f"  Sentiment: {result.parsed_data['sentiment']}")
        else:
            print(f"✗ {result.validation_error}")


if __name__ == "__main__":
    main()
