"""Format evaluation utilities for LLM outputs."""
import json
import re
from typing import Optional, List
from dataclasses import dataclass
from pydantic import ValidationError


@dataclass
class ValidationResult:
    """Result of validating a single output."""
    success: bool
    parsed_data: Optional[dict]
    validation_error: Optional[str]
    raw_output: str


def extract_json(text: str) -> str:
    """Extract JSON from common LLM formatting patterns.
    
    Args:
        text: Raw LLM output that may contain JSON
        
    Returns:
        Extracted JSON string
    """
    # Remove markdown code fences
    pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    
    # Try to find JSON object boundaries
    brace_match = re.search(r'\{[\s\S]*\}', text)
    if brace_match:
        return brace_match.group(0)
    
    return text


def validate_output(raw_output: str, schema_class) -> ValidationResult:
    """Validate LLM output against a Pydantic schema.
    
    Args:
        raw_output: Raw text output from LLM
        schema_class: Pydantic model class to validate against
        
    Returns:
        ValidationResult indicating success or failure with details
    """
    try:
        # Extract and parse JSON
        cleaned = extract_json(raw_output)
        data = json.loads(cleaned)
        
        # Validate against schema
        schema_class(**data)
        
        return ValidationResult(
            success=True,
            parsed_data=data,
            validation_error=None,
            raw_output=raw_output
        )
    
    except json.JSONDecodeError as e:
        return ValidationResult(
            success=False,
            parsed_data=None,
            validation_error=f"JSON parsing error: {str(e)}",
            raw_output=raw_output
        )
    
    except ValidationError as e:
        return ValidationResult(
            success=False,
            parsed_data=None,
            validation_error=f"Schema validation error: {str(e)}",
            raw_output=raw_output
        )


@dataclass
class ConformanceMetrics:
    """Aggregate metrics for a set of validations."""
    total: int
    successful: int
    json_parse_failures: int
    schema_validation_failures: int
    
    @property
    def conformance_rate(self) -> float:
        """Percentage of outputs that fully conform to schema."""
        return self.successful / self.total if self.total > 0 else 0.0
    
    @property
    def parse_rate(self) -> float:
        """Percentage that are valid JSON."""
        parsed = self.total - self.json_parse_failures
        return parsed / self.total if self.total > 0 else 0.0


def compute_conformance(results: List[ValidationResult]) -> ConformanceMetrics:
    """Compute conformance metrics from validation results.
    
    Args:
        results: List of ValidationResult objects
        
    Returns:
        ConformanceMetrics with aggregate statistics
    """
    json_failures = sum(
        1 for r in results 
        if 'JSON parsing error' in (r.validation_error or '')
    )
    schema_failures = sum(
        1 for r in results 
        if 'Schema validation error' in (r.validation_error or '')
    )
    successful = sum(1 for r in results if r.success)
    
    return ConformanceMetrics(
        total=len(results),
        successful=successful,
        json_parse_failures=json_failures,
        schema_validation_failures=schema_failures
    )
