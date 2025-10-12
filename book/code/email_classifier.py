"""Email priority classifier using LLM."""
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal
import json


client = OpenAI()

# start snippet schema
class EmailClassification(BaseModel):
    """Classification result for an email."""
    priority: Literal["urgent", "normal", "low"]
    reasoning: str = Field(description="Brief explanation")
# end snippet schema

# start snippet classify
def classify_email(subject: str, body: str) -> EmailClassification:
    """Classify email priority using LLM.
    
    Args:
        subject: Email subject line
        body: Email body text
        
    Returns:
        EmailClassification with priority and reasoning
    """
    prompt = f"""Classify this email's priority level.

Categories:
- urgent: Requires immediate attention (outages, security, 
  data loss)
- normal: Standard request, handle within 24 hours
- low: Enhancement request, documentation, general inquiry

Provide:
1. priority: one of the three categories
2. reasoning: brief explanation (1 sentence)

Subject: {subject}
Body: {body}

Return JSON with priority and reasoning fields."""
    
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
    return EmailClassification(**data)
# end snippet classify


def main():
    """Run email classification examples."""
    
    # Example emails with ground truth labels
    test_emails = [
        {
            "subject": "Production database is down",
            "body": "Cannot connect to DB. All services failing.",
            "expected": "urgent"
        },
        {
            "subject": "Question about API rate limits",
            "body": "What are the current rate limits for the API?",
            "expected": "low"
        },
        {
            "subject": "Payment failed for invoice #1234",
            "body": "Tried to pay but card was declined. Need help.",
            "expected": "normal"
        },
        {
            "subject": "Security vulnerability discovered",
            "body": "Found SQL injection in login form.",
            "expected": "urgent"
        },
        {
            "subject": "Feature request: dark mode",
            "body": "Would be nice to have a dark mode option.",
            "expected": "low"
        },
        {
            "subject": "Account access issues",
            "body": "Can't log in. Password reset not working.",
            "expected": "normal"
        }
    ]
    
    print("Email Priority Classification\n")
    print("=" * 60)
    
    correct = 0
    total = len(test_emails)
    
    for i, email in enumerate(test_emails, 1):
        result = classify_email(email["subject"], email["body"])
        is_correct = result.priority == email["expected"]
        if is_correct:
            correct += 1
        
        print(f"\n[Email {i}]")
        print(f"Subject: {email['subject']}")
        print(f"Predicted: {result.priority}")
        print(f"Expected: {email['expected']}")
        print(f"Correct: {'yes' if is_correct else 'no'}")
        print(f"Reasoning: {result.reasoning}")
    
    # Summary
    accuracy = correct / total
    print("\n" + "=" * 60)
    print(f"\nOverall Accuracy: {accuracy:.1%} ({correct}/{total})")


if __name__ == "__main__":
    main()
