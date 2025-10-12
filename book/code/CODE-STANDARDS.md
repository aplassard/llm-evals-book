# Code Standards for LLM Evals Book

This document outlines the conventions and standards for code examples in this book.

## Directory Organization

### Chapter-Specific Code
- Each chapter's code lives in its own subdirectory: `code/chapter-N/`
- Example: Chapter 4 code is in `code/chapter-4/`
- This keeps code organized and makes it clear which examples belong to which chapter

### File Naming
- Use descriptive, lowercase names with underscores: `format_evaluation.py`, `email_classifier.py`
- Name files after their primary purpose, not their chapter number
- Keep names concise but meaningful

## Code in the Book vs. Standalone Scripts

### Two Audiences, Two Versions

**In the Book (via snippets):**
- Show only the conceptual implementation
- Focus on teaching the pattern or technique
- Exclude test harnesses, `main()` functions, and demo code
- Keep examples focused and minimal

**In Code Directory (full files):**
- Include complete, runnable examples
- Add `main()` functions with test data and demonstrations
- Make scripts executable and useful for experimentation
- Provide real examples readers can run and modify

### Using Snippet Markers

Wrap the code you want in the book with snippet markers:

```python
# start snippet core_function
def my_important_function(input: str) -> dict:
    """This appears in the book."""
    # Implementation details...
    return result
# end snippet core_function


def main():
    """This does NOT appear in the book."""
    # Test data, loops, print statements, etc.
    test_cases = [...]
    for case in test_cases:
        result = my_important_function(case)
        print(result)


if __name__ == "__main__":
    main()
```

**Key principle:** The snippet ends BEFORE the `main()` function and any test/demo code.

## Code Style Principles

### 1. Concise and Focused
- Every line should serve the teaching purpose
- Remove unnecessary complexity
- Don't show multiple approaches unless comparing them
- Avoid "enterprise" patterns (excessive abstraction, framework overhead)

### 2. Explicit to the Problem
- Make the purpose immediately clear
- Use descriptive variable and function names
- Include docstrings that explain what, not how
- Show the pattern being taught, not everything around it

### 3. Production-Practical
- Code should reflect real-world usage
- Include error handling where it matters
- Show proper type hints
- Demonstrate good practices without being pedantic

### 4. Self-Contained Examples
- Each example should work independently
- Minimize dependencies between examples
- When files do depend on each other (like `format_evaluation.py` importing from `product_review_schema.py`), keep them in the same chapter directory

## What NOT to Include in Book Snippets

❌ **Don't include:**
- `main()` functions
- `if __name__ == "__main__":` blocks
- Test data definitions
- Print statements for demo purposes
- Progress indicators and status messages
- Result formatting and reporting
- Multiple examples of the same pattern
- Commented-out alternative approaches

✅ **Do include:**
- Core function implementations
- Schema definitions (Pydantic models)
- Essential helper functions
- Docstrings that explain purpose
- Type hints
- Error handling that's part of the pattern

## Referencing Code in Text

### When to Include Code
Use code snippets in the book when:
- Teaching a new pattern or technique
- Showing the core implementation of a concept
- Demonstrating a specific approach

### When to Reference Code
Just reference the file (don't include snippets) when:
- The example is primarily for readers to run themselves
- The code is mostly boilerplate or similar to previous examples
- The implementation is straightforward and self-explanatory

**Example references:**
- "The code in `code/chapter-4/format_evaluation.py` demonstrates this pattern."
- "You can run the full example with `python code/chapter-4/email_classifier.py`"
- "See `code/chapter-4/classification_metrics.py` for the complete implementation."

## Including Code in Quarto

### Snippet Syntax
```markdown
```{.python include="code/chapter-4/filename.py" snippet="snippet_name"}
```
```

### Path Convention
- Always use relative paths from the book root: `code/chapter-N/`
- Be consistent with path format across all chapters

### Multiple Snippets Per File
You can have multiple snippets in a single file:

```python
# start snippet schema
class MySchema(BaseModel):
    field: str
# end snippet schema

# start snippet process
def process_data(data: str) -> MySchema:
    # Implementation
    return result
# end snippet process
```

Then reference them separately:
```markdown
First, define the schema:
```{.python include="code/chapter-4/example.py" snippet="schema"}
```

Then process the data:
```{.python include="code/chapter-4/example.py" snippet="process"}
```
```

## Code Documentation

### Docstrings
- Include docstrings for all functions shown in the book
- Use the Google/NumPy style with Args, Returns sections
- Keep them concise but informative
- Focus on what the function does, not implementation details

### Comments
- Use comments sparingly in book examples
- Prefer clear code over explanatory comments
- Comments should explain *why*, not *what*
- Remove debug comments and TODOs before committing

## Testing and Validation

### Before Committing
1. ✅ Verify all standalone scripts run without errors
2. ✅ Check that snippet markers are correctly placed
3. ✅ Ensure imports work within chapter directories
4. ✅ Test that Quarto renders all code blocks correctly
5. ✅ Confirm no `main()` functions appear in rendered book

### After Changes
1. Rebuild the book: `quarto render`
2. Run affected code examples
3. Check for broken references or imports

## Common Patterns

### Example Structure
```python
"""Brief description of what this file demonstrates."""
# Imports
from typing import List
from pydantic import BaseModel

# start snippet example_name
# Schema definitions, core functions, etc.
# Everything that should appear in the book

class MySchema(BaseModel):
    """Schema for the example."""
    field: str

def core_function(input: str) -> MySchema:
    """Main function demonstrating the concept."""
    # Implementation
    return result
# end snippet example_name


# Everything below here is for standalone execution only
def main():
    """Demonstrate the code with test cases."""
    examples = [...]
    for example in examples:
        result = core_function(example)
        print(f"Result: {result}")


if __name__ == "__main__":
    main()
```

## Migration Guide

When adding new code or updating existing code:

1. **Create chapter directory** if it doesn't exist: `code/chapter-N/`
2. **Write full example** with `main()` function and test cases
3. **Add snippet markers** around the code that should appear in book
4. **Update chapter text** to reference `code/chapter-N/filename.py`
5. **Test standalone** by running the script
6. **Build book** to verify snippet inclusion works
7. **Review rendered output** to ensure clean, focused code examples

## Philosophy

The goal is to teach concepts effectively while providing practical, runnable examples. The book should be clean and focused; the code directory should be functional and exploratory. By separating these concerns with snippet markers, we serve both audiences well.

**Remember:** If it helps teach the concept, put it in the book. If it helps readers experiment and learn by doing, put it in the code directory. Often, you need both versions.