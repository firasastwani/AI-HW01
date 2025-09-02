# AI Homework 01 - Python Programming Fundamentals

**Student:** Firas Astwani  
**Course:** AI Programming  
**Assignment:** Homework 01

## Overview

This homework assignment demonstrates proficiency in Python programming fundamentals, including data structures, algorithms, object-oriented programming, and natural language processing using NLTK.

## Features Implemented

### 1. Matrix Operations

- **`get_transpose(matrix)`**: Computes the transpose of a 2D matrix
- **`sort_array(list_of_matrices)`**: Flattens multiple matrices and sorts all values in descending order

### 2. List Processing Functions

- **`merge_elements(sequences)`**: Combines multiple sequences (lists/tuples) into a single flattened list
- **`remove_tail(items)`**: Removes the last element from a list
- **`select_alternates(seq)`**: Returns every other element from a sequence (even indices)
- **`convert_to_mixed(id)`**: Converts underscore-separated identifiers to camelCase

### 3. Generator Functions

- **`prefixes(seq)`**: Generates all prefixes of a sequence
- **`suffixes(seq)`**: Generates all suffixes of a sequence
- **`segments(seq)`**: Generates all contiguous subsequences of a sequence

### 4. Object-Oriented Programming

- **`Polynomial` class**: Implements polynomial arithmetic with the following methods:
  - `__init__(polynomial)`: Initialize with list of (coefficient, exponent) tuples
  - `__neg__()`: Negation operator
  - `__add__(other)`: Addition operator
  - `__sub__(other)`: Subtraction operator
  - `__mul__(other)`: Multiplication operator
  - `__call__(x)`: Evaluation at point x
  - `simplify()`: Combines like terms
  - `__str__()`: String representation

### 5. Natural Language Processing

- **`POS_tag(sentence)`**: Part-of-Speech tagging function that:
  - Converts text to lowercase
  - Tokenizes sentences into words
  - Removes stop words and punctuation
  - Returns POS tags as list of tuples

## Dependencies

```bash
pip install numpy nltk
```

### NLTK Data Requirements

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
```

## Installation & Setup

1. **Clone or download the repository**
2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install numpy nltk
   ```
4. **Download NLTK data:**
   ```bash
   python3 -c "import nltk; nltk.download('stopwords'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"
   ```

## Usage

### Running the Script

```bash
python3 Astwani_Firas_HW01.py
```

### Example Usage

#### Matrix Operations

```python
import numpy as np

# Matrix transpose
matrix = [[1, 2, 3], [4, 5, 6]]
transposed = get_transpose(matrix)
print(transposed)  # [[1, 4], [2, 5], [3, 6]]

# Matrix sorting
mat_a = np.array([[10, 20], [30, 40]])
mat_b = np.array([[15, 25, 35], [45, 55, 65]])
result = sort_array([mat_a, mat_b])
print(result)  # [65, 55, 45, 40, 35, 30, 25, 20, 15, 10]
```

#### Polynomial Operations

```python
# Create polynomials
p1 = Polynomial([(3, 2), (1, 0)])  # 3x² + 1
p2 = Polynomial([(2, 1), (1, 0)])  # 2x + 1

# Arithmetic operations
sum_poly = p1 + p2
product = p1 * p2
evaluation = p1(2)  # Evaluate at x=2

print(f"p1 + p2 = {sum_poly}")
print(f"p1 * p2 = {product}")
print(f"p1(2) = {evaluation}")
```

#### POS Tagging

```python
sentence = "Python programming is fun and exciting!"
pos_tags = POS_tag(sentence)
print(pos_tags)  # [('python', 'NN'), ('programming', 'VBG'), ('fun', 'NN'), ('exciting', 'VBG')]
```

## Test Cases

The script includes comprehensive test cases for all functions:

- **Matrix sorting**: Tests with 2x2 and 3x3 matrices
- **Polynomial operations**: Tests negation, addition, subtraction, multiplication, evaluation, and simplification
- **POS tagging**: Tests with sample sentences
- **Generator functions**: Tests prefixes, suffixes, and segments

## File Structure

```
AI_HW/
├── Astwani_Firas_HW01.py    # Main Python script
├── README.md                # This documentation
└── venv/                    # Virtual environment (created during setup)
```

## Key Features

- **Error Handling**: Functions handle edge cases (empty lists, single elements)
- **Type Safety**: Uses type hints where appropriate
- **Memory Efficiency**: Generator functions for large sequences
- **Mathematical Accuracy**: Proper polynomial arithmetic with like-term combination
- **NLP Integration**: Professional-grade text processing with NLTK

## Technical Notes

- **Matrix Operations**: Uses NumPy for efficient array operations
- **Polynomial Representation**: Uses tuples of (coefficient, exponent) pairs
- **Text Processing**: Implements proper tokenization, stop word removal, and punctuation handling
- **Generator Pattern**: Memory-efficient sequence processing for large datasets

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError for numpy/nltk**: Ensure virtual environment is activated and packages are installed
2. **NLTK data not found**: Run the NLTK download commands listed in setup
3. **Permission errors**: Use virtual environment instead of system-wide installation

### Environment Issues

If you encounter externally-managed-environment errors:

```bash
# Use virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate
pip install numpy nltk
```

## Author

**Firas Astwani**  
AI Programming Course  
Homework Assignment 01

---

_This assignment demonstrates comprehensive Python programming skills including data structures, algorithms, OOP, and NLP processing._
