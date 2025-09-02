import numpy as np
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import string


def get_transpose(matrix: list[list]) -> list[list]:

    rows = len(matrix)
    cols = len(matrix[0])

    # new matrix with swapped dimensions 
    transpose_matrix = [[0 for _ in range(rows)] for _ in range(cols)]
    
    for r in range(rows):
        for c in range(cols):

            transpose_matrix[c][r] = matrix[r][c]

    
    return transpose_matrix



def merge_elements(sequences):

    # takes a list of sequnces (can be list or tuples) and combines them into one list
    return [item for sublist in sequences for item in sublist]


def remove_tail(items):

    if not items:
        return items
    
    return items[:-1]
    

def select_alternates(seq):

    return seq[::2]


def convert_to_mixed(id):

    # remove all the leading and trailing underscores
    cleaned = id.strip('_')

    # return if its empty after 
    if not cleaned: 
        return cleaned

    words = cleaned.split('_')

    # filter out if its empty string
    words = [word for word in words if word]

    # if its only one word, return it in lower case
    if len(words) == 1:
        return words[0].lower() 

    result = words[0].lower()

    for word in words[1:]:  # Skip first word, start from second
        result += word.capitalize()

    return result
    

 
# Generators 

def prefixes(seq):

    res = []
    yield res

    for i in seq: 
        res = res + [i]

        yield res


def suffixes(seq):

    res = list(seq)
    yield res

    for i in range(len(seq)):

        res = res[1:]
        yield res



def segments(seq):

    for i in range(len(seq)):
        for j in range(i + 1, len(seq) + 1):
            yield seq[i:j]
    



# The class initialization syntax in Python is as follows:
#
# class ClassName:
#     def __init__(self, arg1, arg2, ...):
#         self.arg1 = arg1
#         self.arg2 = arg2
#
# Example:
#
# class MyClass:
#     def __init__(self, value):
#         self.value = value


class Polynomial:
    
    def __init__(self, polynomial):

        self._polynomial = tuple(polynomial)

    def get_polynomial(self):
        return self._polynomial
    

    def __neg__(self):

        negated_terms = []
        
        for coefficient, exponent in self._polynomial:
            negated_terms.append((-coefficient, exponent))  
        
        return Polynomial(negated_terms)



    def __add__(self, other):

        self_terms = list(self.get_polynomial())
        other_terms = list(other.get_polynomial())

        combined = self_terms + other_terms

        return Polynomial(combined)
    

    def __sub__(self, other):
      
        return self + (-other)
    

    def __mul__(self, other):

        terms = []

        self_terms = list(self.get_polynomial())
        other_terms = list(other.get_polynomial())
        
        for coefficient, exponent in self_terms:
            for c, e in other_terms: 

                new_coeff = coefficient * c
                new_exp = exponent + e
                
                terms.append((new_coeff, new_exp))

        return Polynomial(terms)
    

    def __call__(self, x):

        return sum(coefficient * (x ** exponent) for coefficient, exponent in self._polynomial)



    def simplify(self):
    
        terms = list(self._polynomial)

        # Sort before combining to find matching exponents
        terms.sort(key=lambda term: term[1], reverse=True)
        
        simplified = []

        i = 0

        while i < len(terms):
            coefficient, exponent = terms[i]

            # if the next is the same in the sorted set
            j = i + 1  

            while j < len(terms) and terms[j][1] == exponent:
                #combine
                coefficient += terms[j][0] 
                j += 1

            if coefficient != 0:
                simplified.append((coefficient, exponent))

            i = j

        # Modify in place by updating self._polynomial
        self._polynomial = tuple(simplified)

    
    def __str__(self):

        if not self._polynomial:
            return "0"
        
        terms = list(self._polynomial)
        res = []

        for i, (coefficient, exponent) in enumerate(terms):

            if coefficient >= 0:
                sign = "+" if i > 0 else ""
            else:
                sign = "-"
                coefficient = abs(coefficient)

            if coefficient == 0:
                term = "0x" if exponent == 1 else f"0x^{exponent}" if exponent != 0 else "0"
            elif exponent == 0:
                term = str(coefficient)
            elif exponent == 1:
                
                if coefficient == 1:
                    term = "x"
                else: 
                    term = f"{coefficient}x"
            else:

                if coefficient == 1: 
                    term = f"x^{exponent}x"
                else: 
                    term = f"{coefficient}^{exponent}"

            if i == 0: 
                res.append(sign + term)
            else: 
                res.append(" " + sign + " " + term)

        return "".join(res)

            
def sort_array(list_of_matricies):

    all_values = []

    for matrix in list_of_matricies: 
        # flatten the stock and add it to the list
        all_values.extend(matrix.flatten())

    result = np.array(all_values)

    sorted = np.sort(result)[::-1]

    return sorted


def POS_tag(sentence):

    # Convert to lowercase
    sentence = sentence.lower()
    
    # Tokenize
    words = word_tokenize(sentence)
    
    # Get English stop words
    stop_words = set(stopwords.words('english'))
    
    # Remove stop words and punctuation
    filtered = []
    for word in words:
        word_clean = word.translate(str.maketrans('', '', string.punctuation))
        
        # Only add if it's not a stop word and not empty after removing punctuation
        if word_clean and word_clean not in stop_words:
            filtered.append(word_clean)
    
    pos_tags = pos_tag(filtered)
    
    return pos_tags

        
    



if __name__ == "__main__":
       
    """
    print("\nTesting segments:")
    print(list(segments([4, 5, 6])))  # [[4], [4, 5], [4, 5, 6], [5], [5, 6], [6]]
    print(list(segments("lmn")))       # ['l', 'lm', 'lmn', 'm', 'mn', 'n']
    """
    
    """
    print("\nTesting Polynomial negation:")
    f = Polynomial([(3, 2), (4, 1)])
    print("Original f:", f.get_polynomial())  # ((3, 2), (4, 1))
    
    g = -f
    print("Negated g:", g.get_polynomial())   # ((-3, 2), (-4, 1))
    
    h = -(-f)
    print("Double negated h:", h.get_polynomial())  # ((3, 2), (4, 1))
    

    print("\nTesting Polynomial addition:")
    a = Polynomial([(1, 2), (2, 0)])
    b = a + a
    print("a + a:", b.get_polynomial())  # ((1, 2), (2, 0), (1, 2), (2, 0))
    
    c = Polynomial([(3, 5), (4, 3)])
    d = a + c
    print("a + c:", d.get_polynomial())  # ((1, 2), (2, 0), (3, 5), (4, 3))
    
    print("\nTesting Polynomial subtraction:")
    e = a - a
    print("a - a:", e.get_polynomial())  # ((1, 2), (2, 0), (-1, 2), (-2, 0))
    
    f = a - c
    print("a - c:", f.get_polynomial())  # ((1, 2), (2, 0), (-3, 5), (-4, 3))
    
    print("\nTesting Polynomial multiplication:")
    g = Polynomial([(3, 2), (1, 0)])
    h = g * g
    print("g * g:", h.get_polynomial())  # ((9, 4), (3, 2), (3, 2), (1, 0))
    
    i = Polynomial([(2, 3), (4, 1)])
    j = g * i
    print("g * i:", j.get_polynomial())  # ((6, 5), (12, 3), (2, 3), (4, 1))
    
    print("\nTesting Polynomial evaluation:")
    a = Polynomial([(3, 2), (1, 0)])
    print("a(x) for x in range(4):", [a(x) for x in range(4)])  # [1, 4, 13, 28]
    
    b = -(a * a) + a
    print("b(x) for x in range(4):", [b(x) for x in range(4)])  # [0, -12, -156, -756]
    
    print("\nTesting Polynomial simplification:")
    # Create a polynomial with like terms that need combining
    messy = Polynomial([(3, 2), (1, 0), (2, 2), (-1, 0), (5, 3), (-2, 2)])
    print("Before simplification:", messy.get_polynomial())
    
    messy.simplify()  # Modifies in place
    print("After simplification:", messy.get_polynomial())
    
    print("\nTesting Polynomial string representation:")
    a = Polynomial([(1, 1), (1, 0)])
    print("a =", str(a))  # "x + 1"
    
    a_plus_a = a + a
    a_plus_a.simplify()
    print("a + a =", str(a_plus_a))  # "2x + 2"
    
    neg_a = -a
    print("-a =", str(neg_a))  # "-x - 1"
    
    neg_a_minus_a = -a - a
    neg_a_minus_a.simplify()
    print("-a - a =", str(neg_a_minus_a))  # "-2x - 2"
    
    a_times_a = a * a
    a_times_a.simplify()
    print("a * a =", str(a_times_a))  # "x^2 + 2x + 1"
    """

    """
    print("\nTesting matrix sorting (flatten and sort all values):")
    mat_a = np.array([[10, 20], [30, 40]])  # 2x2 matrix
    mat_b = np.array([[15, 25, 35], [45, 55, 65], [5, -5, -15]])  # 3x3 matrix
    mat_c = np.array([[1, 2], [3, 4]])  # 2x2 matrix
    
    print("Matrix A (2x2):")
    print(mat_a)
    print("Matrix B (3x3):")
    print(mat_b)
    print("Matrix C (2x2):")
    print(mat_c)
    
    result = sort_array([mat_a, mat_b, mat_c])
    print(f"\nAll values flattened and sorted (descending): {result}")
    print(f"Data type: {result.dtype}")
    print(f"Shape: {result.shape}")
    """

    print("\nTesting POS tagging:")
    sentence = "Python programming is fun and exciting!"
    pos_result = POS_tag(sentence)
    print(f"Sentence: '{sentence}'")
    print(f"POS tags: {pos_result}")

    




