# coding: cp936
import sys
"""https://pypi.python.org/pypi/munkres/"""
"""二分图的匹配的穷举法示例"""
matrix = [[5, 9, 1],
          [10, 3, 2],
          [8, 7, 4]]
		  
		  
def permute(a, results):
    if len(a) == 1:
        results.insert(len(results), a)

    else:
        for i in range(0, len(a)):
            element = a[i]
            a_copy = [a[j] for j in range(0, len(a)) if j != i]
            subresults = []
            permute(a_copy, subresults)
            for subresult in subresults:
                result = [element] + subresult
                results.insert(len(results), result)

results = []
permute(range(len(matrix)), results) # [0, 1, 2] for a 3x3 matrix

n = len(matrix)
minval = sys.maxsize
for row in range(n):
    cost = 0
    for col in range(n):
        cost += matrix[row][col]
    minval = min(cost, minval)

print minval

""" HungarianAlgorithm
http://www.public.iastate.edu/~ddoty/HungarianAlgorithm.html 

The Munkres module provides an implementation of the Munkres algorithm 
(also called the Hungarian algorithm or the Kuhn-Munkres algorithm), 
useful for solving the Assignment Problem."""
from munkres import Munkres

m = Munkres()

from munkres import Munkres, print_matrix

matrix = [[5, 9, 1],
          [10, 3, 2],
          [8, 7, 4]]
m = Munkres()
indexes = m.compute(matrix)
print_matrix(matrix, msg='Lowest cost through this matrix:')
total = 0
for row, column in indexes:
    value = matrix[row][column]
    total += value
    print '(%d, %d) -> %d' % (row, column, value)
print 'total cost: %d' % total

# Output:
# Lowest cost through this matrix:
# [5, 9, 1]
# [10, 3, 2]
# [8, 7, 4]
# (0, 0) -> 5
# (1, 1) -> 3
# (2, 2) -> 4
# total cost=12
