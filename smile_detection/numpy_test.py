import numpy as np
from scipy import linalg as la
import unittest
from homework1_sbuasai2 import *

class hw1_test(unittest.TestCase):
  def test_prob1_1d(self):
    A = np.array([1,1,1])
    B = np.array([2,2,2])
    expected = np.array([3,3,3])
    actual = problem1(A,B)
    np.testing.assert_array_equal(actual, expected)

  def test_prob1_2d(self):
    A = np.array([[1,1,1],[2,2,2]])
    B = np.array([[3,3,3],[4,4,4]])
    expected = np.array([[4,4,4],[6,6,6]])
    actual = problem1(A,B)
    np.testing.assert_array_equal(actual, expected)

  def test_prob2_1d(self):
    A = np.array([1,1,1])
    B = np.array([2,2,2])
    C = np.array([1])
    expected = 5
    actual = problem2(A,B,C)
    np.testing.assert_array_equal(actual, expected)

  def test_prob2_2d(self):
    A = np.array([[1,1],[2,2]])
    B = np.array([[3,3],[4,4]])
    C = np.array([[1,1],[2,2]])
    expected = np.array([[6,6],[12,12]])
    actual = problem2(A, B, C)
    np.testing.assert_array_almost_equal(actual, expected)

  def test_prob3(self):
    A = np.array([[1,1,1],[2,2,2]])
    B = np.array([[3,3,3],[4,4,4]])
    C = np.array([[1,2],[1,2],[1,2]])
    expected = np.array([[4,4,4],[10,10,10]])
    actual = problem3(A, B, C)
    np.testing.assert_array_equal(actual, expected)

  def test_prob4_1d(self):
    x = np.array([1,1])
    S = np.array([[2,2],[3,3]])
    y = np.array([4,4])
    expected = 40
    actual = problem4(x, S, y)
    np.testing.assert_array_equal(actual, expected)

  def test_prob4_2d(self):
    x = np.array([[1,1,1],[2,2,2]])
    S = np.array([[2,2,2],[3,3,3]])
    y = np.array([[3,3],[4,4],[5,5]])
    expected = np.array([[96,96],[96,96],[96,96]])
    actual = problem4(x, S, y)
    np.testing.assert_array_equal(actual, expected)

  def test_prob5(self):
    A = np.array([[1,1,1],[2,2,2],[3,3,3]])
    expected = np.array([[1],[1],[1]])
    actual = problem5(A)
    np.testing.assert_array_equal(actual, expected)

  def test_prob6(self):
    A = np.array([[1,1,1],[2,2,2],[3,3,3]])
    expected = np.array([[0,1,1],[2,0,2],[3,3,0]])
    actual = problem6(A)
    np.testing.assert_array_equal(actual, expected)

  def test_prob7(self):
    A = np.array([[1,1],[2,2]])
    alpha = 2
    expected = np.array([[3,1],[2,4]])
    actual = problem7(A, alpha)
    np.testing.assert_array_equal(actual, expected)

  def test_prob8(self):
    A = np.array([[1,1,1],[2,2,2],[3,3,3]])
    i = 1
    j = 2
    expected = 3
    actual = problem8(A, i, j)
    np.testing.assert_array_equal(actual, expected)

  def test_prob9(self):
    A = np.array([[1,1,1],[2,2,2],[3,3,3]])
    i = 0
    expected = 3
    actual = problem9(A, i)
    np.testing.assert_array_equal(actual, expected)

  def test_prob10(self):
    A = np.array([[1,2,3],[4,5,6],[7,8,9]])
    c = 3
    d = 8
    expected = sum([3,4,5,6,7,8])/6
    actual = problem10(A, c, d)
    np.testing.assert_array_equal(actual, expected)

  def test_prob11(self):
    A = np.diag([4,3,2,1])
    k = 3
    expected = np.array([[0,0,1],[0,1,0],[1,0,0],[0,0,0]])
    actual = problem11(A, k)
    np.testing.assert_array_equal(actual, expected)

  def test_prob12(self):
    A = np.array([[1,2],[3,5]])
    x = np.array([1,2])
    expected = np.array([-1.,1.])
    actual = problem12(A,x)
    np.testing.assert_array_almost_equal(actual, expected)

  def test_prob13(self):
    x = np.array([1,2,3])
    k = 4
    expected = np.array([[1,1,1,1],[2,2,2,2],[3,3,3,3]])
    actual = problem13(x, k)
    np.testing.assert_array_almost_equal(actual, expected)

  def test_prob14(self):
    A = np.array([[0,1],[2,3]])
    e0 = np.array([[0,1],[2,3]])
    e1 = np.array([[0,1],[3,2]])
    e2 = np.array([[1,0],[2,3]])
    e3 = np.array([[1,0],[3,2]])
    actual = problem14(A)
    self.assertTrue(np.array_equal(actual, e0) or np.array_equal(actual, e1) or np.array_equal(actual, e2) or np.array_equal(actual, e3))
