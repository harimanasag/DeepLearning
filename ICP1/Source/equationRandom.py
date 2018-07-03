import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Creating Session
session = tf.Session()

# Creating Random Matrix
matrix1 = tf.random_uniform([2, 3], 0, 10, dtype=tf.int32)
print("Matrix A : \n", session.run(matrix1))

# Creating Random Matrix
matrix2 = tf.random_uniform([2, 3], 0, 10, dtype=tf.int32)

print("Matrix B : \n", session.run(matrix2))

# Creating Random Matrix
matrix3 = tf.random_uniform([2, 3], 0, 10, dtype=tf.int32)

print("Matrix C : \n", session.run(matrix3))

# a ^ 2
matrixA2 = tf.pow(matrix1, 2)

# a ^ 2 + b
matrixAB = tf.add(matrixA2, matrix2)

# (a ^ 2 + b ) * c
matrixABC = tf.multiply(matrixAB, matrix3)

print("(a ^ 2 + b ) * c Result : \n ", session.run(matrixABC))

session.close()
