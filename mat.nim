# Support definitions for nim-nnfsix
# License: MIT

## This module includes basic definitions of a matrix and a dense layer as well
## as needed operations for nim-nnfsix.
##
## This module was created as a training and learning exercise and it is not
## meant to be used for anything else apart from learning Nim and/or deep
## learning exercises. In particular, the code **lacks** checks and tests.
##
## The module only depends on Nim's stdlib.
import random

type
  Matrix*[R, C: static[int]] = array[R, array[C, float]]
    ## Matrix, an array of an array of floats.
    ## Initialize with [[]].
    ## Vectors are Matrix[1, N] or Matrix[M, 1]

  DenseLayer*[M, N: static[int]] = object
    ## Dense layer, with M inputs and N neurons.
    ## Only holds data: weights, biases, output.
    weights: Matrix[M, N]
    biases: Matrix[1, N]
    output: Matrix[M, M]

iterator iterMat*[M, N](a: Matrix[M, N]): float =
  ## Yield all elements of a Matrix[M, N]
  for row in a.low..a.high:
    for col in a[0].low..a[0].high:
      yield a[row][col]

proc `+`*[R, C, H](a: Matrix[R, C], b: Matrix[H, C]): Matrix[R, C] =
  for row in result.low..result.high:
    for col in result[0].low..result[0].high:
      for elem in 0..<H:
        result[row][col] = a[row][col] + b[elem][col]

proc `+`*[R, C](a: Matrix[R, C], b: float): Matrix[R, C] =
  for row in result.low..result.high:
    for col in result[0].low..result[0].high:
      result[row][col] = a[row][col] + b

proc `$`*[R, C](a: Matrix[R, C]): string =
  result = "Matrix " & $R  & $"x" & $C & "\n"
  # for row in 0 ..< a.R
  for row in a.low..a.high:
    for col in low(a[0])..high(a[0]):
      result.add($a[row][col] & "  ")
    result.add("\n")

proc transpose*[R, C](a: Matrix[R, C]): Matrix[C, R] =
  for i in result.low..result.high:
    for j in result[0].low..result[0].high:
      result[i][j] = a[j][i]

template `T`*[R, C](a: Matrix[R, C]): Matrix[C, R] = transpose(a)

proc shape*(a: Matrix): tuple[rows, columns: int] = (a.R, a.C)

proc size*(a: Matrix): int = a.R * a.C

proc `@`*[R, C, H](a: Matrix[R, C], b: Matrix[C, H]): Matrix[R, H] =
  ## Matrix multiplication.
  for row in result.low..result.high:
    for col in result[0].low..result[0].high:
      for elem in a[0].low..a[0].high:
        result[row][col] += a[row][elem] * b[elem][col]


proc maximum*[M, N](thr: float, m: Matrix[M, N]): Matrix[M, N] =
  ## Behaves like numpy.maximum, but thr is a float.
  ## Used as ReLU with `forward <#forward,Matrix[M,N],DenseLayer[N,W]>`_.
  for row in result.low..result.high:
    for col in result[0].low..result[0].high:
      if m[row][ col] < thr:
        result[row][ col] = thr
      else:
        result[row][ col] = m[row][ col]

proc zeroMatrix(rows, columns: static[int]): Matrix[rows, columns] =
  for row in result.low..result.high:
    for col in result[0].low..result[0].high:
      result[row][col] = 0

proc zeroVector*(size: static[int]): Matrix[1, size] =
  zeroMatrix(1, size)

proc zeroVectorCol*(size: static[int]): Matrix[size, 1] =
  zeroVector(size).transpose

proc newRandomMatrix*(rows, columns: static[int], r: int): Matrix[rows, columns] =
  ## Create a rows x columns random `Matrix <#Matrix>`_.
  ## ``r`` is used as seed for the random generator.
  var seed = initRand(r)
  for row in result.low..result.high:
    for col in result[0].low..result[0].high:
      result[row][col] = gauss(r=seed, mu=0.0, sigma=0.01)

proc newDenseLayer*[M, N](w: Matrix[M, N], b: Matrix[1, N]): DenseLayer[M, N] =
  ## Create a `dense layer <#DenseLayer>`_ with ``num_inputs`` and
  ## ``num_neurons``.
  result.weights = w
  result.biases = b

proc newRandomDenseLayer*(num_inputs, num_neurons: static[int], r: int):
  DenseLayer[num_inputs, num_neurons] =
    ## Create a random `dense layer <#DenseLayer>`_ with ``num_inputs`` and
    ## ``num_neurons``.
    ##
    ## ``r``: seed for the random generator.
    result.weights = newRandomMatrix(num_inputs, num_neurons, r)
    result.biases = zeroMatrix(1, num_neurons)

proc forward*[M, N, W](inputs: Matrix[M, N], layer: DenseLayer[N, W]): Matrix[M, W] =
  ## Inputs and weights matrix multiplication plus bias
  result = (inputs @ layer.weights) + layer.biases
