# single neuron with 3 inputs

import mat

proc main =
  let
    inputs: Matrix[1, 3] = [[1.2, 5.1, 2.1]]
    weights: Matrix[1, 3] = [[3.1, 2.1, 8.7]]
    bias = 3.0

  let output = (inputs @ weights.transpose) + bias

  echo output

when isMainModule:
  main()
