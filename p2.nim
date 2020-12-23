import mat

# output layer neuron
proc main =
  let
    inputs: Matrix[1, 4] = [[1.0, 2.0, 3.0, 2.5]]
    weights: Matrix[3, 4] = [[0.2, 0.8, -0.5, 1.0],
                             [0.5, -0.91, 0.26, -0.5],
                             [-0.26, -0.27, 0.17, 0.87]]
    bias: Matrix[1, 3] = [[2.0, 3.0, 0.5]]

  let output = (inputs @ weights.transpose) + bias

  echo output

when isMainModule:
  main()
