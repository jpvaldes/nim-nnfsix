import mat

proc main =
  let
    X: Matrix[3, 4] = [[1.0, 2.0, 3.0, 2.5],
                       [2.0, 5.0, -1.0, 2.0],
                       [-1.5, 2.7, 3.3, -0.8]]
  let layer1 = newRandomDenseLayer(4, 5, 42)
  let layer2 = newRandomDenseLayer(5, 2, 42)
  echo forward(maximum(0, forward(X, layer1)), layer2)

when isMainModule:
  main()
