import CoreML

public enum GELUApproximation {
    case fast
    case precise
    case tanh
}

// Ref: https://github.com/ml-explore/mlx-swift/blob/86ad75ab1ee96cd70325732b37cd830f87d7e43f/Source/MLXNN/Activations.swift#L659
@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
public func gelu(_ x: MLTensor, approximation: GELUApproximation? = nil) -> MLTensor {
    switch approximation {
    case .none:
        return x * (1 + erf(x / sqrt(2 as Float))) / 2
    case .fast:
        return x * sigmoid(1.702 * x)
    case .precise, .tanh:
        return 0.5 * x * (1 + (sqrt(2 / Float.pi) * (x + 0.044715 * x.pow(3))).tanh())
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
public func sigmoid(_ x: MLTensor) -> MLTensor {
    1 / (1 + (-x).exp())
}

// Ref: https://en.wikipedia.org/wiki/Error_function#Numerical_approximations
@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
public func erf(_ x: MLTensor) -> MLTensor {
    let a1: Float = 0.254829592
    let a2: Float = -0.284496736
    let a3: Float = 1.421413741
    let a4: Float = -1.453152027
    let a5: Float = 1.061405429
    let p: Float = 0.3275911

    let sign = x.sign()
    let x = x.abs()

    let t = 1 / (1 + p * x)
    let y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp()

    return sign * y
}
