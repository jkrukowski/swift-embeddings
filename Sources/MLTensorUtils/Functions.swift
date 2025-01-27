import CoreML
import Foundation

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
public func norm(_ x: MLTensor, alongAxes: Int = 1, keepRank: Bool = false) -> MLTensor {
    x.squared().sum(alongAxes: alongAxes, keepRank: keepRank).squareRoot()
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
public func cosineSimilarity(_ x: MLTensor, _ y: MLTensor, alongAxes: Int = 1) -> MLTensor {
    let normX = norm(x, alongAxes: alongAxes)
    let normY = norm(y, alongAxes: alongAxes)
    return x.matmul(y.transposed()) / (normX * normY)
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
public func dotProduct(_ x: MLTensor, _ y: MLTensor) -> MLTensor {
    x.transposed().matmul(y)
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
public func cosineDistance(_ x: MLTensor, _ y: MLTensor, alongAxes: Int = 1) -> MLTensor {
    1 - cosineSimilarity(x, y, alongAxes: alongAxes)
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
public func euclideanDistance(_ x: MLTensor, _ y: MLTensor, alongAxes: Int = 1) -> MLTensor {
    (x - y).squared().sum(alongAxes: alongAxes).squareRoot()
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
public func additiveCausalMask<Scalar: MLTensorScalar>(
    _ n: Int32,
    scalarType: Scalar.Type = Float.self
) -> MLTensor {
    let indices = MLTensor(0..<n)
    let mask = indices.expandingShape(at: 1) .< indices.expandingShape(at: 0)
    return mask.cast(to: scalarType) * -1e9
}
