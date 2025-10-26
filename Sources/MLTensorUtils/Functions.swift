import CoreML
import Foundation

// Scaled dot product attention
// Ref: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch-nn-functional-scaled-dot-product-attention
@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
public func sdpa(
    query: MLTensor,
    key: MLTensor,
    value: MLTensor,
    attntMask: MLTensor? = nil,
    scale: Float? = nil
) -> MLTensor {
    precondition(
        query.rank >= 2,
        "query must have at least 2 dimensions, got shape \(query.shape)")
    precondition(
        key.rank >= 2,
        "key must have at least 2 dimensions, got shape \(key.shape)")
    precondition(
        value.rank >= 2,
        "value must have at least 2 dimensions, got shape \(value.shape)")
    precondition(
        query.rank == key.rank,
        "query and key must have the same rank, got query: \(query.rank), key: \(key.rank)")
    precondition(
        key.rank == value.rank,
        "key and value must have the same rank, got key: \(key.rank), value: \(value.rank)")

    let L = query.shape[query.shape.count - 2]
    let S = key.shape[key.shape.count - 2]
    let queryLastDim = query.shape[query.shape.count - 1]
    let scaleFactor = scale ?? (1.0 / sqrt(Float(queryLastDim)))
    let attnBias = attntMask ?? MLTensor(repeating: 0.0, shape: [L, S])

    let rank = key.rank
    var permutation = Array(0..<rank)
    permutation[rank - 2] = rank - 1
    permutation[rank - 1] = rank - 2
    let keyTransposed = key.transposed(permutation: permutation)

    var attnWeight = query.matmul(keyTransposed) * scaleFactor
    attnWeight = attnWeight + attnBias
    attnWeight = attnWeight.softmax(alongAxis: -1)
    return attnWeight.matmul(value)
}

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
