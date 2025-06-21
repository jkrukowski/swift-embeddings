import CoreML

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
public typealias Layer = @Sendable (MLTensor) -> MLTensor

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
public func embedding(weight: MLTensor) -> Layer {
    { x in
        weight.gathering(atIndices: x, alongAxis: 0)
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
public func linear(weight: MLTensor, bias: MLTensor? = nil) -> Layer {
    { x in
        if let bias {
            x.matmul(weight.transposed()) + bias
        } else {
            x.matmul(weight.transposed())
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
public func layerNorm(weight: MLTensor, bias: MLTensor, epsilon: Float) -> Layer {
    { x in
        let mean = x.mean(alongAxes: -1, keepRank: true)
        let xshift = x - mean
        let variance = xshift.squared().mean(alongAxes: -1, keepRank: true)
        let invstd = (variance + epsilon).rsqrt()
        let norm = xshift * invstd
        return norm * weight + bias
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
public func roPE(dims: Int, base: Int = 10_000) -> Layer {
    { x in
        let shape = x.shape
        let x = x.reshaped(to: [-1, shape[shape.count - 2], shape[shape.count - 1]])
        let N = x.shape[1]
        let (costheta, sintheta) = createCosSinTheta(N: N, D: dims, base: Float(base))
        let rope = computeRoPE(costheta: costheta, sintheta: sintheta, x: x, dims: dims)
        return rope.reshaped(to: shape)
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
func computeRoPE(
    costheta: MLTensor,
    sintheta: MLTensor,
    x: MLTensor,
    dims: Int
) -> MLTensor {
    let x1 = x[..., 0..<(dims / 2)]
    let x2 = x[..., (dims / 2)..<dims]
    let rx1 = x1 * costheta - x2 * sintheta
    let rx2 = x1 * sintheta + x2 * costheta
    if dims < x.shape[x.shape.count - 1] {
        return MLTensor(concatenating: [rx1, rx2, x[..., dims...]], alongAxis: -1)
    } else {
        return MLTensor(concatenating: [rx1, rx2], alongAxis: -1)
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
func createCosSinTheta(
    N: Int,
    D: Int,
    offset: Int = 0,
    base: Float = 10_000
) -> (cos: MLTensor, sin: MLTensor) {
    let D = Float(D / 2)
    let positions = MLTensor(rangeFrom: Float(offset), to: Float(N), by: 1)
    let freqs = (-MLTensor(rangeFrom: Float(0), to: D, by: 1) * (log(base) / D)).exp()
    let theta = positions.reshaped(to: [-1, 1]) * freqs.reshaped(to: [1, -1])
    return (theta.cos(), theta.sin())
}
