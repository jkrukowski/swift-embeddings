import CoreML

public typealias Layer = @Sendable (MLTensor) -> MLTensor

public func embedding(weight: MLTensor) -> Layer {
    { x in
        weight.gathering(atIndices: x, alongAxis: 0)
    }
}

public func linear(weight: MLTensor, bias: MLTensor? = nil) -> Layer {
    { x in
        if let bias {
            x.matmul(weight.transposed()) + bias
        } else {
            x.matmul(weight.transposed())
        }
    }
}

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
