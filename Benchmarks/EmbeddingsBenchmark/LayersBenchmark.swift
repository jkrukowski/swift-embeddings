import Benchmark
import CoreML
import Foundation
import MLTensorUtils

func layersBenchmark() {
    if #available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *) {
        let parameterization = [10, 100, 10_000]
        for count in parameterization {
            Benchmark(
                "embedding",
                configuration: .init(tags: ["count": count.description])
            ) { benchmark in
                let weight = MLTensor(randomUniform: [count, count], scalarType: Float.self)
                let x = MLTensor(randomUniform: [count], scalarType: Int32.self)
                benchmark.startMeasurement()
                blackHole(embedding(weight: weight)(x))
                benchmark.stopMeasurement()
            }

            Benchmark(
                "linear",
                configuration: .init(tags: ["count": count.description])
            ) { benchmark in
                let weight = MLTensor(randomUniform: [count, count], scalarType: Float.self)
                let bias = MLTensor(randomUniform: [count], scalarType: Float.self)
                let x = MLTensor(randomUniform: [count, count], scalarType: Float.self)
                benchmark.startMeasurement()
                blackHole(linear(weight: weight, bias: bias)(x))
                benchmark.stopMeasurement()
            }

            Benchmark(
                "layer_norm",
                configuration: .init(tags: ["count": count.description])
            ) { benchmark in
                let weight = MLTensor(randomUniform: [count, count], scalarType: Float.self)
                let bias = MLTensor(randomUniform: [count], scalarType: Float.self)
                let x = MLTensor(randomUniform: [count, count], scalarType: Float.self)
                benchmark.startMeasurement()
                blackHole(layerNorm(weight: weight, bias: bias, epsilon: 1e-5)(x))
                benchmark.stopMeasurement()
            }
        }
    }
}
