import Benchmark
import CoreML
import Foundation
import MLTensorUtils

func activationsBenchmark() {
    if #available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *) {
        let parameterization = [10, 100, 10_000]
        for count in parameterization {
            Benchmark(
                "sigmoid",
                configuration: .init(tags: ["count": count.description])
            ) { benchmark in
                let x = MLTensor(randomUniform: [count, count], scalarType: Float.self)
                benchmark.startMeasurement()
                blackHole(sigmoid(x))
                benchmark.stopMeasurement()
            }

            Benchmark(
                "erf",
                configuration: .init(tags: ["count": count.description])
            ) { benchmark in
                let x = MLTensor(randomUniform: [count, count], scalarType: Float.self)
                benchmark.startMeasurement()
                blackHole(erf(x))
                benchmark.stopMeasurement()
            }

            Benchmark(
                "gelu",
                configuration: .init(tags: ["count": count.description])
            ) { benchmark in
                let x = MLTensor(randomUniform: [count, count], scalarType: Float.self)
                benchmark.startMeasurement()
                blackHole(gelu(x))
                benchmark.stopMeasurement()
            }

            Benchmark(
                "gelu_fast",
                configuration: .init(tags: ["count": count.description])
            ) { benchmark in
                let x = MLTensor(randomUniform: [count, count], scalarType: Float.self)
                benchmark.startMeasurement()
                blackHole(gelu(x, approximation: .fast))
                benchmark.stopMeasurement()
            }

            Benchmark(
                "gelu_precise",
                configuration: .init(tags: ["count": count.description])
            ) { benchmark in
                let x = MLTensor(randomUniform: [count, count], scalarType: Float.self)
                benchmark.startMeasurement()
                blackHole(gelu(x, approximation: .precise))
                benchmark.stopMeasurement()
            }
        }
    }
}
