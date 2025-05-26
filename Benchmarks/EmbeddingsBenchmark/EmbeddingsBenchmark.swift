import Benchmark
import CoreML
import Foundation
import MLTensorUtils

let benchmarks = { @Sendable in
    activationsBenchmark()
    layersBenchmark()
}
