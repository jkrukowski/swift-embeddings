import CoreML
import MLTensorUtils
import Testing
import TestingUtils

@testable import Embeddings

struct NomicBertTests {
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test func gatedMLP() async {
        let gateUpWeight = MLTensor(
            shape: [4, 2],
            scalars: [
                1, 0,
                0, 1,
                1, 0,
                0, 1,
            ]
        )
        let downWeight = MLTensor(
            shape: [2, 2],
            scalars: [
                1, 0,
                0, 1,
            ]
        )
        let mlp = NomicBert.MLP(
            gateUp: MLTensorUtils.linear(weight: gateUpWeight),
            down: MLTensorUtils.linear(weight: downWeight)
        )
        let result = mlp(
            MLTensor(shape: [1, 1, 2], scalars: [1, 1])
        )
        let data = await result.scalars(of: Float.self)

        #expect(result.shape == [1, 1, 2])
        #expect(allClose(data, [0.7310586, 0.7310586], absoluteTolerance: 1e-5) == true)
    }
}
