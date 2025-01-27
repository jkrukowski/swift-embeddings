import CoreML
import MLTensorUtils
import Testing
import TestingUtils
import XCTest

@testable import Embeddings

struct BertTests {
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test func pooler() async {
        let pooler1 = Bert.Pooler(
            dense: MLTensorUtils.linear(
                weight: MLTensor.float(shape: [5, 5]),
                bias: nil
            )
        )
        let result1 = pooler1(
            MLTensor.float(shape: [1, 3, 5])
        )
        let data1 = await result1.scalars(of: Float.self)

        #expect(result1.shape == [1, 5])
        #expect(allClose(data1, [1, 1, 1, 1, 1]) == true)

        let pooler2 = Bert.Pooler(
            dense: MLTensorUtils.linear(
                weight: MLTensor.float(shape: [5, 5]),
                bias: MLTensor.float(shape: [5])
            )
        )
        let result2 = pooler2(
            MLTensor.float(shape: [1, 3, 5])
        )
        let data2 = await result2.scalars(of: Float.self)

        #expect(result2.shape == [1, 5])
        #expect(allClose(data2, [1, 1, 1, 1, 1]) == true)
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test func intermediate() async {
        let intermediate1 = Bert.Intermediate(
            dense: MLTensorUtils.linear(
                weight: MLTensor.float(shape: [2, 3]),
                bias: nil
            )
        )
        let result1 = intermediate1(
            hiddenStates: MLTensor.float(shape: [1, 2, 3])
        )
        let data1 = await result1.scalars(of: Float.self)

        #expect(result1.shape == [1, 2, 2])
        #expect(allClose(data1, [5.0, 14.0, 14.0, 50.0]) == true)

        let intermediate2 = Bert.Intermediate(
            dense: MLTensorUtils.linear(
                weight: MLTensor.float(shape: [2, 3]),
                bias: MLTensor.float(shape: [2])
            )
        )
        let result2 = intermediate2(
            hiddenStates: MLTensor.float(shape: [1, 2, 3])
        )
        let data2 = await result2.scalars(of: Float.self)

        #expect(result2.shape == [1, 2, 2])
        #expect(allClose(data2, [5.0, 15.0, 14.0, 51.0]) == true)
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test func output() async {
        let output1 = Bert.Output(
            dense: MLTensorUtils.linear(
                weight: MLTensor.float(shape: [4, 4]),
                bias: nil
            ),
            layerNorm: MLTensorUtils.layerNorm(
                weight: MLTensor.float(shape: [4]),
                bias: MLTensor.float(shape: [4]),
                epsilon: 1e-5
            )
        )

        let result1 = output1(
            hiddenStates: MLTensor.float(shape: [1, 2, 4]),
            inputTensor: MLTensor.float(shape: [1, 2, 4])
        )
        let data1 = await result1.scalars(of: Float.self)

        #expect(result1.shape == [1, 2, 4])
        #expect(
            allClose(
                data1,
                [0.0, 0.5527864, 2.8944273, 7.0249224, 0.0, 0.5527864, 2.8944273, 7.0249224])
                == true)

        let output2 = Bert.Output(
            dense: MLTensorUtils.linear(
                weight: MLTensor.float(shape: [4, 4]),
                bias: MLTensor.float(shape: [4])
            ),
            layerNorm: MLTensorUtils.layerNorm(
                weight: MLTensor.float(shape: [4]),
                bias: MLTensor.float(shape: [4]),
                epsilon: 1e-5
            )
        )

        let result2 = output2(
            hiddenStates: MLTensor.float(shape: [1, 2, 4]),
            inputTensor: MLTensor.float(shape: [1, 2, 4])
        )
        let data2 = await result2.scalars(of: Float.self)

        #expect(result2.shape == [1, 2, 4])
        #expect(
            allClose(
                data2,
                [0.0, 0.55278635, 2.8944273, 7.0249224, 0.0, 0.5527864, 2.8944273, 7.0249224])
                == true)
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
final class BertEmbeddingTests: XCTestCase {
    // NOTE: this test is not stable when running using `Testing` library, not sure why
    func testEmbeddings() async {
        let wordEmbeddings = MLTensorUtils.embedding(weight: MLTensor.float(shape: [2, 4]))
        let positionEmbeddings = MLTensorUtils.embedding(weight: MLTensor.float(shape: [1, 4]))
        let tokenTypeEmbeddings = MLTensorUtils.embedding(weight: MLTensor.float(shape: [2, 4]))
        let embeddings = Bert.Embeddings(
            wordEmbeddings: wordEmbeddings,
            positionEmbeddings: positionEmbeddings,
            tokenTypeEmbeddings: tokenTypeEmbeddings,
            layerNorm: MLTensorUtils.layerNorm(
                weight: MLTensor.float(shape: [4]),
                bias: MLTensor.float(shape: [4]),
                epsilon: 1e-5
            )
        )

        let result = embeddings(inputIds: MLTensor.int32(shape: [1, 2]))
        let data = await result.scalars(of: Float.self)

        XCTAssertEqual(result.shape, [1, 2, 4])
        XCTAssertTrue(
            allClose(data, [0, 0.552787, 2.89443, 7.02492, 0, 0.552787, 2.89443, 7.02492]))
    }
}
