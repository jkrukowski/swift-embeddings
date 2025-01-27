import CoreML
import MLTensorUtils
import Testing
import TestingUtils

@testable import Embeddings

struct Word2VecTests {
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test func mostSimilar() async {
        let modelBundle = Word2Vec.ModelBundle(
            keyToIndex: ["a": 0, "b": 1, "c": 2, "d": 3],
            indexToKey: [0: "a", 1: "b", 2: "c", 3: "d"],
            embeddings: MLTensor.float(shape: [4, 2])
        )
        let result1 = await modelBundle.mostSimilar(to: "a", topK: 3)
        let words1 = result1.map(\.word)
        let scores1 = result1.map(\.score)
        #expect(words1 == ["b", "c", "d"])
        #expect(allClose(scores1, [0.8320502, 0.78086865, 0.7592565]) == true)

        let result2 = await modelBundle.mostSimilar(to: "c")
        let words2 = result2.map(\.word)
        let scores2 = result2.map(\.score)
        #expect(words2 == ["d"])
        #expect(allClose(scores2, [0.9994259]) == true)
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test func encode() async throws {
        let modelBundle = Word2Vec.ModelBundle(
            keyToIndex: ["a": 0, "b": 1, "c": 2, "d": 3],
            indexToKey: [0: "a", 1: "b", 2: "c", 3: "d"],
            embeddings: MLTensor.float(shape: [4, 2])
        )
        #expect(modelBundle.encode("e") == nil)

        let encoded = try #require(modelBundle.encode("a"))
        #expect(encoded.shape == [2])
        let encodedShapedArray = await encoded.shapedArray(of: Float.self).scalars
        #expect(encodedShapedArray == [0.0, 1.0])

        let batchEncoded = try #require(modelBundle.batchEncode(["a", "c", "d", "e"]))
        #expect(batchEncoded.shape == [3, 2])
        let batchEncodedShapedArray = await batchEncoded.shapedArray(of: Float.self).scalars
        #expect(batchEncodedShapedArray == [0.0, 1.0, 4.0, 5.0, 6.0, 7.0])
    }
}
