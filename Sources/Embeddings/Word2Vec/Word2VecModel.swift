import CoreML
import Foundation
import MLTensorUtils

public enum Word2Vec {}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Word2Vec {
    public struct ModelBundle: Sendable {
        public let keyToIndex: [String: Int]
        public let indexToKey: [Int: String]
        public let embeddings: MLTensor

        public init(
            keyToIndex: [String: Int],
            indexToKey: [Int: String],
            embeddings: MLTensor
        ) {
            self.keyToIndex = keyToIndex
            self.indexToKey = indexToKey
            self.embeddings = embeddings
        }

        public func encode(_ word: String) -> MLTensor? {
            guard let index = keyToIndex[word] else {
                return nil
            }
            return embeddings[index]
        }

        public func batchEncode(_ words: [String]) -> MLTensor? {
            let indices = words.compactMap { keyToIndex[$0] }
            let rows = indices.map { embeddings[$0] }
            return MLTensor(stacking: rows, alongAxis: 0)
        }

        public func mostSimilar(
            to word: String,
            topK: Int = 1
        ) async -> [(word: String, score: Float)] {
            guard let wordIndex = keyToIndex[word] else {
                return []
            }
            // Get the embedding vector for the input word
            let wordVector = embeddings[wordIndex]

            // Normalize the word vector
            let wordVectorNorm = wordVector / (norm(wordVector, alongAxes: 0) + Float.ulpOfOne)

            // Normalize all embedding vectors
            let norms = norm(embeddings, alongAxes: 1) + Float.ulpOfOne
            let normalizedEmbeddings = embeddings / norms.expandingShape(at: 1)

            // Compute similarity
            let similarities = normalizedEmbeddings.matmul(wordVectorNorm.transposed())

            // +1 to account for the input word. NOTE: using `topK` function results in a hard crash
            let indices = similarities.argsort(descendingOrder: true)[
                ..<min(topK + 1, similarities.shape[0])]
            let values = similarities.gathering(atIndices: indices, alongAxis: 0)

            async let topIndices = indices.shapedArray(of: Int32.self).scalars
            async let topScores = values.shapedArray(of: Float.self).scalars

            return await zip(topIndices, topScores)
                .filter { Int($0.0) != wordIndex }
                .compactMap { index, score in
                    guard let word = indexToKey[Int(index)] else { return nil }
                    return (word, score)
                }
        }
    }
}
