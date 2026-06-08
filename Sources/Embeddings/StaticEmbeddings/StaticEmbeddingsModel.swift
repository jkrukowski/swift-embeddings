import CoreML
import Foundation
import MLTensorUtils

public enum StaticEmbeddings {}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension StaticEmbeddings {
    public struct ModelBundle: Sendable {
        public let model: StaticEmbeddings.Model
        public let tokenizer: any TextTokenizer

        public init(
            model: StaticEmbeddings.Model,
            tokenizer: any TextTokenizer
        ) {
            self.model = model
            self.tokenizer = tokenizer
        }

        public func encode(
            _ text: String,
            normalize: Bool = false,
            maxLength: Int? = nil,
            truncateDimension: Int? = nil,
            computePolicy: MLComputePolicy = .cpuAndGPU
        ) throws -> MLTensor {
            try batchEncode(
                [text],
                normalize: normalize,
                maxLength: maxLength,
                truncateDimension: truncateDimension,
                computePolicy: computePolicy
            )
        }

        public func batchEncode(
            _ texts: [String],
            normalize: Bool = false,
            maxLength: Int? = nil,
            truncateDimension: Int? = nil,
            computePolicy: MLComputePolicy = .cpuAndGPU
        ) throws -> MLTensor {
            try withMLTensorComputePolicy(computePolicy) {
                let dimension =
                    if let truncateDimension {
                        min(truncateDimension, model.dimension)
                    } else {
                        model.dimension
                    }
                precondition(dimension > 0, "Dimension must be greater than 0")
                let inputIdsBatch = try texts.map { try tokenize($0, maxLength: maxLength) }
                let embeddingsBatch = inputIdsBatch.map { inputIds in
                    if let inputIds {
                        model.embeddings
                            .gathering(atIndices: inputIds, alongAxis: 0)
                            .mean(alongAxes: 0)[0..<dimension]
                    } else {
                        MLTensor(zeros: [dimension], scalarType: Int32.self)
                    }
                }
                let embeddings = MLTensor(stacking: embeddingsBatch, alongAxis: 0).cast(
                    to: Float.self
                )
                if normalize {
                    let norm = norm(embeddings, alongAxes: 1, keepRank: true) + Float.ulpOfOne
                    return embeddings / norm
                } else {
                    return embeddings
                }
            }
        }

        private func tokenize(
            _ text: String,
            maxLength: Int?
        ) throws -> MLTensor? {
            let tokens = try tokenizer.tokenizeText(
                text,
                maxLength: maxLength,
                addSpecialTokens: false
            )
            return tokens.isEmpty ? nil : MLTensor(shape: [tokens.count], scalars: tokens)
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension StaticEmbeddings {
    public struct Model: Sendable {
        public let embeddings: MLTensor
        public let dimension: Int

        public init(embeddings: MLTensor) {
            self.embeddings = embeddings
            self.dimension = embeddings.shape[1]
        }
    }
}
