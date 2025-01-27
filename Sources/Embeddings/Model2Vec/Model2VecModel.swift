import CoreML
import Foundation
import MLTensorUtils

public enum Model2Vec {}

extension Model2Vec {
    public struct ModelConfig: Codable {
        public var normalize: Bool?

        public init(normalize: Bool? = nil) {
            self.normalize = normalize
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Model2Vec {
    public struct ModelBundle: Sendable {
        public let model: Model2Vec.Model
        public let tokenizer: any TextTokenizer

        public init(
            model: Model2Vec.Model,
            tokenizer: any TextTokenizer
        ) {
            self.model = model
            self.tokenizer = tokenizer
        }

        public func encode(_ text: String, maxLength: Int? = nil) throws -> MLTensor {
            try batchEncode([text], maxLength: maxLength)
        }

        public func batchEncode(_ texts: [String], maxLength: Int? = nil) throws -> MLTensor {
            let inputIdsBatch = try texts.map { try tokenize($0, maxLength: maxLength) }
            let embeddingsBatch = inputIdsBatch.map { inputIds in
                model.embeddings
                    .gathering(atIndices: inputIds, alongAxis: 0)
                    .mean(alongAxes: 0)
            }
            let embeddings = MLTensor(stacking: embeddingsBatch, alongAxis: 0)
            if model.normalize {
                let norm = norm(embeddings, alongAxes: 1, keepRank: true) + Float.ulpOfOne
                return embeddings / norm
            } else {
                return embeddings
            }
        }

        private func tokenize(_ text: String, maxLength: Int? = nil) throws -> MLTensor {
            let tokensIds = try tokenizer.tokenizeText(
                text, maxLength: maxLength, addSpecialTokens: false)
            let tokens =
                if let unknownTokenId = tokenizer.unknownTokenId {
                    tokensIds.filter { $0 != unknownTokenId }
                } else {
                    tokensIds
                }
            if tokens.isEmpty {
                return MLTensor(zeros: [model.dimienstion], scalarType: Int32.self)
            } else {
                return MLTensor(shape: [tokens.count], scalars: tokens)
            }
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Model2Vec {
    public struct Model: Sendable {
        public let embeddings: MLTensor
        public let dimienstion: Int
        public let normalize: Bool

        public init(embeddings: MLTensor, normalize: Bool = false) {
            self.embeddings = embeddings
            self.dimienstion = embeddings.shape[1]
            self.normalize = normalize
        }
    }
}
