import CoreML
import Foundation
import MLTensorUtils
import Tokenizers

public enum Clip {}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Clip {
    public struct TextConfig: Codable {
        public var numHiddenLayers: Int
        public var hiddenSize: Int
        public var intermediateSize: Int
        public var numAttentionHeads: Int
        public var layerNormEps: Float
        public var maxPositionEmbeddings: Int
        public var vocabSize: Int

        public init(
            numHiddenLayers: Int,
            hiddenSize: Int,
            intermediateSize: Int,
            numAttentionHeads: Int,
            layerNormEps: Float,
            maxPositionEmbeddings: Int,
            vocabSize: Int
        ) {
            self.numHiddenLayers = numHiddenLayers
            self.hiddenSize = hiddenSize
            self.intermediateSize = intermediateSize
            self.numAttentionHeads = numAttentionHeads
            self.layerNormEps = layerNormEps
            self.maxPositionEmbeddings = maxPositionEmbeddings
            self.vocabSize = vocabSize
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Clip {
    public struct VisionConfig: Codable {
        public var numHiddenLayers: Int
        public var hiddenSize: Int
        public var intermediateSize: Int
        public var numAttentionHeads: Int
        public var layerNormEps: Float
        public var numChannels: Int?
        public var imageSize: Int
        public var patchSize: Int

        public init(
            numHiddenLayers: Int,
            hiddenSize: Int,
            intermediateSize: Int,
            numAttentionHeads: Int,
            layerNormEps: Float,
            numChannels: Int? = nil,
            imageSize: Int,
            patchSize: Int
        ) {
            self.numHiddenLayers = numHiddenLayers
            self.hiddenSize = hiddenSize
            self.intermediateSize = intermediateSize
            self.numAttentionHeads = numAttentionHeads
            self.layerNormEps = layerNormEps
            self.numChannels = numChannels
            self.imageSize = imageSize
            self.patchSize = patchSize
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Clip {
    public struct ModelConfig: Codable {
        public var textConfig: TextConfig
        public var visionConfig: VisionConfig
        public var projectionDim: Int

        public init(
            textConfig: TextConfig,
            visionConfig: VisionConfig,
            projectionDim: Int
        ) {
            self.textConfig = textConfig
            self.visionConfig = visionConfig
            self.projectionDim = projectionDim
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Clip {
    public struct Embeddings: Sendable {
        let tokenEmbedding: MLTensorUtils.Layer
        let positionEmbeddingWeight: MLTensor

        public init(
            tokenEmbedding: @escaping MLTensorUtils.Layer,
            positionEmbeddingWeight: MLTensor
        ) {
            self.tokenEmbedding = tokenEmbedding
            self.positionEmbeddingWeight = positionEmbeddingWeight
        }

        public func callAsFunction(
            x: MLTensor
        ) -> MLTensor {
            let embeddings = tokenEmbedding(x)
            return embeddings + positionEmbeddingWeight[0..<x.shape[1]]
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Clip {
    public struct MLP: Sendable {
        let fc1: MLTensorUtils.Layer
        let fc2: MLTensorUtils.Layer

        public init(
            fc1: @escaping MLTensorUtils.Layer,
            fc2: @escaping MLTensorUtils.Layer
        ) {
            self.fc1 = fc1
            self.fc2 = fc2
        }

        public func callAsFunction(
            x: MLTensor
        ) -> MLTensor {
            fc2(gelu(fc1(x), approximation: .fast))
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Clip {
    public struct Attention: Sendable {
        let qProj: MLTensorUtils.Layer
        let kProj: MLTensorUtils.Layer
        let vProj: MLTensorUtils.Layer
        let outProj: MLTensorUtils.Layer
        private let numHeads: Int

        public init(
            qProj: @escaping MLTensorUtils.Layer,
            kProj: @escaping MLTensorUtils.Layer,
            vProj: @escaping MLTensorUtils.Layer,
            outProj: @escaping MLTensorUtils.Layer,
            numHeads: Int
        ) {
            self.qProj = qProj
            self.kProj = kProj
            self.vProj = vProj
            self.outProj = outProj
            self.numHeads = numHeads
        }

        public func callAsFunction(
            queries: MLTensor,
            keys: MLTensor,
            values: MLTensor,
            mask: MLTensor? = nil
        ) -> MLTensor {
            var queries = qProj(queries)
            var keys = kProj(keys)
            var values = vProj(values)
            let B = queries.shape[0]
            let L = queries.shape[1]
            let S = keys.shape[1]
            queries = queries.reshaped(to: [B, L, numHeads, -1]).transposed(permutation: 0, 2, 1, 3)
            keys = keys.reshaped(to: [B, S, numHeads, -1]).transposed(permutation: 0, 2, 3, 1)
            values = values.reshaped(to: [B, S, numHeads, -1]).transposed(permutation: 0, 2, 1, 3)
            let scale = sqrt(1.0 / Float(queries.shape.last!))
            var scores = (queries * scale).matmul(keys)
            if let mask = mask {
                scores = scores + mask.cast(like: scores)
            }
            scores = scores.softmax(alongAxis: -1)
            let valuesHat = scores.matmul(values)
                .transposed(permutation: 0, 2, 1, 3)
                .reshaped(to: [B, L, -1])
            return outProj(valuesHat)
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Clip {
    public struct EncoderLayer: Sendable {
        let selfAttnention: Attention
        let mlp: MLP
        let layerNorm1: MLTensorUtils.Layer
        let layerNorm2: MLTensorUtils.Layer

        public init(
            selfAttnention: Attention,
            mlp: MLP,
            layerNorm1: @escaping MLTensorUtils.Layer,
            layerNorm2: @escaping MLTensorUtils.Layer
        ) {
            self.selfAttnention = selfAttnention
            self.mlp = mlp
            self.layerNorm1 = layerNorm1
            self.layerNorm2 = layerNorm2
        }

        public func callAsFunction(
            x: MLTensor,
            mask: MLTensor? = nil
        ) -> MLTensor {
            var y = layerNorm1(x)
            y = selfAttnention(queries: y, keys: y, values: y, mask: mask)
            let x = x + y
            y = layerNorm2(x)
            y = mlp(x: y)
            return x + y
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Clip {
    public struct Encoder: Sendable {
        let layers: [EncoderLayer]

        public init(layers: [EncoderLayer]) {
            self.layers = layers
        }

        public func callAsFunction(
            x: MLTensor,
            mask: MLTensor? = nil
        ) -> MLTensor {
            var x = x
            for layer in layers {
                x = layer(x: x, mask: mask)
            }
            return x
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Clip {
    public struct TextModel: Sendable {
        let embeddings: Embeddings
        let encoder: Encoder
        let finalLayerNorm: MLTensorUtils.Layer
        let textProjection: MLTensorUtils.Layer

        public init(
            embeddings: Embeddings,
            encoder: Encoder,
            finalLayerNorm: @escaping MLTensorUtils.Layer,
            textProjection: @escaping MLTensorUtils.Layer
        ) {
            self.embeddings = embeddings
            self.encoder = encoder
            self.finalLayerNorm = finalLayerNorm
            self.textProjection = textProjection
        }

        public func callAsFunction(
            inputIds: MLTensor
        ) -> (lastHiddenState: MLTensor, poolerOutput: MLTensor) {
            let N = Int32(inputIds.shape[1])
            let eotTokens = inputIds.argmax(alongAxis: -1)
            var x = embeddings(x: inputIds)
            let mask = additiveCausalMask(N, scalarType: x.scalarType)
            x = encoder(x: x, mask: mask)
            let lastHiddenState = finalLayerNorm(x)
            let poolerOutput = lastHiddenState.gathering(atIndices: eotTokens, alongAxis: 1)
            return (lastHiddenState: lastHiddenState, poolerOutput: poolerOutput)
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Clip {
    public struct ModelBundle: Sendable {
        public let textModel: Clip.TextModel
        public let tokenizer: any TextTokenizer

        public init(
            textModel: Clip.TextModel,
            tokenizer: any TextTokenizer
        ) {
            self.textModel = textModel
            self.tokenizer = tokenizer
        }

        public func encode(_ text: String, maxLength: Int = 77) throws -> MLTensor {
            let tokens = try tokenizer.tokenizeText(text, maxLength: maxLength)
            let inputIds = MLTensor(shape: [1, tokens.count], scalars: tokens)
            let modelOutput = textModel(inputIds: inputIds)
            let textEmbeddings = textModel.textProjection(modelOutput.poolerOutput)
            return textEmbeddings / norm(textEmbeddings, alongAxes: -1, keepRank: true)
        }

        public func batchEncode(
            _ texts: [String],
            padTokenId: Int = 0,
            maxLength: Int = 77
        ) throws -> MLTensor {
            let encodedTexts = try tokenizer.tokenizeTextsPaddingToLongest(
                texts, padTokenId: padTokenId, maxLength: maxLength)
            let inputIds = MLTensor(
                shape: [encodedTexts.count, encodedTexts[0].count],
                scalars: encodedTexts.flatMap { $0 })
            let modelOutput = textModel(inputIds: inputIds)
            let textEmbeddings = textModel.textProjection(modelOutput.poolerOutput)
            return textEmbeddings / norm(textEmbeddings, alongAxes: -1, keepRank: true)
        }
    }
}
