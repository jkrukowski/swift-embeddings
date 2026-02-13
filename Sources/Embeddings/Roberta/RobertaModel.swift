import CoreML
import Foundation
import MLTensorUtils
import Tokenizers

public enum Roberta {}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Roberta {
    public struct ModelConfig: Codable {
        public var modelType: String
        public var numHiddenLayers: Int
        public var numAttentionHeads: Int
        public var hiddenSize: Int
        public var intermediateSize: Int
        public var maxPositionEmbeddings: Int
        public var hiddenDropoutProb: Float
        public var attentionProbsDropoutProb: Float
        public var typeVocabSize: Int
        public var initializerRange: Float
        public var layerNormEps: Float
        public var vocabSize: Int
        public var padTokenId: Int

        public init(
            modelType: String,
            numHiddenLayers: Int,
            numAttentionHeads: Int,
            hiddenSize: Int,
            intermediateSize: Int,
            maxPositionEmbeddings: Int,
            hiddenDropoutProb: Float = 0.1,
            attentionProbsDropoutProb: Float = 0.1,
            typeVocabSize: Int = 1,
            initializerRange: Float = 0.02,
            layerNormEps: Float = 1e-05,
            vocabSize: Int = 50265,
            padTokenId: Int = 1
        ) {
            self.modelType = modelType
            self.numHiddenLayers = numHiddenLayers
            self.numAttentionHeads = numAttentionHeads
            self.hiddenSize = hiddenSize
            self.intermediateSize = intermediateSize
            self.maxPositionEmbeddings = maxPositionEmbeddings
            self.hiddenDropoutProb = hiddenDropoutProb
            self.attentionProbsDropoutProb = attentionProbsDropoutProb
            self.typeVocabSize = typeVocabSize
            self.initializerRange = initializerRange
            self.layerNormEps = layerNormEps
            self.vocabSize = vocabSize
            self.padTokenId = padTokenId
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Roberta {
    public struct Embeddings: Sendable {
        let wordEmbeddings: MLTensorUtils.Layer
        let positionEmbeddings: MLTensorUtils.Layer
        let tokenTypeEmbeddings: MLTensorUtils.Layer
        let layerNorm: MLTensorUtils.Layer
        let paddingIdx: Int32

        public init(
            wordEmbeddings: @escaping MLTensorUtils.Layer,
            positionEmbeddings: @escaping MLTensorUtils.Layer,
            tokenTypeEmbeddings: @escaping MLTensorUtils.Layer,
            layerNorm: @escaping MLTensorUtils.Layer,
            paddingIdx: Int32
        ) {
            self.wordEmbeddings = wordEmbeddings
            self.positionEmbeddings = positionEmbeddings
            self.tokenTypeEmbeddings = tokenTypeEmbeddings
            self.layerNorm = layerNorm
            self.paddingIdx = paddingIdx
        }

        public func callAsFunction(
            inputIds: MLTensor,
            tokenTypeIds: MLTensor? = nil,
            positionIds: MLTensor? = nil
        ) -> MLTensor {
            let positionIds =
                positionIds
                ?? createPositionIdsFromInputIds(
                    inputIds: inputIds,
                    paddingIdx: paddingIdx
                )
            let tokenTypeIds =
                tokenTypeIds
                ?? MLTensor(
                    zeros: inputIds.shape,
                    scalarType: Int32.self
                )
            let wordsEmbeddings = wordEmbeddings(inputIds)
            let positionEmbeddings = positionEmbeddings(positionIds)
            let tokenTypeEmbeddings = tokenTypeEmbeddings(tokenTypeIds)
            let embeddings = wordsEmbeddings + positionEmbeddings + tokenTypeEmbeddings
            return layerNorm(embeddings)
        }

        private func createPositionIdsFromInputIds(
            inputIds: MLTensor,
            paddingIdx: Int32
        ) -> MLTensor {
            let mask = (inputIds .!= paddingIdx).cast(to: Int32.self)
            let incrementalIndices = mask.cumulativeSum(alongAxis: 1).cast(like: mask) * mask
            return incrementalIndices + paddingIdx
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Roberta {
    public struct Output: Sendable {
        let dense: MLTensorUtils.Layer
        let layerNorm: MLTensorUtils.Layer

        public init(
            dense: @escaping MLTensorUtils.Layer,
            layerNorm: @escaping MLTensorUtils.Layer
        ) {
            self.dense = dense
            self.layerNorm = layerNorm
        }

        public func callAsFunction(
            hiddenStates: MLTensor,
            inputTensor: MLTensor
        ) -> MLTensor {
            let dense = dense(hiddenStates)
            let layerNormInput = dense + inputTensor
            return layerNorm(layerNormInput)
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Roberta {
    public struct Intermediate: Sendable {
        let dense: MLTensorUtils.Layer

        public init(dense: @escaping MLTensorUtils.Layer) {
            self.dense = dense
        }

        public func callAsFunction(hiddenStates: MLTensor) -> MLTensor {
            let dense = dense(hiddenStates)
            return gelu(dense)
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Roberta {
    public struct SelfOutput: Sendable {
        let dense: MLTensorUtils.Layer
        let layerNorm: MLTensorUtils.Layer

        public init(
            dense: @escaping MLTensorUtils.Layer,
            layerNorm: @escaping MLTensorUtils.Layer
        ) {
            self.dense = dense
            self.layerNorm = layerNorm
        }

        public func callAsFunction(
            hiddenStates: MLTensor,
            inputTensor: MLTensor
        ) -> MLTensor {
            let dense = dense(hiddenStates)
            let layerNormInput = dense + inputTensor
            return layerNorm(layerNormInput)
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Roberta {
    public struct SelfAttention: Sendable {
        let query: MLTensorUtils.Layer
        let key: MLTensorUtils.Layer
        let value: MLTensorUtils.Layer
        let numAttentionHeads: Int
        let attentionHeadSize: Int
        let allHeadSize: Int

        public init(
            query: @escaping MLTensorUtils.Layer,
            key: @escaping MLTensorUtils.Layer,
            value: @escaping MLTensorUtils.Layer,
            numAttentionHeads: Int,
            attentionHeadSize: Int,
            allHeadSize: Int
        ) {
            self.query = query
            self.key = key
            self.value = value
            self.numAttentionHeads = numAttentionHeads
            self.attentionHeadSize = attentionHeadSize
            self.allHeadSize = allHeadSize
        }

        private func transposeForScores(_ x: MLTensor) -> MLTensor {
            let newShape = x.shape.dropLast() + [numAttentionHeads, attentionHeadSize]
            return x.reshaped(to: Array(newShape)).transposed(permutation: 0, 2, 1, 3)
        }

        public func callAsFunction(
            hiddenStates: MLTensor,
            attentionMask: MLTensor?
        ) -> MLTensor {
            let mixedQueryLayer = query(hiddenStates)
            let mixedKeyLayer = key(hiddenStates)
            let mixedValueLayer = value(hiddenStates)

            let queryLayer = transposeForScores(mixedQueryLayer)
            let keyLayer = transposeForScores(mixedKeyLayer)
            let valueLayer = transposeForScores(mixedValueLayer)

            var attentionScores = queryLayer.matmul(keyLayer.transposed(permutation: 0, 1, 3, 2))
            attentionScores = attentionScores / sqrt(Float(attentionHeadSize))
            if let attentionMask {
                attentionScores = attentionScores + attentionMask
            }
            let attentionProbs = attentionScores.softmax(alongAxis: -1)
            var contextLayer = attentionProbs.matmul(valueLayer)
            contextLayer = contextLayer.transposed(permutation: [0, 2, 1, 3])
            let newContextLayerShape = contextLayer.shape.dropLast(2) + [allHeadSize]
            return contextLayer.reshaped(to: Array(newContextLayerShape))
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Roberta {
    public struct Attention: Sendable {
        let selfAttention: Roberta.SelfAttention
        let output: Roberta.SelfOutput

        public init(
            selfAttention: Roberta.SelfAttention,
            output: Roberta.SelfOutput
        ) {
            self.selfAttention = selfAttention
            self.output = output
        }

        public func callAsFunction(
            hiddenStates: MLTensor,
            attentionMask: MLTensor?
        ) -> MLTensor {
            let selfOutputs = selfAttention(
                hiddenStates: hiddenStates,
                attentionMask: attentionMask
            )
            return output(
                hiddenStates: selfOutputs,
                inputTensor: hiddenStates
            )
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Roberta {
    public struct Layer: Sendable {
        let attention: Roberta.Attention
        let intermediate: Roberta.Intermediate
        let output: Roberta.Output

        public init(
            attention: Roberta.Attention,
            intermediate: Roberta.Intermediate,
            output: Roberta.Output
        ) {
            self.attention = attention
            self.intermediate = intermediate
            self.output = output
        }

        public func callAsFunction(
            hiddenStates: MLTensor,
            attentionMask: MLTensor?
        ) -> MLTensor {
            let attentionOutput = attention(
                hiddenStates: hiddenStates,
                attentionMask: attentionMask
            )
            let intermediateOutput = intermediate(
                hiddenStates: attentionOutput
            )
            return output(
                hiddenStates: intermediateOutput,
                inputTensor: attentionOutput
            )
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Roberta {
    public struct Encoder: Sendable {
        let layers: [Roberta.Layer]

        public init(layers: [Roberta.Layer]) {
            self.layers = layers
        }

        public func callAsFunction(
            hiddenStates: MLTensor,
            attentionMask: MLTensor?
        ) -> MLTensor {
            var hiddenStates = hiddenStates
            for layer in layers {
                hiddenStates = layer(
                    hiddenStates: hiddenStates,
                    attentionMask: attentionMask
                )
            }
            return hiddenStates
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Roberta {
    public struct Model: Sendable {
        let embeddings: Roberta.Embeddings
        let encoder: Roberta.Encoder

        public init(
            embeddings: Roberta.Embeddings,
            encoder: Roberta.Encoder
        ) {
            self.embeddings = embeddings
            self.encoder = encoder
        }

        public func callAsFunction(
            inputIds: MLTensor,
            tokenTypeIds: MLTensor? = nil,
            attentionMask: MLTensor? = nil
        ) -> MLTensor {
            let embeddingOutput = embeddings(inputIds: inputIds, tokenTypeIds: tokenTypeIds)
            let mask: MLTensor? =
                if let attentionMask {
                    (1.0 - attentionMask.expandingShape(at: 1, 1)) * -10000.0
                } else {
                    nil
                }
            return encoder(hiddenStates: embeddingOutput, attentionMask: mask)
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Roberta {
    public struct ModelBundle: Sendable {
        public let model: Roberta.Model
        public let tokenizer: any TextTokenizer

        public init(
            model: Roberta.Model,
            tokenizer: any TextTokenizer
        ) {
            self.model = model
            self.tokenizer = tokenizer
        }

        public func encode(
            _ text: String,
            maxLength: Int = 512,
            postProcess: PostProcess? = nil
        ) throws -> MLTensor {
            let tokens = try tokenizer.tokenizeText(text, maxLength: maxLength)
            let inputIds = MLTensor(shape: [1, tokens.count], scalars: tokens)
            let result = model(inputIds: inputIds)
            return processResult(result, with: postProcess)
        }

        public func batchEncode(
            _ texts: [String],
            padTokenId: Int = 0,
            maxLength: Int = 512,
            postProcess: PostProcess? = nil
        ) throws -> MLTensor {
            let batchTokenizeResult = try tokenizer.tokenizeTextsPaddingToLongest(
                texts, padTokenId: padTokenId, maxLength: maxLength)
            let inputIds = MLTensor(
                shape: batchTokenizeResult.shape,
                scalars: batchTokenizeResult.tokens)
            let attentionMask = MLTensor(
                shape: batchTokenizeResult.shape,
                scalars: batchTokenizeResult.attentionMask)
            let result = model(inputIds: inputIds, attentionMask: attentionMask)
            return processResult(result, with: postProcess, attentionMask: attentionMask)
        }
    }
}
