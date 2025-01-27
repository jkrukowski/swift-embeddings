import CoreML
import Foundation
import MLTensorUtils
@preconcurrency import Tokenizers

public enum XLMRoberta {}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension XLMRoberta {
    public struct ModelConfig: Codable {
        public let hiddenSize: Int
        public let numHiddenLayers: Int
        public let intermediateSize: Int
        public let numAttentionHeads: Int
        public let maxPositionEmbeddings: Int
        public let layerNormEps: Float
        public let vocabSize: Int
        public let addPoolingLayer: Bool?
        public let attentionProbsDropoutProb: Float
        public let hiddenDropoutProb: Float
        public let typeVocabSize: Int
        public let outputPast: Bool
        public let padTokenId: Int
        public let positionEmbeddingType: String
        public let poolingConfig: [String: String]?

        public init(
            hiddenSize: Int,
            numHiddenLayers: Int,
            intermediateSize: Int,
            numAttentionHeads: Int,
            maxPositionEmbeddings: Int,
            layerNormEps: Float,
            vocabSize: Int,
            addPoolingLayer: Bool?,
            attentionProbsDropoutProb: Float,
            hiddenDropoutProb: Float,
            typeVocabSize: Int,
            outputPast: Bool,
            padTokenId: Int,
            positionEmbeddingType: String,
            poolingConfig: [String: String]?
        ) {
            self.hiddenSize = hiddenSize
            self.numHiddenLayers = numHiddenLayers
            self.intermediateSize = intermediateSize
            self.numAttentionHeads = numAttentionHeads
            self.maxPositionEmbeddings = maxPositionEmbeddings
            self.layerNormEps = layerNormEps
            self.vocabSize = vocabSize
            self.addPoolingLayer = addPoolingLayer
            self.attentionProbsDropoutProb = attentionProbsDropoutProb
            self.hiddenDropoutProb = hiddenDropoutProb
            self.typeVocabSize = typeVocabSize
            self.outputPast = outputPast
            self.padTokenId = padTokenId
            self.positionEmbeddingType = positionEmbeddingType
            self.poolingConfig = poolingConfig
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension XLMRoberta {
    public struct Pooler: Sendable {
        let dense: MLTensorUtils.Layer

        public init(dense: @escaping MLTensorUtils.Layer) {
            self.dense = dense
        }

        public func callAsFunction(hiddenStates: MLTensor) -> MLTensor {
            let firstTokenTensor = hiddenStates[0..., 0]
            let pooledOutput = dense(firstTokenTensor)
            return pooledOutput.tanh()
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension XLMRoberta {
    public struct Embeddings: Sendable {
        let wordEmbeddings: MLTensorUtils.Layer
        let positionEmbeddings: MLTensorUtils.Layer
        let tokenTypeEmbeddings: MLTensorUtils.Layer
        let layerNorm: MLTensorUtils.Layer
        let paddingIndex: Int32

        public init(
            wordEmbeddings: @escaping MLTensorUtils.Layer,
            positionEmbeddings: @escaping MLTensorUtils.Layer,
            tokenTypeEmbeddings: @escaping MLTensorUtils.Layer,
            layerNorm: @escaping MLTensorUtils.Layer,
            paddingIndex: Int32
        ) {
            self.wordEmbeddings = wordEmbeddings
            self.positionEmbeddings = positionEmbeddings
            self.tokenTypeEmbeddings = tokenTypeEmbeddings
            self.layerNorm = layerNorm
            self.paddingIndex = paddingIndex
        }

        private func createPositionIds(
            from inputIds: MLTensor,
            pastKeyValuesLength: Int32
        ) -> MLTensor {
            let mask = (inputIds .!= paddingIndex).cast(to: Int32.self)
            let incrementalIndices = (mask.cumulativeSum(alongAxis: 1) + pastKeyValuesLength) * mask
            return incrementalIndices + paddingIndex
        }

        public func callAsFunction(
            inputIds: MLTensor,
            tokenTypeIds: MLTensor? = nil,
            positionIds: MLTensor? = nil,
            inputsEmbeds: MLTensor? = nil,
            pastKeyValuesLength: Int32 = 0
        ) -> MLTensor {
            let positionIds =
                positionIds
                ?? createPositionIds(from: inputIds, pastKeyValuesLength: pastKeyValuesLength)
            let tokenTypeIds =
                tokenTypeIds
                ?? MLTensor(
                    zeros: inputIds.shape,
                    scalarType: Int32.self
                )
            let inputEmbeddings = inputsEmbeds ?? wordEmbeddings(inputIds)
            let positionEmbeddings = positionEmbeddings(positionIds)
            let tokenTypeEmbeddings = tokenTypeEmbeddings(tokenTypeIds)
            let embeddings = inputEmbeddings + tokenTypeEmbeddings + positionEmbeddings
            return layerNorm(embeddings)
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension XLMRoberta {
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
extension XLMRoberta {
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
extension XLMRoberta {
    public struct SelfAttention: Sendable {
        let query: MLTensorUtils.Layer
        let key: MLTensorUtils.Layer
        let value: MLTensorUtils.Layer
        let numAttentionHeads: Int
        let attentionHeadSize: Int
        let allHeadSize: Int
        let scale: Float

        public init(
            query: @escaping MLTensorUtils.Layer,
            key: @escaping MLTensorUtils.Layer,
            value: @escaping MLTensorUtils.Layer,
            numAttentionHeads: Int,
            attentionHeadSize: Int,
            allHeadSize: Int,
            scale: Float
        ) {
            self.query = query
            self.key = key
            self.value = value
            self.numAttentionHeads = numAttentionHeads
            self.attentionHeadSize = attentionHeadSize
            self.allHeadSize = allHeadSize
            self.scale = scale
        }

        private func transposeForScores(_ x: MLTensor) -> MLTensor {
            let newShape = x.shape.dropLast() + [numAttentionHeads, attentionHeadSize]
            return x.reshaped(to: Array(newShape)).transposed(permutation: 0, 2, 1, 3)
        }

        public func callAsFunction(
            hiddenStates: MLTensor,
            attentionMask: MLTensor?,
            headMask: MLTensor?
        ) -> MLTensor {
            let queries = query(hiddenStates)
            let keys = key(hiddenStates)
            let values = value(hiddenStates)

            let queryLayer = transposeForScores(queries)
            let keyLayer = transposeForScores(keys)
            let valueLayer = transposeForScores(values)

            var attentionScores = queryLayer.matmul(keyLayer.transposed(permutation: 0, 1, 3, 2))
            attentionScores = attentionScores / sqrt(Float(attentionHeadSize))
            if let attentionMask {
                attentionScores = attentionScores + attentionMask
            }
            var attentionProbs = attentionScores.softmax(alongAxis: -1)
            if let headMask {
                attentionProbs = attentionProbs * headMask
            }
            var contextLayer = attentionProbs.matmul(valueLayer)
            contextLayer = contextLayer.transposed(permutation: 0, 2, 1, 3)
            let newContextLayerShape = contextLayer.shape.dropLast(2) + [allHeadSize]
            return contextLayer.reshaped(to: Array(newContextLayerShape))
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension XLMRoberta {
    public struct Attention: Sendable {
        let selfAttention: XLMRoberta.SelfAttention
        let output: XLMRoberta.SelfOutput

        public init(
            selfAttention: XLMRoberta.SelfAttention,
            output: XLMRoberta.SelfOutput
        ) {
            self.selfAttention = selfAttention
            self.output = output
        }

        public func callAsFunction(
            hiddenStates: MLTensor,
            attentionMask: MLTensor?,
            headMask: MLTensor?
        ) -> MLTensor {
            let selfOutputs = selfAttention(
                hiddenStates: hiddenStates,
                attentionMask: attentionMask,
                headMask: headMask
            )
            return output(
                hiddenStates: selfOutputs,
                inputTensor: hiddenStates
            )
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension XLMRoberta {
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
            let hiddenStates = dense(hiddenStates)
            return layerNorm(hiddenStates + inputTensor)
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension XLMRoberta {
    public struct Layer: Sendable {
        let attention: XLMRoberta.Attention
        let intermediate: XLMRoberta.Intermediate
        let output: XLMRoberta.Output

        public init(
            attention: XLMRoberta.Attention,
            intermediate: XLMRoberta.Intermediate,
            output: XLMRoberta.Output
        ) {
            self.attention = attention
            self.intermediate = intermediate
            self.output = output
        }

        public func callAsFunction(
            hiddenStates: MLTensor,
            attentionMask: MLTensor?,
            headMask: MLTensor?
        ) -> MLTensor {
            let attentionOutput = attention(
                hiddenStates: hiddenStates,
                attentionMask: attentionMask,
                headMask: headMask
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
extension XLMRoberta {
    public struct Encoder: Sendable {
        let layers: [XLMRoberta.Layer]

        public init(layers: [XLMRoberta.Layer]) {
            self.layers = layers
        }

        public func callAsFunction(
            hiddenStates: MLTensor,
            attentionMask: MLTensor?,
            headMask: MLTensor?
        ) -> MLTensor {
            var hiddenStates = hiddenStates
            for (index, layer) in layers.enumerated() {
                let layerHeadMask: MLTensor? =
                    if let headMask {
                        headMask[index]
                    } else {
                        nil
                    }
                hiddenStates = layer(
                    hiddenStates: hiddenStates,
                    attentionMask: attentionMask,
                    headMask: layerHeadMask
                )
            }
            return hiddenStates
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension XLMRoberta {
    public struct Model: Sendable {
        let embeddings: XLMRoberta.Embeddings
        let encoder: XLMRoberta.Encoder
        let pooler: XLMRoberta.Pooler?
        let numHiddenLayers: Int

        public init(
            embeddings: XLMRoberta.Embeddings,
            encoder: XLMRoberta.Encoder,
            pooler: XLMRoberta.Pooler?,
            numHiddenLayers: Int
        ) {
            self.embeddings = embeddings
            self.encoder = encoder
            self.pooler = pooler
            self.numHiddenLayers = numHiddenLayers
        }

        private func extendedAttentionMask(_ attentionMask: MLTensor) -> MLTensor {
            let attentionMask: MLTensor =
                if attentionMask.rank == 3 {
                    attentionMask.expandingShape(at: 1)
                } else if attentionMask.rank == 2 {
                    attentionMask.expandingShape(at: 1, 1)
                } else {
                    fatalError("Wrong shape for attentionMask (shape \(attentionMask.shape))")
                }
            return (1.0 - attentionMask) * -10000.0
        }

        public func callAsFunction(
            inputIds: MLTensor,
            tokenTypeIds: MLTensor? = nil,
            attentionMask: MLTensor? = nil,
            positionIds: MLTensor? = nil
        ) -> (sequenceOutput: MLTensor, pooledOutput: MLTensor?) {
            let attentionMask =
                attentionMask
                ?? MLTensor(
                    ones: inputIds.shape,
                    scalarType: Float32.self
                )
            let tokenTypeIds =
                tokenTypeIds
                ?? MLTensor(
                    zeros: inputIds.shape,
                    scalarType: Int32.self
                )
            let headMask = MLTensor(
                repeating: 1 as Int32, shape: [numHiddenLayers])
            let embeddingOutput = embeddings(
                inputIds: inputIds,
                tokenTypeIds: tokenTypeIds,
                positionIds: positionIds
            )
            let encoderOutput = encoder(
                hiddenStates: embeddingOutput,
                attentionMask: extendedAttentionMask(attentionMask),
                headMask: headMask
            )
            let pooledOutput: MLTensor? =
                if let pooler {
                    pooler(hiddenStates: encoderOutput)
                } else {
                    nil
                }
            return (encoderOutput, pooledOutput)
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension XLMRoberta {
    public struct ModelBundle: Sendable {
        public let model: XLMRoberta.Model
        public let tokenizer: any TextTokenizer

        public init(
            model: XLMRoberta.Model,
            tokenizer: any TextTokenizer
        ) {
            self.model = model
            self.tokenizer = tokenizer
        }

        public func encode(_ text: String, maxLength: Int = 128) throws -> MLTensor {
            let tokens = try tokenizer.tokenizeText(text, maxLength: maxLength)
            let inputIds = MLTensor(shape: [1, tokens.count], scalars: tokens)
            let result = model(inputIds: inputIds)
            return result.sequenceOutput[0..., 0, 0...]
        }

        public func batchEncode(
            _ texts: [String],
            padTokenId: Int = 0,
            maxLength: Int = 128
        ) throws -> MLTensor {
            let encodedTexts = try tokenizer.tokenizeTextsPaddingToLongest(
                texts, padTokenId: padTokenId, maxLength: maxLength)
            let inputIds = MLTensor(
                shape: [encodedTexts.count, encodedTexts[0].count],
                scalars: encodedTexts.flatMap { $0 })
            return model(inputIds: inputIds).sequenceOutput[0..., 0, 0...]
        }
    }
}
