import CoreML
import Foundation
import MLTensorUtils
import Tokenizers

public enum Bert {}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Bert {
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

        public init(
            modelType: String,
            numHiddenLayers: Int,
            numAttentionHeads: Int,
            hiddenSize: Int,
            intermediateSize: Int,
            maxPositionEmbeddings: Int,
            hiddenDropoutProb: Float = 0.1,
            attentionProbsDropoutProb: Float = 0.1,
            typeVocabSize: Int = 2,
            initializerRange: Float = 0.02,
            layerNormEps: Float = 1e-12,
            vocabSize: Int = 30522
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
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Bert {
    public struct Pooler: Sendable {
        let dense: MLTensorUtils.Layer

        public init(dense: @escaping MLTensorUtils.Layer) {
            self.dense = dense
        }

        public func callAsFunction(_ hiddenStates: MLTensor) -> MLTensor {
            let firstTokenTensor = hiddenStates[0..., 0]
            let pooledOutput = dense(firstTokenTensor)
            return pooledOutput.tanh()
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Bert {
    public struct Embeddings: Sendable {
        let wordEmbeddings: MLTensorUtils.Layer
        let positionEmbeddings: MLTensorUtils.Layer
        let tokenTypeEmbeddings: MLTensorUtils.Layer
        let layerNorm: MLTensorUtils.Layer

        public init(
            wordEmbeddings: @escaping MLTensorUtils.Layer,
            positionEmbeddings: @escaping MLTensorUtils.Layer,
            tokenTypeEmbeddings: @escaping MLTensorUtils.Layer,
            layerNorm: @escaping MLTensorUtils.Layer
        ) {
            self.wordEmbeddings = wordEmbeddings
            self.positionEmbeddings = positionEmbeddings
            self.tokenTypeEmbeddings = tokenTypeEmbeddings
            self.layerNorm = layerNorm
        }

        public func callAsFunction(
            inputIds: MLTensor,
            tokenTypeIds: MLTensor? = nil,
            positionIds: MLTensor? = nil
        ) -> MLTensor {
            let seqLength = inputIds.shape[1]
            let positionIds =
                positionIds
                ?? MLTensor(
                    shape: [1, seqLength],
                    scalars: 0..<Int32(seqLength),
                    scalarType: Int32.self
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
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Bert {
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
extension Bert {
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
extension Bert {
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
extension Bert {
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
extension Bert {
    public struct Attention: Sendable {
        let selfAttention: Bert.SelfAttention
        let output: Bert.SelfOutput

        public init(
            selfAttention: Bert.SelfAttention,
            output: Bert.SelfOutput
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
extension Bert {
    public struct Layer: Sendable {
        let attention: Bert.Attention
        let intermediate: Bert.Intermediate
        let output: Bert.Output

        public init(
            attention: Bert.Attention,
            intermediate: Bert.Intermediate,
            output: Bert.Output
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
extension Bert {
    public struct Encoder: Sendable {
        let layers: [Bert.Layer]

        public init(layers: [Bert.Layer]) {
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
extension Bert {
    public struct Model: Sendable {
        let embeddings: Bert.Embeddings
        let encoder: Bert.Encoder
        let pooler: Bert.Pooler

        public init(
            embeddings: Bert.Embeddings,
            encoder: Bert.Encoder,
            pooler: Bert.Pooler
        ) {
            self.embeddings = embeddings
            self.encoder = encoder
            self.pooler = pooler
        }

        public func callAsFunction(
            inputIds: MLTensor,
            tokenTypeIds: MLTensor? = nil,
            attentionMask: MLTensor? = nil
        ) -> (sequenceOutput: MLTensor, pooledOutput: MLTensor) {
            let embeddingOutput = embeddings(inputIds: inputIds, tokenTypeIds: tokenTypeIds)
            let mask: MLTensor? =
                if let attentionMask {
                    (1.0 - attentionMask.expandingShape(at: 1, 1)) * -10000.0
                } else {
                    nil
                }
            let encoderOutput = encoder(hiddenStates: embeddingOutput, attentionMask: mask)
            let pooledOutput = pooler(encoderOutput)
            return (encoderOutput, pooledOutput)
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Bert {
    public struct ModelBundle: Sendable {
        public let model: Bert.Model
        public let tokenizer: any TextTokenizer

        public init(
            model: Bert.Model,
            tokenizer: any TextTokenizer
        ) {
            self.model = model
            self.tokenizer = tokenizer
        }

        public func encode(
            _ text: String,
            maxLength: Int = 512
        ) throws -> MLTensor {
            let tokens = try tokenizer.tokenizeText(text, maxLength: maxLength)
            let inputIds = MLTensor(shape: [1, tokens.count], scalars: tokens)
            let result = model(inputIds: inputIds)
            return result.sequenceOutput[0..., 0, 0...]
        }

        public func batchEncode(
            _ texts: [String],
            padTokenId: Int = 0,
            maxLength: Int = 512
        ) throws -> MLTensor {
            let encodedTexts = try tokenizer.tokenizeTextsPaddingToLongest(
                texts, padTokenId: padTokenId, maxLength: maxLength)

            let batchSize = encodedTexts.count
            let seqLength = encodedTexts[0].count

            let inputIds = MLTensor(
                shape: [batchSize, seqLength],
                scalars: encodedTexts.flatMap { $0 })

            // Create attention mask: 1 for real tokens, 0 for padding
            var attentionMaskScalars: [Float] = []
            for tokens in encodedTexts {
                for token in tokens {
                    attentionMaskScalars.append(token == padTokenId ? 0.0 : 1.0)
                }
            }
            let attentionMask = MLTensor(
                shape: [batchSize, seqLength],
                scalars: attentionMaskScalars,
                scalarType: Float.self)

            let result = model(inputIds: inputIds, attentionMask: attentionMask)
            return result.sequenceOutput[0..., 0, 0...]
        }
    }
}
