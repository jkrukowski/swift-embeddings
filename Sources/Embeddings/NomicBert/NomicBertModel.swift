import CoreML
import Foundation
import MLTensorUtils
import Tokenizers

public enum NomicBert {}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension NomicBert {
    public enum ComputePolicy: Sendable {
        case cpuOnly
        case cpuAndGPU
    }

    public struct ModelConfig: Codable, Sendable {
        public var modelType: String
        public var nEmbd: Int
        public var nHead: Int
        public var nLayer: Int
        public var nInner: Int?
        public var nPositions: Int
        public var vocabSize: Int
        public var typeVocabSize: Int
        public var layerNormEpsilon: Float
        public var rotaryEmbBase: Float
        public var rotaryEmbFraction: Float
        public var rotaryEmbInterleaved: Bool
        public var qkvProjBias: Bool
        public var mlpFc1Bias: Bool
        public var mlpFc2Bias: Bool
        public var useBias: Bool
        public var prenorm: Bool
        public var useRmsNorm: Bool
        public var activationFunction: String?
        public var maxTrainedPositions: Int?
        public var padTokenId: Int?
        public var bosTokenId: Int?
        public var eosTokenId: Int?

        public init(from decoder: Swift.Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "nomic_bert"
            nEmbd = try container.decodeIfPresent(Int.self, forKey: .nEmbd) ?? 768
            nHead = try container.decodeIfPresent(Int.self, forKey: .nHead) ?? 12
            nLayer = try container.decodeIfPresent(Int.self, forKey: .nLayer) ?? 12
            nInner = try container.decodeIfPresent(Int.self, forKey: .nInner)
            nPositions = try container.decodeIfPresent(Int.self, forKey: .nPositions) ?? 2048
            vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 30522
            typeVocabSize = try container.decodeIfPresent(Int.self, forKey: .typeVocabSize) ?? 2
            layerNormEpsilon = try container.decodeIfPresent(Float.self, forKey: .layerNormEpsilon) ?? 1e-12
            rotaryEmbBase = try container.decodeIfPresent(Float.self, forKey: .rotaryEmbBase) ?? 10_000
            rotaryEmbFraction = try container.decodeIfPresent(Float.self, forKey: .rotaryEmbFraction) ?? 1.0
            rotaryEmbInterleaved = try container.decodeIfPresent(Bool.self, forKey: .rotaryEmbInterleaved) ?? false
            qkvProjBias = try container.decodeIfPresent(Bool.self, forKey: .qkvProjBias) ?? false
            mlpFc1Bias = try container.decodeIfPresent(Bool.self, forKey: .mlpFc1Bias) ?? false
            mlpFc2Bias = try container.decodeIfPresent(Bool.self, forKey: .mlpFc2Bias) ?? false
            useBias = try container.decodeIfPresent(Bool.self, forKey: .useBias) ?? false
            prenorm = try container.decodeIfPresent(Bool.self, forKey: .prenorm) ?? false
            useRmsNorm = try container.decodeIfPresent(Bool.self, forKey: .useRmsNorm) ?? false
            activationFunction =
                try container.decodeIfPresent(String.self, forKey: .activationFunction) ?? "swiglu"
            maxTrainedPositions = try container.decodeIfPresent(Int.self, forKey: .maxTrainedPositions)
            padTokenId = try container.decodeIfPresent(Int.self, forKey: .padTokenId)
            bosTokenId = try container.decodeIfPresent(Int.self, forKey: .bosTokenId)
            eosTokenId = try container.decodeIfPresent(Int.self, forKey: .eosTokenId)
        }

        public init(
            modelType: String = "nomic_bert",
            nEmbd: Int = 768,
            nHead: Int = 12,
            nLayer: Int = 12,
            nInner: Int? = nil,
            nPositions: Int = 2048,
            vocabSize: Int = 30522,
            typeVocabSize: Int = 2,
            layerNormEpsilon: Float = 1e-12,
            rotaryEmbBase: Float = 10_000,
            rotaryEmbFraction: Float = 1.0,
            rotaryEmbInterleaved: Bool = false,
            qkvProjBias: Bool = false,
            mlpFc1Bias: Bool = false,
            mlpFc2Bias: Bool = false,
            useBias: Bool = false,
            prenorm: Bool = false,
            useRmsNorm: Bool = false,
            activationFunction: String? = "swiglu",
            maxTrainedPositions: Int? = nil,
            padTokenId: Int? = nil,
            bosTokenId: Int? = nil,
            eosTokenId: Int? = nil
        ) {
            self.modelType = modelType
            self.nEmbd = nEmbd
            self.nHead = nHead
            self.nLayer = nLayer
            self.nInner = nInner
            self.nPositions = nPositions
            self.vocabSize = vocabSize
            self.typeVocabSize = typeVocabSize
            self.layerNormEpsilon = layerNormEpsilon
            self.rotaryEmbBase = rotaryEmbBase
            self.rotaryEmbFraction = rotaryEmbFraction
            self.rotaryEmbInterleaved = rotaryEmbInterleaved
            self.qkvProjBias = qkvProjBias
            self.mlpFc1Bias = mlpFc1Bias
            self.mlpFc2Bias = mlpFc2Bias
            self.useBias = useBias
            self.prenorm = prenorm
            self.useRmsNorm = useRmsNorm
            self.activationFunction = activationFunction
            self.maxTrainedPositions = maxTrainedPositions
            self.padTokenId = padTokenId
            self.bosTokenId = bosTokenId
            self.eosTokenId = eosTokenId
        }

        public var hiddenSize: Int { nEmbd }
        public var numAttentionHeads: Int { nHead }
        public var numHiddenLayers: Int { nLayer }
        public var intermediateSize: Int { nInner ?? (nEmbd * 4) }
        public var maxPositionEmbeddings: Int { nPositions }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension NomicBert {
    public struct Embeddings: Sendable {
        let wordEmbeddings: MLTensorUtils.Layer
        let positionEmbeddings: MLTensorUtils.Layer?
        let tokenTypeEmbeddings: MLTensorUtils.Layer

        public init(
            wordEmbeddings: @escaping MLTensorUtils.Layer,
            positionEmbeddings: MLTensorUtils.Layer?,
            tokenTypeEmbeddings: @escaping MLTensorUtils.Layer
        ) {
            self.wordEmbeddings = wordEmbeddings
            self.positionEmbeddings = positionEmbeddings
            self.tokenTypeEmbeddings = tokenTypeEmbeddings
        }

        public func callAsFunction(
            inputIds: MLTensor,
            tokenTypeIds: MLTensor? = nil,
            positionIds: MLTensor? = nil
        ) -> MLTensor {
            let seqLength = inputIds.shape[1]
            let tokenTypeIds =
                tokenTypeIds
                ?? MLTensor(
                    zeros: inputIds.shape,
                    scalarType: Int32.self
                )
            var embeddings = wordEmbeddings(inputIds) + tokenTypeEmbeddings(tokenTypeIds)
            if let positionEmbeddings {
                let positionIds =
                    positionIds
                    ?? MLTensor(
                        shape: [1, seqLength],
                        scalars: 0..<Int32(seqLength),
                        scalarType: Int32.self
                    )
                embeddings = embeddings + positionEmbeddings(positionIds)
            }
            return embeddings
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension NomicBert {
    public struct Attention: Sendable {
        let wqkv: MLTensorUtils.Layer
        let wo: MLTensorUtils.Layer
        let rotaryEmbeddings: MLTensorUtils.Layer?
        let numHeads: Int
        let headDim: Int
        let allHeadSize: Int
        let scale: Float

        public init(
            wqkv: @escaping MLTensorUtils.Layer,
            wo: @escaping MLTensorUtils.Layer,
            rotaryEmbeddings: MLTensorUtils.Layer?,
            numHeads: Int,
            headDim: Int,
            scale: Float
        ) {
            self.wqkv = wqkv
            self.wo = wo
            self.rotaryEmbeddings = rotaryEmbeddings
            self.numHeads = numHeads
            self.headDim = headDim
            self.allHeadSize = headDim * numHeads
            self.scale = scale
        }

        public func callAsFunction(
            _ hiddenStates: MLTensor,
            attentionMask: MLTensor? = nil
        ) -> MLTensor {
            var qkv = wqkv(hiddenStates)
            let bs = hiddenStates.shape[0]
            qkv = qkv.reshaped(to: [bs, -1, 3, numHeads, headDim])
            qkv = qkv.transposed(permutation: [0, 3, 2, 1, 4])
            let qkvSplit = qkv.split(count: 3, alongAxis: 2)
            var query = qkvSplit[0].squeezingShape(at: 2)
            var key = qkvSplit[1].squeezingShape(at: 2)
            let value = qkvSplit[2].squeezingShape(at: 2)
            if let rotaryEmbeddings {
                query = rotaryEmbeddings(query)
                key = rotaryEmbeddings(key)
            }
            var attentionOutput = sdpa(
                query: query,
                key: key,
                value: value,
                mask: attentionMask,
                scale: scale
            )
            attentionOutput = attentionOutput.transposed(permutation: [0, 2, 1, 3])
            attentionOutput = attentionOutput.reshaped(to: [bs, -1, allHeadSize])
            return wo(attentionOutput)
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension NomicBert {
    public struct MLP: Sendable {
        let gateUp: MLTensorUtils.Layer
        let down: MLTensorUtils.Layer

        public init(
            gateUp: @escaping MLTensorUtils.Layer,
            down: @escaping MLTensorUtils.Layer
        ) {
            self.gateUp = gateUp
            self.down = down
        }

        public func callAsFunction(_ hiddenStates: MLTensor) -> MLTensor {
            let x = gateUp(hiddenStates)
            let splitDim = x.shape[x.rank - 1] / 2
            let gate = x[0..., 0..., ..<splitDim]
            let up = x[0..., 0..., splitDim...]
            return down(silu(gate) * up)
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension NomicBert {
    public struct Block: Sendable {
        let attentionNorm: MLTensorUtils.Layer
        let attention: NomicBert.Attention
        let mlpNorm: MLTensorUtils.Layer
        let mlp: NomicBert.MLP
        let prenorm: Bool

        public init(
            attentionNorm: @escaping MLTensorUtils.Layer,
            attention: NomicBert.Attention,
            mlpNorm: @escaping MLTensorUtils.Layer,
            mlp: NomicBert.MLP,
            prenorm: Bool
        ) {
            self.attentionNorm = attentionNorm
            self.attention = attention
            self.mlpNorm = mlpNorm
            self.mlp = mlp
            self.prenorm = prenorm
        }

        public func callAsFunction(
            _ hiddenStates: MLTensor,
            attentionMask: MLTensor? = nil
        ) -> MLTensor {
            if prenorm {
                let normalizedHiddenStates = attentionNorm(hiddenStates)
                let attentionOutput = attention(
                    normalizedHiddenStates,
                    attentionMask: attentionMask
                )
                let hs = hiddenStates + attentionOutput
                let mlpOutput = mlp(mlpNorm(hs))
                return hs + mlpOutput
            } else {
                let attentionOutput = attention(
                    hiddenStates,
                    attentionMask: attentionMask
                )
                let hs = attentionNorm(hiddenStates + attentionOutput)
                let mlpOutput = mlp(hs)
                return mlpNorm(hs + mlpOutput)
            }
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension NomicBert {
    public struct Model: Sendable {
        let embeddings: NomicBert.Embeddings
        let embeddingNorm: MLTensorUtils.Layer
        let layers: [NomicBert.Block]
        let prenorm: Bool

        public init(
            embeddings: NomicBert.Embeddings,
            embeddingNorm: @escaping MLTensorUtils.Layer,
            layers: [NomicBert.Block],
            prenorm: Bool
        ) {
            self.embeddings = embeddings
            self.embeddingNorm = embeddingNorm
            self.layers = layers
            self.prenorm = prenorm
        }

        public func callAsFunction(
            inputIds: MLTensor,
            tokenTypeIds: MLTensor? = nil,
            attentionMask: MLTensor? = nil
        ) -> MLTensor {
            var hiddenStates = embeddings(
                inputIds: inputIds,
                tokenTypeIds: tokenTypeIds
            )
            hiddenStates = embeddingNorm(hiddenStates)
            let mask: MLTensor? =
                if let attentionMask {
                    (1.0 - attentionMask.expandingShape(at: 1, 1)) * -10000.0
                } else {
                    nil
                }
            for layer in layers {
                hiddenStates = layer(
                    hiddenStates,
                    attentionMask: mask
                )
            }
            return hiddenStates
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension NomicBert {
    public struct ModelBundle: Sendable {
        public let model: NomicBert.Model
        public let tokenizer: any TextTokenizer

        public init(
            model: NomicBert.Model,
            tokenizer: any TextTokenizer
        ) {
            self.model = model
            self.tokenizer = tokenizer
        }

        public func encode(
            _ text: String,
            maxLength: Int = 2048,
            postProcess: PostProcess? = nil,
            computePolicy: ComputePolicy = .cpuAndGPU
        ) throws -> MLTensor {
            try withMLTensorComputePolicy(
                computePolicy == .cpuOnly ? .cpuOnly : .cpuAndGPU
            ) {
                let tokens = try tokenizer.tokenizeText(text, maxLength: maxLength)
                let inputIds = MLTensor(shape: [1, tokens.count], scalars: tokens)
                let result = model(inputIds: inputIds)
                return processResult(result, with: postProcess)
            }
        }

        public func batchEncode(
            _ texts: [String],
            padTokenId: Int = 0,
            maxLength: Int = 2048,
            postProcess: PostProcess? = nil,
            computePolicy: ComputePolicy = .cpuAndGPU
        ) throws -> MLTensor {
            try withMLTensorComputePolicy(
                computePolicy == .cpuOnly ? .cpuOnly : .cpuAndGPU
            ) {
                let batchTokenizeResult = try tokenizer.tokenizeTextsPaddingToLongest(
                    texts, padTokenId: padTokenId, maxLength: maxLength)
                let inputIds = MLTensor(
                    shape: batchTokenizeResult.shape,
                    scalars: batchTokenizeResult.tokens)
                let attentionMask = MLTensor(
                    shape: batchTokenizeResult.shape,
                    scalars: batchTokenizeResult.attentionMask)
                let result = model(
                    inputIds: inputIds,
                    attentionMask: attentionMask
                )
                return processResult(result, with: postProcess, attentionMask: attentionMask)
            }
        }
    }
}
