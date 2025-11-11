import CoreML
import Foundation
import MLTensorUtils
import Tokenizers

public enum ModernBert {}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension ModernBert {
    public struct ModelConfig: Codable, Sendable {
        public var numHiddenLayers: Int
        public var numAttentionHeads: Int
        public var hiddenSize: Int
        public var intermediateSize: Int
        public var maxPositionEmbeddings: Int
        public var layerNormEps: Float
        public var vocabSize: Int
        public var padTokenId: Int
        public var bosTokenId: Int
        public var eosTokenId: Int
        public var clsTokenId: Int
        public var sepTokenId: Int
        public var globalRopeTheta: Float
        public var localRopeTheta: Float
        public var globalAttnEveryNLayers: Int
        public var localAttention: Int?
        public var normBias: Bool
        public var attentionBias: Bool
        public var mlpBias: Bool

        public init(
            numHiddenLayers: Int = 22,
            numAttentionHeads: Int = 12,
            hiddenSize: Int = 768,
            intermediateSize: Int = 1152,
            maxPositionEmbeddings: Int = 8192,
            layerNormEps: Float = 1e-05,
            vocabSize: Int = 50368,
            padTokenId: Int = 50283,
            bosTokenId: Int = 50281,
            eosTokenId: Int = 50282,
            clsTokenId: Int = 50281,
            sepTokenId: Int = 50282,
            globalRopeTheta: Float = 160000.0,
            localRopeTheta: Float = 10000.0,
            globalAttnEveryNLayers: Int = 3,
            localAttention: Int? = 128,
            normBias: Bool = false,
            attentionBias: Bool = false,
            mlpBias: Bool = false
        ) {
            self.numHiddenLayers = numHiddenLayers
            self.numAttentionHeads = numAttentionHeads
            self.hiddenSize = hiddenSize
            self.intermediateSize = intermediateSize
            self.maxPositionEmbeddings = maxPositionEmbeddings
            self.layerNormEps = layerNormEps
            self.vocabSize = vocabSize
            self.padTokenId = padTokenId
            self.bosTokenId = bosTokenId
            self.eosTokenId = eosTokenId
            self.clsTokenId = clsTokenId
            self.sepTokenId = sepTokenId
            self.globalRopeTheta = globalRopeTheta
            self.localRopeTheta = localRopeTheta
            self.globalAttnEveryNLayers = globalAttnEveryNLayers
            self.localAttention = localAttention
            self.normBias = normBias
            self.attentionBias = attentionBias
            self.mlpBias = mlpBias
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension ModernBert {
    public struct Embeddings: Sendable {
        let tokenEmbeddings: MLTensorUtils.Layer
        let layerNorm: MLTensorUtils.Layer

        public init(
            tokenEmbeddings: @escaping MLTensorUtils.Layer,
            layerNorm: @escaping MLTensorUtils.Layer
        ) {
            self.tokenEmbeddings = tokenEmbeddings
            self.layerNorm = layerNorm
        }

        public func callAsFunction(_ inputIds: MLTensor) -> MLTensor {
            layerNorm(tokenEmbeddings(inputIds))
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension ModernBert {
    public struct MLP: Sendable {
        let wi: MLTensorUtils.Layer
        let wo: MLTensorUtils.Layer

        public init(
            wi: @escaping MLTensorUtils.Layer,
            wo: @escaping MLTensorUtils.Layer
        ) {
            self.wi = wi
            self.wo = wo
        }

        public func callAsFunction(_ inputIds: MLTensor) -> MLTensor {
            let x = wi(inputIds)
            let splitDim = x.shape[x.rank - 1] / 2
            let input = x[0..., 0..., ..<splitDim]
            let gate = x[0..., 0..., splitDim...]
            return wo(gelu(input) * gate)
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension ModernBert {
    public struct Attention: Sendable {
        let wqkv: MLTensorUtils.Layer
        let wo: MLTensorUtils.Layer
        let rotaryEmbeddings: MLTensorUtils.Layer
        let numHeads: Int
        let headDim: Int
        let allHeadSize: Int
        let scale: Float
        let isLocalAttention: Bool

        public init(
            wqkv: @escaping MLTensorUtils.Layer,
            wo: @escaping MLTensorUtils.Layer,
            rotaryEmbeddings: @escaping MLTensorUtils.Layer,
            numHeads: Int,
            headDim: Int,
            isLocalAttention: Bool
        ) {
            self.wqkv = wqkv
            self.wo = wo
            self.rotaryEmbeddings = rotaryEmbeddings
            self.numHeads = numHeads
            self.headDim = headDim
            self.allHeadSize = headDim * numHeads
            self.scale = pow(Float(headDim), -0.5)
            self.isLocalAttention = isLocalAttention
        }

        public func callAsFunction(
            _ hiddenStates: MLTensor,
            attentionMask: MLTensor? = nil,
            slidingWindowMask: MLTensor? = nil,
            positionIds: MLTensor? = nil
        ) -> MLTensor {
            var qkv = wqkv(hiddenStates)
            let bs = hiddenStates.shape[0]
            qkv = qkv.reshaped(to: [bs, -1, 3, numHeads, headDim])
            qkv = qkv.transposed(permutation: [0, 3, 2, 1, 4])
            let qkvSplit = qkv.split(count: 3, alongAxis: 2)
            var query = qkvSplit[0].squeezingShape(at: 2)
            var key = qkvSplit[1].squeezingShape(at: 2)
            let value = qkvSplit[2].squeezingShape(at: 2)
            query = rotaryEmbeddings(query)
            key = rotaryEmbeddings(key)
            var attentionOutput = sdpa(
                query: query,
                key: key,
                value: value,
                mask: isLocalAttention ? slidingWindowMask : attentionMask,
                scale: scale
            )
            attentionOutput = attentionOutput.transposed(permutation: [0, 2, 1, 3])
            attentionOutput = attentionOutput.reshaped(to: [bs, -1, allHeadSize])
            return wo(attentionOutput)
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension ModernBert {
    public struct Encoder: Sendable {
        let attentionNorm: MLTensorUtils.Layer
        let attention: ModernBert.Attention
        let mlpNorm: MLTensorUtils.Layer
        let mlp: ModernBert.MLP

        public init(
            attentionNorm: @escaping MLTensorUtils.Layer,
            attention: ModernBert.Attention,
            mlpNorm: @escaping MLTensorUtils.Layer,
            mlp: ModernBert.MLP
        ) {
            self.attentionNorm = attentionNorm
            self.attention = attention
            self.mlpNorm = mlpNorm
            self.mlp = mlp
        }

        public func callAsFunction(
            _ hiddenStates: MLTensor,
            attentionMask: MLTensor? = nil,
            slidingWindowMask: MLTensor? = nil,
            positionIds: MLTensor? = nil
        ) -> MLTensor {
            let normalizedHiddenStates = attentionNorm(hiddenStates)
            let attentionOutput = attention(
                normalizedHiddenStates,
                attentionMask: attentionMask,
                slidingWindowMask: slidingWindowMask,
                positionIds: positionIds
            )
            let hs = hiddenStates + attentionOutput
            let mlpOutput = mlp(mlpNorm(hs))
            return hs + mlpOutput
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension ModernBert {
    public struct PredictionHead: Sendable {
        let dense: MLTensorUtils.Layer
        let activation: MLTensorUtils.Layer
        let norm: MLTensorUtils.Layer

        public init(
            dense: @escaping MLTensorUtils.Layer,
            activation: @escaping MLTensorUtils.Layer,
            norm: @escaping MLTensorUtils.Layer
        ) {
            self.dense = dense
            self.activation = activation
            self.norm = norm
        }

        public func callAsFunction(_ hiddenStates: MLTensor) -> MLTensor {
            norm(activation(dense(hiddenStates)))
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension ModernBert {
    public struct Model: Sendable {
        let embeddings: ModernBert.Embeddings
        let layers: [ModernBert.Encoder]
        let finalNorm: MLTensorUtils.Layer
        let localAttention: Int

        public init(
            embeddings: ModernBert.Embeddings,
            layers: [ModernBert.Encoder],
            finalNorm: @escaping MLTensorUtils.Layer,
            localAttention: Int
        ) {
            self.embeddings = embeddings
            self.layers = layers
            self.finalNorm = finalNorm
            self.localAttention = localAttention
        }

        static func updateAttentionMask(
            _ attentionMask: MLTensor,
            localAttention: Int
        ) -> (MLTensor, MLTensor) {
            let batchSize = attentionMask.shape[0]
            let seqLen = attentionMask.shape[1]
            let halfWindow = Float(localAttention / 2)
            let positions = MLTensor(
                rangeFrom: 0.0,
                to: Float(seqLen),
                by: 1.0
            )
            let rows = positions.expandingShape(at: 0)  // [seqLen, 1]
            let distanceMatrix = (rows - rows.transposed()).abs()

            let windowMask = distanceMatrix .<= halfWindow
            let maskValue = MLTensor(repeating: -1e9, shape: [batchSize, seqLen])
            let zeroValue = MLTensor(zeros: [batchSize, seqLen], scalarType: Float.self)
            let additiveMask2D = zeroValue.replacing(with: maskValue, where: attentionMask .== 0.0)

            let additiveMask4D =
                additiveMask2D
                .expandingShape(at: 1)  // [batchSize, 1, seqLen]
                .expandingShape(at: 2)  // [batchSize, 1, 1, seqLen]

            // Broadcast to full shape by adding with zero tensor
            let broadcastZero = MLTensor(
                zeros: [batchSize, 1, seqLen, seqLen], scalarType: Float.self)
            let globalAttentionMask = broadcastZero + additiveMask4D

            // Create sliding window mask
            // Start with -1e9 everywhere (outside window)
            let slidingMask2D = MLTensor(repeating: -1e9, shape: [seqLen, seqLen])
                .replacing(with: 0.0, where: windowMask)

            let slidingMask4D =
                slidingMask2D
                .expandingShape(at: 0)  // [1, seqLen, seqLen]
                .expandingShape(at: 1)  // [1, 1, seqLen, seqLen]

            let broadcastSliding = broadcastZero + slidingMask4D
            let slidingWindowMask = pointwiseMin(globalAttentionMask, broadcastSliding)
            return (globalAttentionMask, slidingWindowMask)
        }

        public func callAsFunction(
            _ inputIds: MLTensor,
            attentionMask: MLTensor? = nil,
            slidingWindowMask: MLTensor? = nil,
            positionIds: MLTensor? = nil
        ) -> MLTensor {
            let batchSize = inputIds.shape[0]
            let seqLen = inputIds.shape[1]
            let inputAttentionMask =
                attentionMask ?? MLTensor(ones: [batchSize, seqLen], scalarType: Float.self)
            let (globalMask, windowMask) = Self.updateAttentionMask(
                inputAttentionMask,
                localAttention: localAttention
            )
            var hiddenStates = embeddings(inputIds)
            for encoder in layers {
                hiddenStates = encoder(
                    hiddenStates,
                    attentionMask: globalMask,
                    slidingWindowMask: windowMask,
                    positionIds: positionIds
                )
            }
            return finalNorm(hiddenStates)
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension ModernBert {
    public struct ModelBundle: Sendable {
        public let model: ModernBert.Model
        public let tokenizer: any TextTokenizer

        public init(
            model: ModernBert.Model,
            tokenizer: any TextTokenizer
        ) {
            self.model = model
            self.tokenizer = tokenizer
        }

        public func encode(
            _ text: String,
            maxLength: Int = 8192,
            postProcess: PostProcess? = nil
        ) throws -> MLTensor {
            try withMLTensorComputePolicy(.cpuAndGPU) {
                let tokens = try tokenizer.tokenizeText(text, maxLength: maxLength)
                let inputIds = MLTensor(shape: [1, tokens.count], scalars: tokens)
                let result = model(inputIds)
                return processResult(result, with: postProcess)
            }
        }

        public func batchEncode(
            _ texts: [String],
            padTokenId: Int = 50283,
            maxLength: Int = 8192,
            postProcess: PostProcess? = nil
        ) throws -> MLTensor {
            try withMLTensorComputePolicy(.cpuAndGPU) {
                let encodedTexts = try tokenizer.tokenizeTextsPaddingToLongest(
                    texts, padTokenId: padTokenId, maxLength: maxLength)
                let inputIds = MLTensor(
                    shape: [encodedTexts.count, encodedTexts[0].count],
                    scalars: encodedTexts.flatMap { $0 })
                let result = model(inputIds)
                return processResult(result, with: postProcess)
            }
        }
    }
}
