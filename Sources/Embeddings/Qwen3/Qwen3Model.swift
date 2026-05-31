import CoreML
import Foundation
import MLTensorUtils
import Tokenizers

public enum Qwen3 {}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Qwen3 {
    public struct ModelConfig: Codable, Sendable {
        public var hiddenSize: Int
        public var intermediateSize: Int
        public var numHiddenLayers: Int
        public var numAttentionHeads: Int
        public var numKeyValueHeads: Int
        public var headDim: Int
        public var vocabSize: Int
        public var maxPositionEmbeddings: Int
        public var rmsNormEps: Float
        public var ropeTheta: Float
        public var attentionBias: Bool

        public init(from decoder: Swift.Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
            intermediateSize =
                try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3072
            numHiddenLayers =
                try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 28
            numAttentionHeads =
                try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
            numKeyValueHeads =
                try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8
            let decodedHeadDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
            headDim = decodedHeadDim ?? (hiddenSize / numAttentionHeads)
            vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 151_669
            maxPositionEmbeddings =
                try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32_768
            rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
            ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
            attentionBias =
                try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        }

        public init(
            hiddenSize: Int = 1024,
            intermediateSize: Int = 3072,
            numHiddenLayers: Int = 28,
            numAttentionHeads: Int = 16,
            numKeyValueHeads: Int = 8,
            headDim: Int = 128,
            vocabSize: Int = 151_669,
            maxPositionEmbeddings: Int = 32_768,
            rmsNormEps: Float = 1e-6,
            ropeTheta: Float = 1_000_000,
            attentionBias: Bool = false
        ) {
            self.hiddenSize = hiddenSize
            self.intermediateSize = intermediateSize
            self.numHiddenLayers = numHiddenLayers
            self.numAttentionHeads = numAttentionHeads
            self.numKeyValueHeads = numKeyValueHeads
            self.headDim = headDim
            self.vocabSize = vocabSize
            self.maxPositionEmbeddings = maxPositionEmbeddings
            self.rmsNormEps = rmsNormEps
            self.ropeTheta = ropeTheta
            self.attentionBias = attentionBias
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Qwen3 {
    public struct MLP: Sendable {
        let gate: MLTensorUtils.Layer
        let up: MLTensorUtils.Layer
        let down: MLTensorUtils.Layer

        public init(
            gate: @escaping MLTensorUtils.Layer,
            up: @escaping MLTensorUtils.Layer,
            down: @escaping MLTensorUtils.Layer
        ) {
            self.gate = gate
            self.up = up
            self.down = down
        }

        public func callAsFunction(_ x: MLTensor) -> MLTensor {
            down(silu(gate(x)) * up(x))
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Qwen3 {
    public struct Attention: Sendable {
        let qProj: MLTensorUtils.Layer
        let kProj: MLTensorUtils.Layer
        let vProj: MLTensorUtils.Layer
        let oProj: MLTensorUtils.Layer
        let qNorm: MLTensorUtils.Layer
        let kNorm: MLTensorUtils.Layer
        let rotaryEmbeddings: MLTensorUtils.Layer
        let numHeads: Int
        let numKeyValueHeads: Int
        let headDim: Int
        let scale: Float

        public init(
            qProj: @escaping MLTensorUtils.Layer,
            kProj: @escaping MLTensorUtils.Layer,
            vProj: @escaping MLTensorUtils.Layer,
            oProj: @escaping MLTensorUtils.Layer,
            qNorm: @escaping MLTensorUtils.Layer,
            kNorm: @escaping MLTensorUtils.Layer,
            rotaryEmbeddings: @escaping MLTensorUtils.Layer,
            numHeads: Int,
            numKeyValueHeads: Int,
            headDim: Int
        ) {
            self.qProj = qProj
            self.kProj = kProj
            self.vProj = vProj
            self.oProj = oProj
            self.qNorm = qNorm
            self.kNorm = kNorm
            self.rotaryEmbeddings = rotaryEmbeddings
            self.numHeads = numHeads
            self.numKeyValueHeads = numKeyValueHeads
            self.headDim = headDim
            self.scale = pow(Float(headDim), -0.5)
        }

        // Repeats key/value heads to match the number of query heads (grouped-query attention).
        static func repeatKeyValue(_ x: MLTensor, nRep: Int) -> MLTensor {
            guard nRep > 1 else { return x }
            let bs = x.shape[0]
            let kvHeads = x.shape[1]
            let seqLen = x.shape[2]
            let headDim = x.shape[3]
            // Tile each kv head nRep times along a new axis, then merge it into
            // the head axis. Avoids allocating a `ones` tensor and multiplying.
            let expanded = x.expandingShape(at: 2)  // [bs, kvHeads, 1, seq, headDim]
            let tiled = expanded.tiled(multiples: [1, 1, nRep, 1, 1])
            return tiled.reshaped(to: [bs, kvHeads * nRep, seqLen, headDim])
        }

        public func callAsFunction(
            _ hiddenStates: MLTensor,
            attentionMask: MLTensor? = nil
        ) -> MLTensor {
            let bs = hiddenStates.shape[0]
            let seqLen = hiddenStates.shape[1]

            // Project and split into heads. QK-Norm is applied per head over `headDim`.
            var query = qProj(hiddenStates).reshaped(to: [bs, seqLen, numHeads, headDim])
            var key = kProj(hiddenStates).reshaped(to: [bs, seqLen, numKeyValueHeads, headDim])
            var value = vProj(hiddenStates).reshaped(to: [bs, seqLen, numKeyValueHeads, headDim])

            query = qNorm(query).transposed(permutation: [0, 2, 1, 3])
            key = kNorm(key).transposed(permutation: [0, 2, 1, 3])
            value = value.transposed(permutation: [0, 2, 1, 3])

            query = rotaryEmbeddings(query)
            key = rotaryEmbeddings(key)

            let nRep = numHeads / numKeyValueHeads
            key = Self.repeatKeyValue(key, nRep: nRep)
            value = Self.repeatKeyValue(value, nRep: nRep)

            var attentionOutput = sdpa(
                query: query,
                key: key,
                value: value,
                mask: attentionMask,
                scale: scale
            )
            attentionOutput = attentionOutput.transposed(permutation: [0, 2, 1, 3])
            attentionOutput = attentionOutput.reshaped(to: [bs, seqLen, numHeads * headDim])
            return oProj(attentionOutput)
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Qwen3 {
    public struct DecoderLayer: Sendable {
        let inputLayerNorm: MLTensorUtils.Layer
        let attention: Qwen3.Attention
        let postAttentionLayerNorm: MLTensorUtils.Layer
        let mlp: Qwen3.MLP

        public init(
            inputLayerNorm: @escaping MLTensorUtils.Layer,
            attention: Qwen3.Attention,
            postAttentionLayerNorm: @escaping MLTensorUtils.Layer,
            mlp: Qwen3.MLP
        ) {
            self.inputLayerNorm = inputLayerNorm
            self.attention = attention
            self.postAttentionLayerNorm = postAttentionLayerNorm
            self.mlp = mlp
        }

        public func callAsFunction(
            _ hiddenStates: MLTensor,
            attentionMask: MLTensor? = nil
        ) -> MLTensor {
            let attentionOutput = attention(
                inputLayerNorm(hiddenStates),
                attentionMask: attentionMask
            )
            let hs = hiddenStates + attentionOutput
            let mlpOutput = mlp(postAttentionLayerNorm(hs))
            return hs + mlpOutput
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Qwen3 {
    public struct Model: Sendable {
        let embedTokens: MLTensorUtils.Layer
        let layers: [Qwen3.DecoderLayer]
        let norm: MLTensorUtils.Layer

        public init(
            embedTokens: @escaping MLTensorUtils.Layer,
            layers: [Qwen3.DecoderLayer],
            norm: @escaping MLTensorUtils.Layer
        ) {
            self.embedTokens = embedTokens
            self.layers = layers
            self.norm = norm
        }

        // Combines the causal mask with the optional padding attention mask.
        // Returns an additive mask broadcastable to [batch, numHeads, seqLen, seqLen].
        static func makeAttentionMask(
            seqLen: Int,
            attentionMask: MLTensor?
        ) -> MLTensor {
            let causalMask = additiveCausalMask(Int32(seqLen))  // [seqLen, seqLen]
            let causal4D =
                causalMask
                .expandingShape(at: 0)  // [1, seqLen, seqLen]
                .expandingShape(at: 0)  // [1, 1, seqLen, seqLen]
            guard let attentionMask else {
                return causal4D
            }
            // Padding mask: 0 -> -1e9 on the key axis.
            let paddingAdditive = (1.0 - attentionMask) * -1e9  // [batch, seqLen]
            let padding4D =
                paddingAdditive
                .expandingShape(at: 1)  // [batch, 1, seqLen]
                .expandingShape(at: 1)  // [batch, 1, 1, seqLen]
            return causal4D + padding4D
        }

        public func callAsFunction(
            _ inputIds: MLTensor,
            attentionMask: MLTensor? = nil
        ) -> MLTensor {
            let seqLen = inputIds.shape[1]
            let mask = Self.makeAttentionMask(seqLen: seqLen, attentionMask: attentionMask)
            var hiddenStates = embedTokens(inputIds)
            for layer in layers {
                hiddenStates = layer(hiddenStates, attentionMask: mask)
            }
            return norm(hiddenStates)
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Qwen3 {
    public struct ModelBundle: Sendable {
        public let model: Qwen3.Model
        public let tokenizer: any TextTokenizer

        public init(
            model: Qwen3.Model,
            tokenizer: any TextTokenizer
        ) {
            self.model = model
            self.tokenizer = tokenizer
        }

        public func encode(
            _ text: String,
            maxLength: Int = 32_768,
            postProcess: PostProcess? = .lastTokenPoolAndNormalize
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
            padTokenId: Int = 151_643,
            maxLength: Int = 32_768,
            postProcess: PostProcess? = .lastTokenPoolAndNormalize
        ) throws -> MLTensor {
            try withMLTensorComputePolicy(.cpuAndGPU) {
                let batchTokenizeResult = try tokenizer.tokenizeTextsPaddingToLongest(
                    texts, padTokenId: padTokenId, maxLength: maxLength)
                let inputIds = MLTensor(
                    shape: batchTokenizeResult.shape,
                    scalars: batchTokenizeResult.tokens)
                let attentionMask = MLTensor(
                    shape: batchTokenizeResult.shape,
                    scalars: batchTokenizeResult.attentionMask)
                let result = model(inputIds, attentionMask: attentionMask)
                return processResult(result, with: postProcess, attentionMask: attentionMask)
            }
        }
    }
}
