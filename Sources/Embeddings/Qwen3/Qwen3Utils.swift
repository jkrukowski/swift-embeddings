import CoreML
import Foundation
import Hub
import MLTensorUtils
import Safetensors
import Tokenizers

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Qwen3 {
    public static func loadConfig(at url: URL) throws -> Qwen3.ModelConfig {
        try loadConfigFromFile(at: url)
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Qwen3 {
    public static func loadModelBundle(
        from hubRepoId: String,
        downloadBase: URL? = nil,
        useBackgroundSession: Bool = false,
        loadConfig: LoadConfig = LoadConfig()
    ) async throws -> Qwen3.ModelBundle {
        let modelFolder = try await downloadModelFromHub(
            from: hubRepoId,
            downloadBase: downloadBase,
            useBackgroundSession: useBackgroundSession
        )
        return try await loadModelBundle(
            from: modelFolder,
            loadConfig: loadConfig
        )
    }

    public static func loadModelBundle(
        from modelFolder: URL,
        loadConfig: LoadConfig = LoadConfig()
    ) async throws -> Qwen3.ModelBundle {
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: modelFolder,
            tokenizerConfig: loadConfig.tokenizerConfig
        )
        let weightsUrl = modelFolder.appendingPathComponent(loadConfig.modelConfig.weightsFileName)
        let configUrl = modelFolder.appendingPathComponent(loadConfig.modelConfig.configFileName)
        let config = try Qwen3.loadConfig(at: configUrl)
        let model = try await Qwen3.loadModel(
            weightsUrl: weightsUrl,
            config: config,
            loadConfig: loadConfig
        )
        return Qwen3.ModelBundle(
            model: model,
            tokenizer: TokenizerWrapper(tokenizer)
        )
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Qwen3 {
    public static func loadModel(
        weightsUrl: URL,
        config: Qwen3.ModelConfig,
        loadConfig: LoadConfig = LoadConfig()
    ) async throws -> Qwen3.Model {
        let safetensors = try Safetensors.read(at: weightsUrl)

        func weight(_ key: String) throws -> MLTensor {
            try safetensors.mlTensor(forKey: loadConfig.modelConfig.weightKeyTransform(key))
        }

        func projection(_ key: String) throws -> MLTensorUtils.Layer {
            if config.attentionBias {
                try MLTensorUtils.linear(
                    weight: weight("\(key).weight"),
                    bias: weight("\(key).bias"))
            } else {
                MLTensorUtils.linear(weight: try weight("\(key).weight"))
            }
        }

        let embedTokens = try MLTensorUtils.embedding(
            weight: weight("embed_tokens.weight"))

        var decoderLayers = [Qwen3.DecoderLayer]()
        for layerId in 0..<config.numHiddenLayers {
            let prefix = "layers.\(layerId)"

            let inputLayerNorm = MLTensorUtils.rmsNorm(
                weight: try weight("\(prefix).input_layernorm.weight"),
                epsilon: config.rmsNormEps)

            let qNorm = MLTensorUtils.rmsNorm(
                weight: try weight("\(prefix).self_attn.q_norm.weight"),
                epsilon: config.rmsNormEps)
            let kNorm = MLTensorUtils.rmsNorm(
                weight: try weight("\(prefix).self_attn.k_norm.weight"),
                epsilon: config.rmsNormEps)

            let rotaryEmbeddings = MLTensorUtils.roPE(
                dims: config.headDim, base: Int(config.ropeTheta))

            let attention = Qwen3.Attention(
                qProj: try projection("\(prefix).self_attn.q_proj"),
                kProj: try projection("\(prefix).self_attn.k_proj"),
                vProj: try projection("\(prefix).self_attn.v_proj"),
                oProj: try projection("\(prefix).self_attn.o_proj"),
                qNorm: qNorm,
                kNorm: kNorm,
                rotaryEmbeddings: rotaryEmbeddings,
                numHeads: config.numAttentionHeads,
                numKeyValueHeads: config.numKeyValueHeads,
                headDim: config.headDim)

            let postAttentionLayerNorm = MLTensorUtils.rmsNorm(
                weight: try weight("\(prefix).post_attention_layernorm.weight"),
                epsilon: config.rmsNormEps)

            let mlp = Qwen3.MLP(
                gate: MLTensorUtils.linear(weight: try weight("\(prefix).mlp.gate_proj.weight")),
                up: MLTensorUtils.linear(weight: try weight("\(prefix).mlp.up_proj.weight")),
                down: MLTensorUtils.linear(weight: try weight("\(prefix).mlp.down_proj.weight")))

            let decoderLayer = Qwen3.DecoderLayer(
                inputLayerNorm: inputLayerNorm,
                attention: attention,
                postAttentionLayerNorm: postAttentionLayerNorm,
                mlp: mlp)

            decoderLayers.append(decoderLayer)
        }

        let norm = MLTensorUtils.rmsNorm(
            weight: try weight("norm.weight"),
            epsilon: config.rmsNormEps)

        return Qwen3.Model(
            embedTokens: embedTokens,
            layers: decoderLayers,
            norm: norm)
    }
}
