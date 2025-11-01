import CoreML
import Foundation
import Hub
import MLTensorUtils
import Safetensors
import Tokenizers

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension ModernBert {
    public static func loadConfig(at url: URL) throws -> ModernBert.ModelConfig {
        try loadConfigFromFile(at: url)
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension ModernBert {
    public static func loadModelBundle(
        from hubRepoId: String,
        downloadBase: URL? = nil,
        useBackgroundSession: Bool = false,
        loadConfig: LoadConfig = LoadConfig()
    ) async throws -> ModernBert.ModelBundle {
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
    ) async throws -> ModernBert.ModelBundle {
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: modelFolder,
            tokenizerConfig: loadConfig.tokenizerConfig
        )
        let weightsUrl = modelFolder.appendingPathComponent(loadConfig.modelConfig.weightsFileName)
        let configUrl = modelFolder.appendingPathComponent(loadConfig.modelConfig.configFileName)
        let config = try ModernBert.loadConfig(at: configUrl)
        let model = try await ModernBert.loadModel(
            weightsUrl: weightsUrl,
            config: config,
            loadConfig: loadConfig
        )
        return ModernBert.ModelBundle(
            model: model,
            tokenizer: TokenizerWrapper(tokenizer)
        )
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension ModernBert {
    public static func loadModel(
        weightsUrl: URL,
        config: ModernBert.ModelConfig,
        loadConfig: LoadConfig = LoadConfig()
    ) async throws -> ModernBert.Model {
        let safetensors = try Safetensors.read(at: weightsUrl)
        let tokenEmbeddings = try MLTensorUtils.embedding(
            weight: safetensors.mlTensor(
                forKey: loadConfig.modelConfig.weightKeyTransform(
                    "embeddings.tok_embeddings.weight")))

        let embeddingsLayerNormWeight = try safetensors.mlTensor(
            forKey: loadConfig.modelConfig.weightKeyTransform(
                "embeddings.norm.weight"))
        let embeddingsLayerNormBias: MLTensor =
            if config.normBias {
                try safetensors.mlTensor(
                    forKey: loadConfig.modelConfig.weightKeyTransform(
                        "embeddings.norm.bias"))
            } else {
                MLTensor(zeros: embeddingsLayerNormWeight.shape, scalarType: Float.self)
            }
        let embeddingsLayerNorm = MLTensorUtils.layerNorm(
            weight: embeddingsLayerNormWeight,
            bias: embeddingsLayerNormBias,
            epsilon: config.layerNormEps)

        let embeddings = ModernBert.Embeddings(
            tokenEmbeddings: tokenEmbeddings,
            layerNorm: embeddingsLayerNorm)

        var encoderLayers = [ModernBert.Encoder]()
        for layerId in 0..<config.numHiddenLayers {
            let attentionNorm: MLTensorUtils.Layer
            if layerId == 0 {
                attentionNorm = identity()
            } else {
                let attnNormWeight = try safetensors.mlTensor(
                    forKey: loadConfig.modelConfig.weightKeyTransform(
                        "layers.\(layerId).attn_norm.weight"))
                let attnNormBias: MLTensor =
                    if config.normBias {
                        try safetensors.mlTensor(
                            forKey: loadConfig.modelConfig.weightKeyTransform(
                                "layers.\(layerId).attn_norm.bias"))
                    } else {
                        MLTensor(zeros: attnNormWeight.shape, scalarType: Float.self)
                    }
                attentionNorm = MLTensorUtils.layerNorm(
                    weight: attnNormWeight,
                    bias: attnNormBias,
                    epsilon: config.layerNormEps)
            }

            let wqkvWeight = try safetensors.mlTensor(
                forKey: loadConfig.modelConfig.weightKeyTransform(
                    "layers.\(layerId).attn.Wqkv.weight"))
            let wqkv: MLTensorUtils.Layer =
                if config.attentionBias {
                    try MLTensorUtils.linear(
                        weight: wqkvWeight,
                        bias: safetensors.mlTensor(
                            forKey: loadConfig.modelConfig.weightKeyTransform(
                                "layers.\(layerId).attn.Wqkv.bias")))
                } else {
                    MLTensorUtils.linear(weight: wqkvWeight)
                }

            let woWeight = try safetensors.mlTensor(
                forKey: loadConfig.modelConfig.weightKeyTransform(
                    "layers.\(layerId).attn.Wo.weight"))
            let wo: MLTensorUtils.Layer =
                if config.attentionBias {
                    try MLTensorUtils.linear(
                        weight: woWeight,
                        bias: safetensors.mlTensor(
                            forKey: loadConfig.modelConfig.weightKeyTransform(
                                "layers.\(layerId).attn.Wo.bias")))
                } else {
                    MLTensorUtils.linear(weight: woWeight)
                }

            let isLocalAttention = (layerId % config.globalAttnEveryNLayers) != 0

            let ropeTheta = isLocalAttention ? config.localRopeTheta : config.globalRopeTheta
            let headDim = config.hiddenSize / config.numAttentionHeads
            let rotaryEmbeddings = MLTensorUtils.roPE(dims: headDim, base: Int(ropeTheta))

            let attention = ModernBert.Attention(
                wqkv: wqkv,
                wo: wo,
                rotaryEmbeddings: rotaryEmbeddings,
                numHeads: config.numAttentionHeads,
                headDim: headDim,
                isLocalAttention: isLocalAttention)

            let mlpNormWeight = try safetensors.mlTensor(
                forKey: loadConfig.modelConfig.weightKeyTransform(
                    "layers.\(layerId).mlp_norm.weight"))
            let mlpNormBias: MLTensor =
                if config.normBias {
                    try safetensors.mlTensor(
                        forKey: loadConfig.modelConfig.weightKeyTransform(
                            "layers.\(layerId).mlp_norm.bias"))
                } else {
                    MLTensor(zeros: mlpNormWeight.shape, scalarType: Float.self)
                }
            let mlpNorm = MLTensorUtils.layerNorm(
                weight: mlpNormWeight,
                bias: mlpNormBias,
                epsilon: config.layerNormEps)

            let wiWeight = try safetensors.mlTensor(
                forKey: loadConfig.modelConfig.weightKeyTransform(
                    "layers.\(layerId).mlp.Wi.weight"))
            let wi: MLTensorUtils.Layer =
                if config.mlpBias {
                    try MLTensorUtils.linear(
                        weight: wiWeight,
                        bias: safetensors.mlTensor(
                            forKey: loadConfig.modelConfig.weightKeyTransform(
                                "layers.\(layerId).mlp.Wi.bias")))
                } else {
                    MLTensorUtils.linear(weight: wiWeight)
                }

            let woMlpWeight = try safetensors.mlTensor(
                forKey: loadConfig.modelConfig.weightKeyTransform(
                    "layers.\(layerId).mlp.Wo.weight"))
            let woMlp: MLTensorUtils.Layer =
                if config.mlpBias {
                    try MLTensorUtils.linear(
                        weight: woMlpWeight,
                        bias: safetensors.mlTensor(
                            forKey: loadConfig.modelConfig.weightKeyTransform(
                                "layers.\(layerId).mlp.Wo.bias")))
                } else {
                    MLTensorUtils.linear(weight: woMlpWeight)
                }

            let mlp = ModernBert.MLP(
                wi: wi,
                wo: woMlp)

            let encoder = ModernBert.Encoder(
                attentionNorm: attentionNorm,
                attention: attention,
                mlpNorm: mlpNorm,
                mlp: mlp)

            encoderLayers.append(encoder)
        }

        let finalNormWeight = try safetensors.mlTensor(
            forKey: loadConfig.modelConfig.weightKeyTransform(
                "final_norm.weight"))
        let finalNormBias: MLTensor =
            if config.normBias {
                try safetensors.mlTensor(
                    forKey: loadConfig.modelConfig.weightKeyTransform(
                        "final_norm.bias"))
            } else {
                MLTensor(zeros: finalNormWeight.shape, scalarType: Float.self)
            }
        let finalNorm = MLTensorUtils.layerNorm(
            weight: finalNormWeight,
            bias: finalNormBias,
            epsilon: config.layerNormEps)

        return ModernBert.Model(
            embeddings: embeddings,
            layers: encoderLayers,
            finalNorm: finalNorm,
            localAttention: config.localAttention ?? 128)
    }
}
