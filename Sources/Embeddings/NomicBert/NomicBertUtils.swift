import CoreML
import Foundation
import Hub
import MLTensorUtils
import Safetensors
import Tokenizers

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension NomicBert {
    public static func loadConfig(at url: URL) throws -> NomicBert.ModelConfig {
        try loadConfigFromFile(at: url)
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension NomicBert {
    public static func loadModelBundle(
        from hubRepoId: String,
        downloadBase: URL? = nil,
        useBackgroundSession: Bool = false,
        loadConfig: LoadConfig = LoadConfig()
    ) async throws -> NomicBert.ModelBundle {
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
    ) async throws -> NomicBert.ModelBundle {
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: modelFolder,
            tokenizerConfig: loadConfig.tokenizerConfig
        )
        let weightsUrl = modelFolder.appendingPathComponent(loadConfig.modelConfig.weightsFileName)
        let configUrl = modelFolder.appendingPathComponent(loadConfig.modelConfig.configFileName)
        let config = try NomicBert.loadConfig(at: configUrl)
        let model = try NomicBert.loadModel(
            weightsUrl: weightsUrl,
            config: config,
            loadConfig: loadConfig
        )
        return NomicBert.ModelBundle(
            model: model,
            tokenizer: TokenizerWrapper(tokenizer)
        )
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension NomicBert {
    public static func loadModel(
        weightsUrl: URL,
        config: NomicBert.ModelConfig,
        loadConfig: LoadConfig = LoadConfig()
    ) throws -> NomicBert.Model {
        if config.useRmsNorm {
            throw EmbeddingsError.configurationNotSupported
        }
        if let activationFunction = config.activationFunction,
            activationFunction.lowercased() != "swiglu"
        {
            throw EmbeddingsError.configurationNotSupported
        }
        if config.rotaryEmbInterleaved {
            throw EmbeddingsError.configurationNotSupported
        }

        let safetensors = try Safetensors.read(at: weightsUrl)
        let wordEmbeddings = try MLTensorUtils.embedding(
            weight: safetensors.mlTensor(
                forKey: loadConfig.modelConfig.weightKeyTransform(
                    "embeddings.word_embeddings.weight")))

        let tokenTypeEmbeddings = try MLTensorUtils.embedding(
            weight: safetensors.mlTensor(
                forKey: loadConfig.modelConfig.weightKeyTransform(
                    "embeddings.token_type_embeddings.weight")))

        let positionEmbeddings: MLTensorUtils.Layer? =
            if config.rotaryEmbFraction <= 0 {
                try MLTensorUtils.embedding(
                    weight: safetensors.mlTensor(
                        forKey: loadConfig.modelConfig.weightKeyTransform(
                            "embeddings.position_embeddings.weight")))
            } else {
                nil
            }

        let embeddings = NomicBert.Embeddings(
            wordEmbeddings: wordEmbeddings,
            positionEmbeddings: positionEmbeddings,
            tokenTypeEmbeddings: tokenTypeEmbeddings)

        let embNorm = try MLTensorUtils.layerNorm(
            weight: safetensors.mlTensor(
                forKey: loadConfig.modelConfig.weightKeyTransform("emb_ln.weight")),
            bias: safetensors.mlTensor(
                forKey: loadConfig.modelConfig.weightKeyTransform("emb_ln.bias")),
            epsilon: config.layerNormEpsilon)

        let headDim = config.hiddenSize / config.numAttentionHeads
        let rotaryDim = max(0, Int(Float(headDim) * config.rotaryEmbFraction))
        let ropeDim = (rotaryDim / 2) * 2
        let rotaryEmbeddings: MLTensorUtils.Layer? =
            if ropeDim > 0 {
                MLTensorUtils.roPE(dims: ropeDim, base: Int(config.rotaryEmbBase))
            } else {
                nil
            }

        var layers = [NomicBert.Block]()
        layers.reserveCapacity(config.numHiddenLayers)
        for layerId in 0..<config.numHiddenLayers {
            let wqkvWeight = try safetensors.mlTensor(
                forKey: loadConfig.modelConfig.weightKeyTransform(
                    "encoder.layers.\(layerId).attn.Wqkv.weight"))
            let wqkv: MLTensorUtils.Layer =
                if config.qkvProjBias {
                    try MLTensorUtils.linear(
                        weight: wqkvWeight,
                        bias: safetensors.mlTensor(
                            forKey: loadConfig.modelConfig.weightKeyTransform(
                                "encoder.layers.\(layerId).attn.Wqkv.bias")))
                } else {
                    MLTensorUtils.linear(weight: wqkvWeight)
                }

            let woWeight = try safetensors.mlTensor(
                forKey: loadConfig.modelConfig.weightKeyTransform(
                    "encoder.layers.\(layerId).attn.out_proj.weight"))
            let wo: MLTensorUtils.Layer =
                if config.qkvProjBias {
                    try MLTensorUtils.linear(
                        weight: woWeight,
                        bias: safetensors.mlTensor(
                            forKey: loadConfig.modelConfig.weightKeyTransform(
                                "encoder.layers.\(layerId).attn.out_proj.bias")))
                } else {
                    MLTensorUtils.linear(weight: woWeight)
                }

            let attention = NomicBert.Attention(
                wqkv: wqkv,
                wo: wo,
                rotaryEmbeddings: rotaryEmbeddings,
                numHeads: config.numAttentionHeads,
                headDim: headDim,
                scale: pow(Float(headDim), -0.5)
            )

            let norm1 = try MLTensorUtils.layerNorm(
                weight: safetensors.mlTensor(
                    forKey: loadConfig.modelConfig.weightKeyTransform(
                        "encoder.layers.\(layerId).norm1.weight")),
                bias: safetensors.mlTensor(
                    forKey: loadConfig.modelConfig.weightKeyTransform(
                        "encoder.layers.\(layerId).norm1.bias")),
                epsilon: config.layerNormEpsilon)

            let norm2 = try MLTensorUtils.layerNorm(
                weight: safetensors.mlTensor(
                    forKey: loadConfig.modelConfig.weightKeyTransform(
                        "encoder.layers.\(layerId).norm2.weight")),
                bias: safetensors.mlTensor(
                    forKey: loadConfig.modelConfig.weightKeyTransform(
                        "encoder.layers.\(layerId).norm2.bias")),
                epsilon: config.layerNormEpsilon)

            let gateUpBias: MLTensor? =
                if config.mlpFc1Bias {
                    try? safetensors.mlTensor(
                        forKey: loadConfig.modelConfig.weightKeyTransform(
                            "encoder.layers.\(layerId).mlp.gate_up_proj.bias"))
                } else {
                    nil
                }
            let gateUpWeightResolved: MLTensor
            let gateUpBiasResolved: MLTensor?
            if let combined = try? safetensors.mlTensor(
                forKey: loadConfig.modelConfig.weightKeyTransform(
                    "encoder.layers.\(layerId).mlp.gate_up_proj.weight"))
            {
                gateUpWeightResolved = combined
                gateUpBiasResolved = gateUpBias
            } else {
                guard
                    let fc11 = try? safetensors.mlTensor(
                        forKey: loadConfig.modelConfig.weightKeyTransform(
                            "encoder.layers.\(layerId).mlp.fc11.weight")),
                    let fc12 = try? safetensors.mlTensor(
                        forKey: loadConfig.modelConfig.weightKeyTransform(
                            "encoder.layers.\(layerId).mlp.fc12.weight"))
                else {
                    throw EmbeddingsError.invalidFile
                }
                gateUpWeightResolved = MLTensor(concatenating: [fc12, fc11], alongAxis: 0)
                gateUpBiasResolved = nil
            }
            let gateUp: MLTensorUtils.Layer =
                if let gateUpBiasResolved {
                    MLTensorUtils.linear(weight: gateUpWeightResolved, bias: gateUpBiasResolved)
                } else {
                    MLTensorUtils.linear(weight: gateUpWeightResolved)
                }

            let downWeight: MLTensor
            if let weight = try? safetensors.mlTensor(
                forKey: loadConfig.modelConfig.weightKeyTransform(
                    "encoder.layers.\(layerId).mlp.down_proj.weight"))
            {
                downWeight = weight
            } else if let weight = try? safetensors.mlTensor(
                forKey: loadConfig.modelConfig.weightKeyTransform(
                    "encoder.layers.\(layerId).mlp.fc2.weight"))
            {
                downWeight = weight
            } else {
                throw EmbeddingsError.invalidFile
            }
            let down: MLTensorUtils.Layer =
                if config.mlpFc2Bias {
                    try MLTensorUtils.linear(
                        weight: downWeight,
                        bias: safetensors.mlTensor(
                            forKey: loadConfig.modelConfig.weightKeyTransform(
                                "encoder.layers.\(layerId).mlp.down_proj.bias")))
                } else {
                    MLTensorUtils.linear(weight: downWeight)
                }

            let mlp = NomicBert.MLP(
                gateUp: gateUp,
                down: down
            )

            let layer = NomicBert.Block(
                attentionNorm: norm1,
                attention: attention,
                mlpNorm: norm2,
                mlp: mlp,
                prenorm: config.prenorm
            )
            layers.append(layer)
        }

        return NomicBert.Model(
            embeddings: embeddings,
            embeddingNorm: embNorm,
            layers: layers,
            prenorm: config.prenorm
        )
    }
}
