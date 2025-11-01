import CoreML
import Foundation
import Hub
import MLTensorUtils
import Safetensors
import Tokenizers

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Roberta {
    public static func loadConfig(at url: URL) throws -> Roberta.ModelConfig {
        try loadConfigFromFile(at: url)
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Roberta {
    public static func loadModelBundle(
        from hubRepoId: String,
        downloadBase: URL? = nil,
        useBackgroundSession: Bool = false,
        loadConfig: LoadConfig = LoadConfig()
    ) async throws -> Roberta.ModelBundle {
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
    ) async throws -> Roberta.ModelBundle {
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: modelFolder,
            tokenizerConfig: loadConfig.tokenizerConfig
        )
        let weightsUrl = modelFolder.appendingPathComponent(loadConfig.modelConfig.weightsFileName)
        let configUrl = modelFolder.appendingPathComponent(loadConfig.modelConfig.configFileName)
        let config = try Roberta.loadConfig(at: configUrl)
        let model = try Roberta.loadModel(
            weightsUrl: weightsUrl,
            config: config,
            loadConfig: loadConfig
        )
        return Roberta.ModelBundle(model: model, tokenizer: TokenizerWrapper(tokenizer))
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Roberta {
    public static func loadModel(
        weightsUrl: URL,
        config: Roberta.ModelConfig,
        loadConfig: LoadConfig = LoadConfig()
    ) throws -> Roberta.Model {
        // NOTE: just `safetensors` support for now
        let safetensors = try Safetensors.read(at: weightsUrl)
        let wordEmbeddings = try MLTensorUtils.embedding(
            weight: safetensors.mlTensor(
                forKey: loadConfig.modelConfig.weightKeyTransform(
                    "embeddings.word_embeddings.weight")))

        let tokenTypeEmbeddings = try MLTensorUtils.embedding(
            weight: safetensors.mlTensor(
                forKey: loadConfig.modelConfig.weightKeyTransform(
                    "embeddings.token_type_embeddings.weight")))

        let positionEmbeddings = try MLTensorUtils.embedding(
            weight: safetensors.mlTensor(
                forKey: loadConfig.modelConfig.weightKeyTransform(
                    "embeddings.position_embeddings.weight")))

        let layerNorm = try MLTensorUtils.layerNorm(
            weight: safetensors.mlTensor(
                forKey: loadConfig.modelConfig.weightKeyTransform(
                    "embeddings.LayerNorm.weight")),
            bias: safetensors.mlTensor(
                forKey: loadConfig.modelConfig.weightKeyTransform(
                    "embeddings.LayerNorm.bias")),
            epsilon: config.layerNormEps)

        let embeddings = Roberta.Embeddings(
            wordEmbeddings: wordEmbeddings,
            positionEmbeddings: positionEmbeddings,
            tokenTypeEmbeddings: tokenTypeEmbeddings,
            layerNorm: layerNorm,
            paddingIdx: Int32(config.padTokenId))

        var layers = [Roberta.Layer]()
        for layer in 0..<config.numHiddenLayers {
            let selfAttention = try Roberta.SelfAttention(
                query: MLTensorUtils.linear(
                    weight: safetensors.mlTensor(
                        forKey: loadConfig.modelConfig.weightKeyTransform(
                            "encoder.layer.\(layer).attention.self.query.weight")),
                    bias: safetensors.mlTensor(
                        forKey: loadConfig.modelConfig.weightKeyTransform(
                            "encoder.layer.\(layer).attention.self.query.bias"))),
                key: MLTensorUtils.linear(
                    weight: safetensors.mlTensor(
                        forKey: loadConfig.modelConfig.weightKeyTransform(
                            "encoder.layer.\(layer).attention.self.key.weight")),
                    bias: safetensors.mlTensor(
                        forKey: loadConfig.modelConfig.weightKeyTransform(
                            "encoder.layer.\(layer).attention.self.key.bias")
                    )),
                value: MLTensorUtils.linear(
                    weight: safetensors.mlTensor(
                        forKey: loadConfig.modelConfig.weightKeyTransform(
                            "encoder.layer.\(layer).attention.self.value.weight")),
                    bias: safetensors.mlTensor(
                        forKey: loadConfig.modelConfig.weightKeyTransform(
                            "encoder.layer.\(layer).attention.self.value.bias"))),
                numAttentionHeads: config.numAttentionHeads,
                attentionHeadSize: config.hiddenSize / config.numAttentionHeads,
                allHeadSize: config.numAttentionHeads
                    * (config.hiddenSize / config.numAttentionHeads)
            )
            let selfOutput = try Roberta.SelfOutput(
                dense: MLTensorUtils.linear(
                    weight: safetensors.mlTensor(
                        forKey: loadConfig.modelConfig.weightKeyTransform(
                            "encoder.layer.\(layer).attention.output.dense.weight")),
                    bias: safetensors.mlTensor(
                        forKey: loadConfig.modelConfig.weightKeyTransform(
                            "encoder.layer.\(layer).attention.output.dense.bias"))),
                layerNorm: MLTensorUtils.layerNorm(
                    weight: safetensors.mlTensor(
                        forKey: loadConfig.modelConfig.weightKeyTransform(
                            "encoder.layer.\(layer).attention.output.LayerNorm.weight")),
                    bias: safetensors.mlTensor(
                        forKey: loadConfig.modelConfig.weightKeyTransform(
                            "encoder.layer.\(layer).attention.output.LayerNorm.bias")),
                    epsilon: config.layerNormEps)
            )
            let attention = Roberta.Attention(
                selfAttention: selfAttention,
                output: selfOutput
            )
            let intermediate = try Roberta.Intermediate(
                dense: MLTensorUtils.linear(
                    weight: safetensors.mlTensor(
                        forKey: loadConfig.modelConfig.weightKeyTransform(
                            "encoder.layer.\(layer).intermediate.dense.weight")),
                    bias: safetensors.mlTensor(
                        forKey: loadConfig.modelConfig.weightKeyTransform(
                            "encoder.layer.\(layer).intermediate.dense.bias")
                    ))
            )
            let output = try Roberta.Output(
                dense: MLTensorUtils.linear(
                    weight: safetensors.mlTensor(
                        forKey: loadConfig.modelConfig.weightKeyTransform(
                            "encoder.layer.\(layer).output.dense.weight")),
                    bias: safetensors.mlTensor(
                        forKey: loadConfig.modelConfig.weightKeyTransform(
                            "encoder.layer.\(layer).output.dense.bias"))),
                layerNorm: MLTensorUtils.layerNorm(
                    weight: safetensors.mlTensor(
                        forKey: loadConfig.modelConfig.weightKeyTransform(
                            "encoder.layer.\(layer).output.LayerNorm.weight")
                    ),
                    bias: safetensors.mlTensor(
                        forKey: loadConfig.modelConfig.weightKeyTransform(
                            "encoder.layer.\(layer).output.LayerNorm.bias")),
                    epsilon: config.layerNormEps))

            let layer = Roberta.Layer(
                attention: attention,
                intermediate: intermediate,
                output: output
            )
            layers.append(layer)
        }
        return Roberta.Model(
            embeddings: embeddings,
            encoder: Roberta.Encoder(layers: layers))
    }
}
