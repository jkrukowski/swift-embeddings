import CoreML
import Foundation
import Hub
import MLTensorUtils
import Safetensors
@preconcurrency import Tokenizers

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Bert {
    public static func loadConfig(at url: URL) throws -> Bert.ModelConfig {
        try loadConfigFromFile(at: url)
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Bert {
    public static func loadModelBundle(
        from hubRepoId: String,
        downloadBase: URL? = nil,
        useBackgroundSession: Bool = false,
        loadConfig: LoadConfig = LoadConfig()
    ) async throws -> Bert.ModelBundle {
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
    ) async throws -> Bert.ModelBundle {
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: modelFolder,
            tokenizerConfig: loadConfig.tokenizerConfig
        )
        let weightsUrl = modelFolder.appendingPathComponent(loadConfig.modelConfig.weightsFileName)
        let configUrl = modelFolder.appendingPathComponent(loadConfig.modelConfig.configFileName)
        let config = try Bert.loadConfig(at: configUrl)
        let model = try Bert.loadModel(
            weightsUrl: weightsUrl,
            config: config,
            loadConfig: loadConfig
        )
        return Bert.ModelBundle(model: model, tokenizer: TokenizerWrapper(tokenizer))
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Bert {
    // NOTE: this is a simple key transformation that is required for the Google BERT weights.
    // Model available here: [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)
    public static func googleWeightsKeyTransform(_ key: String) -> String {
        "bert.\(key)"
            .replace(suffix: ".LayerNorm.weight", with: ".LayerNorm.gamma")
            .replace(suffix: ".LayerNorm.bias", with: ".LayerNorm.beta")
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Bert {
    public static func loadModel(
        weightsUrl: URL,
        config: Bert.ModelConfig,
        loadConfig: LoadConfig = LoadConfig()
    ) throws -> Bert.Model {
        // NOTE: just `safetensors` support for now
        let safetensors = try Safetensors.read(at: weightsUrl)
        let pooler = try Bert.Pooler(
            dense: MLTensorUtils.linear(
                weight: safetensors.mlTensor(
                    forKey: loadConfig.modelConfig.weightKeyTransform("pooler.dense.weight")),
                bias: safetensors.mlTensor(
                    forKey: loadConfig.modelConfig.weightKeyTransform("pooler.dense.bias"))))

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
                forKey: loadConfig.modelConfig.weightKeyTransform("embeddings.LayerNorm.weight")),
            bias: safetensors.mlTensor(
                forKey: loadConfig.modelConfig.weightKeyTransform("embeddings.LayerNorm.bias")),
            epsilon: config.layerNormEps)

        let embeddings = Bert.Embeddings(
            wordEmbeddings: wordEmbeddings,
            positionEmbeddings: positionEmbeddings,
            tokenTypeEmbeddings: tokenTypeEmbeddings,
            layerNorm: layerNorm)

        var layers = [Bert.Layer]()
        for layer in 0..<config.numHiddenLayers {
            let bertSelfAttention = try Bert.SelfAttention(
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
            let bertSelfOutput = try Bert.SelfOutput(
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
            let bertAttention = Bert.Attention(
                selfAttention: bertSelfAttention,
                output: bertSelfOutput
            )
            let bertIntermediate = try Bert.Intermediate(
                dense: MLTensorUtils.linear(
                    weight: safetensors.mlTensor(
                        forKey: loadConfig.modelConfig.weightKeyTransform(
                            "encoder.layer.\(layer).intermediate.dense.weight")),
                    bias: safetensors.mlTensor(
                        forKey: loadConfig.modelConfig.weightKeyTransform(
                            "encoder.layer.\(layer).intermediate.dense.bias")
                    ))
            )
            let bertOutput = try Bert.Output(
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

            let bertLayer = Bert.Layer(
                attention: bertAttention,
                intermediate: bertIntermediate,
                output: bertOutput
            )
            layers.append(bertLayer)
        }
        return Bert.Model(
            embeddings: embeddings,
            encoder: Bert.Encoder(layers: layers),
            pooler: pooler)
    }
}
