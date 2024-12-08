import CoreML
import Foundation
import Hub
import MLTensorUtils
import Safetensors
import SentencepieceTokenizer
@preconcurrency import Tokenizers

extension XLMRoberta {
    public static func loadConfig(at url: URL) throws -> XLMRoberta.ModelConfig {
        try loadConfigFromFile(at: url)
    }
}

extension XLMRoberta {
    public static func loadModelBundle(
        from hubRepoId: String,
        downloadBase: URL? = nil,
        useBackgroundSession: Bool = false
    ) async throws -> XLMRoberta.ModelBundle {
        let modelFolder = try await downloadModelFromHub(
            from: hubRepoId,
            downloadBase: downloadBase,
            useBackgroundSession: useBackgroundSession
        )
        return try await loadModelBundle(from: modelFolder)
    }

    public static func loadModelBundle(
        from modelFolder: URL
    ) async throws -> XLMRoberta.ModelBundle {
        let addedTokens = try await loadAddedTokens(from: modelFolder)
        let tokenizerModelUrl = try findSentencePieceModel(in: modelFolder)
        let tokenizer = try XLMRobetaTokenizer(
            tokenizerModelUrl: tokenizerModelUrl,
            addedTokens: addedTokens
        )
        // NOTE: just `safetensors` support for now
        let weightsUrl = modelFolder.appendingPathComponent("model.safetensors")
        let configUrl = modelFolder.appendingPathComponent("config.json")
        let config = try XLMRoberta.loadConfig(at: configUrl)
        let model = try XLMRoberta.loadModel(weightsUrl: weightsUrl, config: config)
        return XLMRoberta.ModelBundle(model: model, tokenizer: tokenizer)
    }

    private static func loadAddedTokens(from modelFolder: URL) async throws -> [String: Int] {
        let hubConfiguration = LanguageModelConfigurationFromHub(modelFolder: modelFolder)
        let addedTokens = try await hubConfiguration.tokenizerData.addedTokens?.arrayValue?.map {
            $0.dictionary as [String: Any]
        }
        guard let addedTokens else {
            return [:]
        }
        var result = [String: Int]()
        for addedToken in addedTokens {
            if let content = addedToken["content"] as? String, let id = addedToken["id"] as? Int {
                result[content] = id
            }
        }
        return result
    }

    private static func findSentencePieceModel(in folder: URL) throws -> URL {
        let fileManager = FileManager.default
        let contents = try fileManager.contentsOfDirectory(
            at: folder, includingPropertiesForKeys: nil)
        for url in contents {
            if url.pathExtension == "model", url.lastPathComponent.contains("sentencepiece") {
                return url
            }
        }
        throw EmbeddingsError.fileNotFound
    }
}

extension XLMRoberta {
    public static func loadModel(
        weightsUrl: URL,
        config: XLMRoberta.ModelConfig
    ) throws -> XLMRoberta.Model {
        let safetensors = try Safetensors.read(at: weightsUrl)
        let wordEmbeddings = try MLTensorUtils.embedding(
            weight: safetensors.mlTensor(
                forKey: "embeddings.word_embeddings.weight"))

        let tokenTypeEmbeddings = try MLTensorUtils.embedding(
            weight: safetensors.mlTensor(
                forKey: "embeddings.token_type_embeddings.weight"))

        let positionEmbeddings = try MLTensorUtils.embedding(
            weight: safetensors.mlTensor(
                forKey: "embeddings.position_embeddings.weight"))

        let layerNorm = try MLTensorUtils.layerNorm(
            weight: safetensors.mlTensor(
                forKey: "embeddings.LayerNorm.weight"),
            bias: safetensors.mlTensor(
                forKey: "embeddings.LayerNorm.bias"),
            epsilon: config.layerNormEps)

        let embeddings = XLMRoberta.Embeddings(
            wordEmbeddings: wordEmbeddings,
            positionEmbeddings: positionEmbeddings,
            tokenTypeEmbeddings: tokenTypeEmbeddings,
            layerNorm: layerNorm,
            paddingIndex: Int32(config.padTokenId))

        var layers = [XLMRoberta.Layer]()
        for layer in 0..<config.numHiddenLayers {
            let attentionHeadSize = config.hiddenSize / config.numAttentionHeads
            let allHeadSize =
                config.numAttentionHeads * attentionHeadSize
            let bertSelfAttention = try XLMRoberta.SelfAttention(
                query: MLTensorUtils.linear(
                    weight: safetensors.mlTensor(
                        forKey: "encoder.layer.\(layer).attention.self.query.weight"),
                    bias: safetensors.mlTensor(
                        forKey: "encoder.layer.\(layer).attention.self.query.bias")),
                key: MLTensorUtils.linear(
                    weight: safetensors.mlTensor(
                        forKey: "encoder.layer.\(layer).attention.self.key.weight"),
                    bias: safetensors.mlTensor(
                        forKey: "encoder.layer.\(layer).attention.self.key.bias")),
                value: MLTensorUtils.linear(
                    weight: safetensors.mlTensor(
                        forKey: "encoder.layer.\(layer).attention.self.value.weight"),
                    bias: safetensors.mlTensor(
                        forKey: "encoder.layer.\(layer).attention.self.value.bias")),
                numAttentionHeads: config.numAttentionHeads,
                attentionHeadSize: attentionHeadSize,
                allHeadSize: allHeadSize,
                scale: 1.0 / sqrtf(Float(allHeadSize))
            )
            let bertSelfOutput = try XLMRoberta.SelfOutput(
                dense: MLTensorUtils.linear(
                    weight: safetensors.mlTensor(
                        forKey: "encoder.layer.\(layer).attention.output.dense.weight"),
                    bias: safetensors.mlTensor(
                        forKey: "encoder.layer.\(layer).attention.output.dense.bias")),
                layerNorm: MLTensorUtils.layerNorm(
                    weight: safetensors.mlTensor(
                        forKey: "encoder.layer.\(layer).attention.output.LayerNorm.weight"),
                    bias: safetensors.mlTensor(
                        forKey: "encoder.layer.\(layer).attention.output.LayerNorm.bias"),
                    epsilon: config.layerNormEps)
            )
            let bertAttention = XLMRoberta.Attention(
                selfAttention: bertSelfAttention,
                output: bertSelfOutput
            )
            let bertIntermediate = try XLMRoberta.Intermediate(
                dense: MLTensorUtils.linear(
                    weight: safetensors.mlTensor(
                        forKey: "encoder.layer.\(layer).intermediate.dense.weight"),
                    bias: safetensors.mlTensor(
                        forKey: "encoder.layer.\(layer).intermediate.dense.bias"))
            )
            let bertOutput = try XLMRoberta.Output(
                dense: MLTensorUtils.linear(
                    weight: safetensors.mlTensor(
                        forKey: "encoder.layer.\(layer).output.dense.weight"),
                    bias: safetensors.mlTensor(
                        forKey: "encoder.layer.\(layer).output.dense.bias")),
                layerNorm: MLTensorUtils.layerNorm(
                    weight: safetensors.mlTensor(
                        forKey: "encoder.layer.\(layer).output.LayerNorm.weight"),
                    bias: safetensors.mlTensor(
                        forKey: "encoder.layer.\(layer).output.LayerNorm.bias"),
                    epsilon: config.layerNormEps))

            let bertLayer = XLMRoberta.Layer(
                attention: bertAttention,
                intermediate: bertIntermediate,
                output: bertOutput
            )
            layers.append(bertLayer)
        }
        let pooler: XLMRoberta.Pooler? =
            if let addPoolingLayer = config.addPoolingLayer {
                if addPoolingLayer {
                    try XLMRoberta.Pooler(
                        dense: MLTensorUtils.linear(
                            weight: safetensors.mlTensor(forKey: "pooler.dense.weight"),
                            bias: safetensors.mlTensor(forKey: "pooler.dense.bias")))
                } else {
                    nil
                }
            } else {
                // when value is not provided, default to true
                try XLMRoberta.Pooler(
                    dense: MLTensorUtils.linear(
                        weight: safetensors.mlTensor(forKey: "pooler.dense.weight"),
                        bias: safetensors.mlTensor(forKey: "pooler.dense.bias")))
            }

        return XLMRoberta.Model(
            embeddings: embeddings,
            encoder: XLMRoberta.Encoder(layers: layers),
            pooler: pooler,
            numHiddenLayers: config.numHiddenLayers)
    }
}
