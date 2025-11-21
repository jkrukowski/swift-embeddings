import CoreML
import Foundation
import Hub
import MLTensorUtils
import Tokenizers

extension AutoTokenizer {
    static func from(
        modelFolder: URL,
        tokenizerConfig: TokenizerConfig?
    ) async throws -> any Tokenizer {
        if let tokenizerConfig {
            try AutoTokenizer.from(
                tokenizerConfig: resolveConfig(tokenizerConfig.config, in: modelFolder),
                tokenizerData: resolveConfig(tokenizerConfig.data, in: modelFolder)
            )
        } else {
            try await AutoTokenizer.from(modelFolder: modelFolder)
        }
    }
}

func resolveConfig(_ tokenizerConfig: TokenizerConfigType, in modelFolder: URL) throws -> Config {
    switch tokenizerConfig {
    case .filePath(let filePath):
        let fileURL = modelFolder.appendingPathComponent(filePath)
        let data = try loadJSONConfig(at: fileURL)
        return Config(data as [NSString: Any])
    case .data(let data):
        return Config(data as [NSString: Any])
    }
}

func loadJSONConfig(at filePath: URL) throws -> [String: Any] {
    let data = try Data(contentsOf: filePath)
    let parsedData = try JSONSerialization.jsonObject(with: data, options: [])
    guard let config = parsedData as? [String: Any] else {
        throw EmbeddingsError.invalidFile
    }
    return config
}

@discardableResult
func downloadModelFromHub(
    from hubRepoId: String,
    downloadBase: URL? = nil,
    useBackgroundSession: Bool = false,
    globs: [String] = Constants.modelGlobs
) async throws -> URL {
    let hubApi = HubApi(downloadBase: downloadBase, useBackgroundSession: useBackgroundSession)
    let repo = Hub.Repo(id: hubRepoId, type: .models)
    return try await hubApi.snapshot(
        from: repo,
        matching: globs
    )
}

public enum PostProcess {
    case meanPool
    case meanPoolAndNormalize
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
func processResult(
    _ result: MLTensor, with postProcess: PostProcess?, attentionMask: MLTensor? = nil
) -> MLTensor {
    switch postProcess {
    case .none:
        return result
    case .meanPool:
        let pooled: MLTensor =
            if let attentionMask {
                maskedMeanPool(result, attentionMask: attentionMask)
            } else {
                result.mean(alongAxes: 1, keepRank: false)
            }
        return pooled
    case .meanPoolAndNormalize:
        let pooled: MLTensor =
            if let attentionMask {
                maskedMeanPool(result, attentionMask: attentionMask)
            } else {
                result.mean(alongAxes: 1, keepRank: false)
            }
        return normalizeEmbeddings(pooled)
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
func maskedMeanPool(_ tensor: MLTensor, attentionMask: MLTensor) -> MLTensor {
    // tensor shape: [batch, seq_len, hidden_dim], attentionMask shape: [batch, seq_len]
    // Expand attention mask to match tensor dimensions: [batch, seq_len, 1]
    let expandedMask = attentionMask.expandingShape(at: 2)
    let maskedTensor = tensor * expandedMask
    let sumPooled = maskedTensor.sum(alongAxes: 1, keepRank: false)
    let tokenCounts = attentionMask.sum(alongAxes: 1, keepRank: true)
    return sumPooled / tokenCounts
}

enum EmbeddingsError: Error {
    case fileNotFound
    case invalidFile
}

enum Constants {
    static let modelGlobs = [
        "*.json",
        "*.safetensors",
        "*.py",
        "tokenizer.model",
        "sentencepiece*.model",
        "*.tiktoken",
        "*.txt",
    ]
}

func loadConfigFromFile<Config: Codable>(at url: URL) throws -> Config {
    let configData = try Data(contentsOf: url)
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(Config.self, from: configData)
}

extension String {
    func replace(suffix: String, with string: String) -> String {
        guard hasSuffix(suffix) else { return self }
        return String(dropLast(suffix.count) + string)
    }
}

public enum TokenizerConfigType {
    case filePath(String)
    case data([String: Any])
}

public struct TokenizerConfig {
    public let data: TokenizerConfigType
    public let config: TokenizerConfigType

    public init(
        data: TokenizerConfigType = .filePath("tokenizer.json"),
        config: TokenizerConfigType = .filePath("tokenizer_config.json")
    ) {
        self.data = data
        self.config = config
    }
}

public struct ModelConfig {
    public let configFileName: String
    public let weightsFileName: String
    public let weightKeyTransform: ((String) -> String)

    public init(
        configFileName: String = "config.json",
        weightsFileName: String = "model.safetensors",
        weightKeyTransform: @escaping ((String) -> String) = { $0 }
    ) {
        self.configFileName = configFileName
        self.weightsFileName = weightsFileName
        self.weightKeyTransform = weightKeyTransform
    }
}

public struct LoadConfig {
    public let modelConfig: ModelConfig
    public let tokenizerConfig: TokenizerConfig?

    public init(
        modelConfig: ModelConfig = ModelConfig(),
        tokenizerConfig: TokenizerConfig? = nil
    ) {
        self.modelConfig = modelConfig
        self.tokenizerConfig = tokenizerConfig
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension LoadConfig {
    public static var googleBert: LoadConfig {
        LoadConfig(
            modelConfig: ModelConfig(
                weightKeyTransform: Bert.googleWeightsKeyTransform
            )
        )
    }

    public static func addWeightKeyPrefix(_ prefix: String) -> LoadConfig {
        LoadConfig(
            modelConfig: ModelConfig(
                weightKeyTransform: { key in
                    "\(prefix)\(key)"
                }
            )
        )
    }

    public static var staticEmbeddings: LoadConfig {
        LoadConfig(
            modelConfig: ModelConfig(
                weightsFileName: "0_StaticEmbedding/model.safetensors"
            ),
            // In case of `StaticEmbeddings` tokenizer `data` is loaded from `0_StaticEmbedding/tokenizer.json` file
            // and tokenizer `config` is a dictionary with a single key `tokenizerClass` and value `BertTokenizer`.
            tokenizerConfig: TokenizerConfig(
                data: .filePath("0_StaticEmbedding/tokenizer.json"),
                config: .data(["tokenizerClass": "BertTokenizer"])
            )
        )
    }
}
