import CoreML
import Foundation
import Hub

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
