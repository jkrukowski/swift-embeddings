import CoreML
import Foundation
import Hub
import MLTensorUtils
import Tokenizers
import os

let logger = Logger(
    subsystem: "com.jkrukowski.swift-embeddings",
    category: "Configuration"
)

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

public enum PostProcess: Sendable, Equatable {
    case clsTokenPool(normalize: Bool)
    case meanPool(normalize: Bool)
    case lastTokenPool(normalize: Bool)

    init?(representing poolingMode: String, normalize: Bool) {
        switch poolingMode.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "cls": self = .clsTokenPool(normalize: normalize)
        case "mean": self = .meanPool(normalize: normalize)
        case "lasttoken": self = .lastTokenPool(normalize: normalize)
        default: return nil
        }
    }

    public func withNormalize(_ normalize: Bool) -> PostProcess {
        switch self {
        case .clsTokenPool: .clsTokenPool(normalize: normalize)
        case .meanPool: .meanPool(normalize: normalize)
        case .lastTokenPool: .lastTokenPool(normalize: normalize)
        }
    }
}

struct SentenceTransformersDefaults {
    var postProcess: PostProcess?
    var normalize: Bool?
    var maxLength: Int?
}

private let transformerConfigFilenames = [
    "sentence_bert_config.json",
    "sentence_roberta_config.json",
    "sentence_distilbert_config.json",
    "sentence_camembert_config.json",
    "sentence_albert_config.json",
    "sentence_xlm-roberta_config.json",
    "sentence_xlnet_config.json",
]

private struct ModuleLayout {
    var poolingPath: String?
    var hasPooling = false
    var transformerPath: String?
    var hasNormalize = false
}

func loadSentenceTransformersDefaults(in modelFolder: URL) throws -> SentenceTransformersDefaults? {
    guard let layout = try parseModuleLayout(in: modelFolder) else {
        return nil
    }
    let postProcess = try postProcess(for: layout, in: modelFolder)
    let normalize = (postProcess == nil && layout.hasNormalize) ? true : nil
    let transformerFolder =
        if let transformerPath = layout.transformerPath, !transformerPath.isEmpty {
            modelFolder.appendingPathComponent(transformerPath)
        } else {
            modelFolder
        }
    let maxLength =
        try transformerMaxSeqLength(in: transformerFolder)
        ?? tokenizerModelMaxLength(in: modelFolder)
    return SentenceTransformersDefaults(
        postProcess: postProcess,
        normalize: normalize,
        maxLength: maxLength
    )
}

private func parseModuleLayout(in modelFolder: URL) throws -> ModuleLayout? {
    let modulesUrl = modelFolder.appendingPathComponent("modules.json")
    guard FileManager.default.fileExists(atPath: modulesUrl.path) else {
        return nil
    }
    let modulesData = try Data(contentsOf: modulesUrl)
    guard
        let modules = try JSONSerialization.jsonObject(with: modulesData) as? [[String: Any]]
    else {
        throw EmbeddingsError.invalidFile
    }

    var layout = ModuleLayout()
    for module in modules {
        guard let type = module["type"] as? String else { continue }
        if type.hasSuffix(".Pooling") {
            layout.hasPooling = true
            layout.poolingPath = module["path"] as? String
        } else if type.hasSuffix(".Transformer") {
            layout.transformerPath = module["path"] as? String
        } else if type.hasSuffix(".Normalize") {
            layout.hasNormalize = true
        }
    }

    if layout.hasPooling && layout.poolingPath == nil {
        throw EmbeddingsError.invalidFile
    }
    return layout
}

private func postProcess(for layout: ModuleLayout, in modelFolder: URL) throws -> PostProcess? {
    guard let poolingPath = layout.poolingPath else { return nil }
    let poolingUrl =
        modelFolder
        .appendingPathComponent(poolingPath)
        .appendingPathComponent("config.json")
    let pooling = try loadJSONConfig(at: poolingUrl)
    let postProcess = poolingPostProcess(from: pooling, normalize: layout.hasNormalize)
    if postProcess == nil {
        logger.error(
            """
            Unsupported sentence-transformers pooling at \(poolingUrl.path, privacy: .public); \
            keeping the bundle's default pooling. Embeddings may differ from the upstream model.
            """
        )
    }
    return postProcess
}

private func transformerMaxSeqLength(in transformerFolder: URL) throws -> Int? {
    for filename in transformerConfigFilenames {
        let configUrl = transformerFolder.appendingPathComponent(filename)
        if FileManager.default.fileExists(atPath: configUrl.path) {
            if let maxLength = try loadJSONConfig(at: configUrl)["max_seq_length"] as? Int,
                maxLength > 0
            {
                return maxLength
            }
        }
    }
    return nil
}

private func tokenizerModelMaxLength(in modelFolder: URL) throws -> Int? {
    let tokenizerConfigUrl = modelFolder.appendingPathComponent("tokenizer_config.json")
    guard FileManager.default.fileExists(atPath: tokenizerConfigUrl.path),
        let modelMaxLength = try loadJSONConfig(at: tokenizerConfigUrl)["model_max_length"]
            as? NSNumber
    else {
        return nil
    }
    let maxReasonableSequenceLength: Double = 1_000_000
    let value = modelMaxLength.doubleValue
    guard value > 0, value < maxReasonableSequenceLength else {
        return nil
    }
    return Int(value)
}

private func poolingPostProcess(
    from pooling: [String: Any],
    normalize: Bool
) -> PostProcess? {
    if let mode = pooling["pooling_mode"] as? String {
        return PostProcess(representing: mode, normalize: normalize)
    }

    let booleanModes: [(key: String, postProcess: PostProcess?)] = [
        ("pooling_mode_cls_token", .clsTokenPool(normalize: normalize)),
        ("pooling_mode_mean_tokens", .meanPool(normalize: normalize)),
        ("pooling_mode_lasttoken", .lastTokenPool(normalize: normalize)),
        ("pooling_mode_max_tokens", nil),
        ("pooling_mode_mean_sqrt_len_tokens", nil),
        ("pooling_mode_weightedmean_tokens", nil),
    ]
    let enabled = booleanModes.filter { pooling[$0.key] as? Bool == true }
    guard enabled.count == 1 else { return nil }
    return enabled.first?.postProcess
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
func processResult(
    _ result: MLTensor,
    with postProcess: PostProcess?,
    attentionMask: MLTensor? = nil
) -> MLTensor {
    let pooled: MLTensor
    let normalize: Bool
    switch postProcess {
    case .none:
        return result
    case .clsTokenPool(let shouldNormalize):
        pooled = result[0..., 0, 0...]
        normalize = shouldNormalize
    case .meanPool(let shouldNormalize):
        pooled =
            if let attentionMask {
                maskedMeanPool(result, attentionMask: attentionMask)
            } else {
                result.mean(alongAxes: 1, keepRank: false)
            }
        normalize = shouldNormalize
    case .lastTokenPool(let shouldNormalize):
        pooled = lastTokenPool(result, attentionMask: attentionMask)
        normalize = shouldNormalize
    }
    return normalize ? normalizeEmbeddings(pooled) : pooled
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
func lastTokenPool(_ tensor: MLTensor, attentionMask: MLTensor? = nil) -> MLTensor {
    let seqLen = tensor.shape[1]
    let lastIndex: MLTensor =
        if let attentionMask {
            attentionMask.sum(alongAxes: 1, keepRank: true) - 1.0
        } else {
            MLTensor(repeating: Float(seqLen - 1), shape: [tensor.shape[0], 1])
        }
    let positions = MLTensor(
        rangeFrom: 0.0,
        to: Float(seqLen),
        by: 1.0
    ).expandingShape(at: 0)
    let oneHot = (positions .== lastIndex).cast(to: Float.self)
    let selected = tensor * oneHot.expandingShape(at: 2)
    return selected.sum(alongAxes: 1, keepRank: false)
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
func maskedMeanPool(_ tensor: MLTensor, attentionMask: MLTensor) -> MLTensor {
    let expandedMask = attentionMask.expandingShape(at: 2)
    let maskedTensor = tensor * expandedMask
    let sumPooled = maskedTensor.sum(alongAxes: 1, keepRank: false)
    let tokenCounts = attentionMask.sum(alongAxes: 1, keepRank: true)
    return sumPooled / tokenCounts
}

enum EmbeddingsError: Error {
    case fileNotFound
    case invalidFile
    case configurationNotSupported
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
