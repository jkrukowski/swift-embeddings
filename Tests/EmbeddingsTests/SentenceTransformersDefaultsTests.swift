import Foundation
import Testing

@testable import Embeddings

@Suite struct SentenceTransformersDefaultsTests {
    private func makeModel(
        modules: String,
        poolingConfig: String?,
        sentenceBertConfig: String? = nil,
        transformerConfigFilename: String = "sentence_bert_config.json",
        tokenizerConfig: String? = nil
    ) throws -> URL {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("st-defaults-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try modules.write(
            to: folder.appendingPathComponent("modules.json"),
            atomically: true,
            encoding: .utf8
        )
        if let poolingConfig {
            let pooling = folder.appendingPathComponent("1_Pooling")
            try FileManager.default.createDirectory(at: pooling, withIntermediateDirectories: true)
            try poolingConfig.write(
                to: pooling.appendingPathComponent("config.json"),
                atomically: true,
                encoding: .utf8
            )
        }
        if let sentenceBertConfig {
            try sentenceBertConfig.write(
                to: folder.appendingPathComponent(transformerConfigFilename),
                atomically: true,
                encoding: .utf8
            )
        }
        if let tokenizerConfig {
            try tokenizerConfig.write(
                to: folder.appendingPathComponent("tokenizer_config.json"),
                atomically: true,
                encoding: .utf8
            )
        }
        return folder
    }

    private let poolingAndNormalizeModules = """
        [
          {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
          {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
          {"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"}
        ]
        """

    private let poolingOnlyModules = """
        [
          {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
          {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"}
        ]
        """

    @Test func legacyBooleanMeanWithNormalize() throws {
        let folder = try makeModel(
            modules: poolingAndNormalizeModules,
            poolingConfig: #"{"pooling_mode_cls_token": false, "pooling_mode_mean_tokens": true}"#,
            sentenceBertConfig: #"{"max_seq_length": 8192}"#
        )
        let defaults = try loadSentenceTransformersDefaults(in: folder)
        #expect(defaults?.postProcess == .meanPool(normalize: true))
        #expect(defaults?.maxLength == 8192)
    }

    @Test func newStringFormatMeanNoNormalize() throws {
        let folder = try makeModel(
            modules: poolingOnlyModules,
            poolingConfig: #"{"pooling_mode": "mean", "include_prompt": true}"#
        )
        let defaults = try loadSentenceTransformersDefaults(in: folder)
        #expect(defaults?.postProcess == .meanPool(normalize: false))
        #expect(defaults?.maxLength == nil)
    }

    @Test func newStringFormatLastTokenWithNormalize() throws {
        let folder = try makeModel(
            modules: poolingAndNormalizeModules,
            poolingConfig: #"{"pooling_mode": "lasttoken"}"#
        )
        let defaults = try loadSentenceTransformersDefaults(in: folder)
        #expect(defaults?.postProcess == .lastTokenPool(normalize: true))
    }

    @Test func clsStringFormat() throws {
        let folder = try makeModel(
            modules: poolingOnlyModules,
            poolingConfig: #"{"pooling_mode": "cls"}"#
        )
        #expect(
            try loadSentenceTransformersDefaults(in: folder)?.postProcess
                == .clsTokenPool(normalize: false)
        )
    }

    @Test func unsupportedModeIsTolerated() throws {
        // A recognized-but-unrepresentable mode leaves `postProcess` nil so the bundle keeps its
        // own default pooling and the model still loads.
        let folder = try makeModel(
            modules: poolingOnlyModules,
            poolingConfig: #"{"pooling_mode": "weightedmean"}"#
        )
        let defaults = try loadSentenceTransformersDefaults(in: folder)
        #expect(defaults != nil)
        #expect(defaults?.postProcess == nil)
    }

    @Test func unsupportedModeWithNormalizePreservesNormalizeRequest() throws {
        // The declared pooling mode is unrepresentable, so `postProcess` is nil, but the layout
        // still has a `.Normalize` module. The normalize request must be recorded so it can be
        // applied on top of the bundle's default pooling rather than silently dropped.
        let folder = try makeModel(
            modules: poolingAndNormalizeModules,
            poolingConfig: #"{"pooling_mode_max_tokens": true}"#
        )
        let defaults = try loadSentenceTransformersDefaults(in: folder)
        #expect(defaults?.postProcess == nil)
        #expect(defaults?.normalize == true)
    }

    @Test func poolingModeStringIsCaseAndWhitespaceInsensitive() throws {
        let folder = try makeModel(
            modules: poolingOnlyModules,
            poolingConfig: #"{"pooling_mode": "  Mean  "}"#
        )
        #expect(
            try loadSentenceTransformersDefaults(in: folder)?.postProcess
                == .meanPool(normalize: false)
        )
    }

    @Test func multipleEnabledBooleanModesAreTolerated() throws {
        // Concatenated pooling (two modes true) cannot be represented and must not silently
        // collapse to one of them; it is tolerated by falling back to the bundle's default.
        let folder = try makeModel(
            modules: poolingOnlyModules,
            poolingConfig:
                #"{"pooling_mode_cls_token": true, "pooling_mode_mean_tokens": true}"#
        )
        let defaults = try loadSentenceTransformersDefaults(in: folder)
        #expect(defaults != nil)
        #expect(defaults?.postProcess == nil)
    }

    @Test func singleUnsupportedBooleanModeIsTolerated() throws {
        let folder = try makeModel(
            modules: poolingOnlyModules,
            poolingConfig:
                #"{"pooling_mode_mean_tokens": false, "pooling_mode_max_tokens": true}"#
        )
        let defaults = try loadSentenceTransformersDefaults(in: folder)
        #expect(defaults != nil)
        #expect(defaults?.postProcess == nil)
    }

    @Test func noEnabledBooleanModeIsTolerated() throws {
        let folder = try makeModel(
            modules: poolingOnlyModules,
            poolingConfig: #"{"pooling_mode_mean_tokens": false}"#
        )
        let defaults = try loadSentenceTransformersDefaults(in: folder)
        #expect(defaults != nil)
        #expect(defaults?.postProcess == nil)
    }

    @Test func poolingModuleWithoutPathThrows() throws {
        let modules = """
            [
              {"idx": 0, "name": "0", "type": "sentence_transformers.models.Transformer"},
              {"idx": 1, "name": "1", "type": "sentence_transformers.models.Pooling"}
            ]
            """
        let folder = try makeModel(modules: modules, poolingConfig: nil)
        #expect(throws: (any Error).self) {
            try loadSentenceTransformersDefaults(in: folder)
        }
    }

    @Test func maxLengthFromLegacyTransformerConfigFilename() throws {
        let folder = try makeModel(
            modules: poolingOnlyModules,
            poolingConfig: #"{"pooling_mode": "mean"}"#,
            sentenceBertConfig: #"{"max_seq_length": 514}"#,
            transformerConfigFilename: "sentence_xlm-roberta_config.json"
        )
        #expect(try loadSentenceTransformersDefaults(in: folder)?.maxLength == 514)
    }

    @Test func maxLengthFallsBackToTokenizerModelMaxLength() throws {
        let folder = try makeModel(
            modules: poolingOnlyModules,
            poolingConfig: #"{"pooling_mode": "mean"}"#,
            tokenizerConfig: #"{"model_max_length": 512}"#
        )
        #expect(try loadSentenceTransformersDefaults(in: folder)?.maxLength == 512)
    }

    @Test func explicitMaxSeqLengthPreferredOverTokenizer() throws {
        let folder = try makeModel(
            modules: poolingOnlyModules,
            poolingConfig: #"{"pooling_mode": "mean"}"#,
            sentenceBertConfig: #"{"max_seq_length": 8192}"#,
            tokenizerConfig: #"{"model_max_length": 512}"#
        )
        #expect(try loadSentenceTransformersDefaults(in: folder)?.maxLength == 8192)
    }

    @Test func tokenizerSentinelModelMaxLengthIgnored() throws {
        let folder = try makeModel(
            modules: poolingOnlyModules,
            poolingConfig: #"{"pooling_mode": "mean"}"#,
            tokenizerConfig: #"{"model_max_length": 1000000000000000000}"#
        )
        #expect(try loadSentenceTransformersDefaults(in: folder)?.maxLength == nil)
    }

    @Test func postProcessFromPoolingMode() throws {
        #expect(
            PostProcess(representing: "cls", normalize: false) == .clsTokenPool(normalize: false)
        )
        #expect(PostProcess(representing: "mean", normalize: true) == .meanPool(normalize: true))
        #expect(
            PostProcess(representing: "lasttoken", normalize: true)
                == .lastTokenPool(normalize: true)
        )
        #expect(PostProcess(representing: "max", normalize: false) == nil)
        #expect(PostProcess(representing: "weightedmean", normalize: false) == nil)
    }

    // MARK: - Real model layouts (config files copied from the actual Hugging Face repos)

    private func fixtureFolder(_ name: String) throws -> URL {
        let url = try #require(
            Bundle.module.url(
                forResource: "modules",
                withExtension: "json",
                subdirectory: "Resources/SentenceTransformersModels/\(name)"
            ),
            "Missing fixture for \(name)"
        )
        return url.deletingLastPathComponent()
    }

    @Test func realNomicEmbedTextV15() throws {
        let defaults = try loadSentenceTransformersDefaults(
            in: try fixtureFolder("nomic-embed-text-v1.5")
        )
        #expect(defaults?.postProcess == .meanPool(normalize: false))
        #expect(defaults?.maxLength == 8192)
    }

    @Test func realModernBertEmbedBase() throws {
        let defaults = try loadSentenceTransformersDefaults(
            in: try fixtureFolder("modernbert-embed-base")
        )
        #expect(defaults?.postProcess == .meanPool(normalize: true))
        #expect(defaults?.maxLength == 8192)
    }

    @Test func realQwen3Embedding() throws {
        // No `sentence_*_config.json`; max length comes from the tokenizer's `model_max_length`.
        let defaults = try loadSentenceTransformersDefaults(
            in: try fixtureFolder("Qwen3-Embedding-0.6B-F32")
        )
        #expect(defaults?.postProcess == .lastTokenPool(normalize: true))
        #expect(defaults?.maxLength == 131072)
    }

    @Test func notASentenceTransformersModel() throws {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("st-defaults-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        #expect(try loadSentenceTransformersDefaults(in: folder) == nil)
    }
}
