import CoreML
import Foundation
import Hub
import MLTensorUtils
import Safetensors
import Tokenizers

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension StaticEmbeddings {
    public static func loadModelBundle(
        from hubRepoId: String,
        downloadBase: URL? = nil,
        useBackgroundSession: Bool = false,
        loadConfig: LoadConfig = LoadConfig()
    ) async throws -> StaticEmbeddings.ModelBundle {
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
    ) async throws -> StaticEmbeddings.ModelBundle {
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: modelFolder,
            tokenizerConfig: loadConfig.tokenizerConfig
        )
        let weightsUrl = modelFolder.appendingPathComponent(loadConfig.modelConfig.weightsFileName)
        let model = try StaticEmbeddings.loadModel(
            weightsUrl: weightsUrl,
            loadConfig: loadConfig
        )
        return StaticEmbeddings.ModelBundle(
            model: model,
            tokenizer: TokenizerWrapper(tokenizer)
        )
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension StaticEmbeddings {
    public static func loadModel(
        weightsUrl: URL,
        loadConfig: LoadConfig = LoadConfig()
    ) throws -> StaticEmbeddings.Model {
        let data = try Safetensors.read(at: weightsUrl)
        let embeddings = try data.mlTensor(
            forKey: loadConfig.modelConfig.weightKeyTransform("embedding.weight"))
        return StaticEmbeddings.Model(embeddings: embeddings)
    }
}
