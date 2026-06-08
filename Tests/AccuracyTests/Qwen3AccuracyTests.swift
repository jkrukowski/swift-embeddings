import CoreML
import MLTensorUtils
import Testing
import TestingUtils

@testable import Embeddings

@Suite(
    .enabled(if: ProcessInfo.processInfo.environment["UV_PATH"] != nil),
    .downloadModel(modelId: Utils.ModelId.qwen3Embedding, downloadBase: Utils.modelPath)
)
struct Qwen3AccuracyTests {
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test("Qwen3 Accuracy", .serialized, arguments: ["Text to encode", "❤️", "The dog barks"])
    func qwen3Accuracy(_ text: String) async throws {
        let modelBundle = try await Qwen3.loadModelBundle(
            from: Utils.ModelId.qwen3Embedding,
            downloadBase: Utils.modelPath
        )
        let encoded = try modelBundle.encode(text)
        let swiftData = await encoded.cast(to: Float.self).scalars(of: Float.self)
        let modelPath = modelPath(
            modelId: Utils.ModelId.qwen3Embedding,
            cacheDirectory: Utils.modelPath
        )
        let pythonData = try await generateUsingTransformers(
            modelPath: modelPath,
            texts: [text],
            modelType: .qwen3
        )

        #expect(allClose(pythonData, swiftData, absoluteTolerance: 1e-5) == true)
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test("Qwen3 Batch Accuracy")
    func qwen3BatchAccuracy() async throws {
        let modelBundle = try await Qwen3.loadModelBundle(
            from: Utils.ModelId.qwen3Embedding,
            downloadBase: Utils.modelPath
        )
        let encoded = try modelBundle.batchEncode(Utils.batchTextsToTests)
        let swiftData = await encoded.cast(to: Float.self).scalars(of: Float.self)
        let modelPath = modelPath(
            modelId: Utils.ModelId.qwen3Embedding,
            cacheDirectory: Utils.modelPath
        )
        let pythonData = try await generateUsingTransformers(
            modelPath: modelPath,
            texts: Utils.batchTextsToTests,
            modelType: .qwen3
        )

        #expect(allClose(pythonData, swiftData, absoluteTolerance: 1e-5) == true)
    }
}
