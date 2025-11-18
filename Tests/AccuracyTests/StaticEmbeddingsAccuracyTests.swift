import CoreML
import MLTensorUtils
import Testing
import TestingUtils

@testable import Embeddings

@Suite(
    .enabled(if: ProcessInfo.processInfo.environment["UV_PATH"] != nil),
    .downloadModel(modelId: Utils.ModelId.staticEmbeddings, downloadBase: Utils.modelPath)
)
struct StaticEmbeddingsAccuracyTests {
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test("Static Embeddings Accuracy", arguments: ["Text to encode", "", "❤️"])
    func staticEmbeddingsAccuracy(_ text: String) async throws {
        let modelBundle = try await StaticEmbeddings.loadModelBundle(
            from: Utils.ModelId.staticEmbeddings,
            downloadBase: Utils.modelPath,
            loadConfig: LoadConfig.staticEmbeddings
        )
        let encoded = try modelBundle.encode(text, normalize: true, truncateDimension: 1023)
        let swiftData = await encoded.cast(to: Float.self).scalars(of: Float.self)
        let modelPath = modelPath(
            modelId: Utils.ModelId.staticEmbeddings, cacheDirectory: Utils.modelPath)
        let pythonData = try await generateUsingTransformers(
            modelPath: modelPath,
            texts: [text],
            modelType: .staticEmbeddings
        )

        #expect(allClose(pythonData, swiftData, absoluteTolerance: 1e-5) == true)
    }
}
