import CoreML
import MLTensorUtils
import Testing
import TestingUtils

@testable import Embeddings

@Suite(
    .enabled(if: ProcessInfo.processInfo.environment["UV_PATH"] != nil),
    .downloadModel(modelId: Utils.ModelId.model2Vec, downloadBase: Utils.modelPath)
)
struct Model2VecAccuracyTests {
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test("Model2Vec Accuracy", arguments: ["Text to encode", "", "❤️"])
    func model2VecAccuracy(_ text: String) async throws {
        let modelBundle = try await Model2Vec.loadModelBundle(
            from: Utils.ModelId.model2Vec,
            downloadBase: Utils.modelPath
        )
        let encoded = try modelBundle.encode(text, normalize: modelBundle.model.normalize)
        let swiftData = await encoded.cast(to: Float.self).scalars(of: Float.self)
        let modelPath = modelPath(modelId: Utils.ModelId.model2Vec, cacheDirectory: Utils.modelPath)
        let pythonData = try await generateUsingTransformers(
            modelPath: modelPath,
            texts: [text],
            modelType: .model2Vec
        )

        #expect(allClose(pythonData, swiftData, absoluteTolerance: 1e-5) == true)
    }
}
