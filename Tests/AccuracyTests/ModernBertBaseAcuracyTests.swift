import CoreML
import MLTensorUtils
import Testing
import TestingUtils

@testable import Embeddings

@Suite(
    .enabled(if: ProcessInfo.processInfo.environment["UV_PATH"] != nil),
    .downloadModel(modelId: Utils.ModelId.modernBertBase, downloadBase: Utils.modelPath)
)
struct ModernBertBaseAccuracyTests {
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test("ModernBert Base Accuracy", arguments: ["Text to encode", "", "❤️"])
    func modernBertBaseAccuracy(_ text: String) async throws {
        let modelBundle = try await ModernBert.loadModelBundle(
            from: Utils.ModelId.modernBertBase,
            downloadBase: Utils.modelPath,
            loadConfig: .addWeightKeyPrefix("model.")
        )
        let encoded = try modelBundle.encode(text, postProcess: .meanPool)
        let swiftData = await encoded.cast(to: Float.self).scalars(of: Float.self)
        let modelPath = modelPath(
            modelId: Utils.ModelId.modernBertBase, cacheDirectory: Utils.modelPath)
        let pythonData = try await generateUsingTransformers(
            modelPath: modelPath,
            texts: [text],
            modelType: .modernbert
        )

        #expect(allClose(pythonData, swiftData, absoluteTolerance: 1e-5) == true)
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test("ModernBert Base Batch Accuracy")
    func modernBertBaseBatchAccuracy() async throws {
        let modelBundle = try await ModernBert.loadModelBundle(
            from: Utils.ModelId.modernBertBase,
            downloadBase: Utils.modelPath,
            loadConfig: .addWeightKeyPrefix("model.")
        )
        let encoded = try modelBundle.batchEncode(Utils.batchTextsToTests, postProcess: .meanPool)
        let swiftData = await encoded.cast(to: Float.self).scalars(of: Float.self)
        let modelPath = modelPath(
            modelId: Utils.ModelId.modernBertBase, cacheDirectory: Utils.modelPath)
        let pythonData = try await generateUsingTransformers(
            modelPath: modelPath,
            texts: Utils.batchTextsToTests,
            modelType: .modernbert
        )

        #expect(allClose(pythonData, swiftData, absoluteTolerance: 1e-5) == true)
    }
}
