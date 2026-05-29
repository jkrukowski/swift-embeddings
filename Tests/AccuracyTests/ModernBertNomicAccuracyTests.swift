import CoreML
import MLTensorUtils
import Testing
import TestingUtils

@testable import Embeddings

@Suite(
    .enabled(if: ProcessInfo.processInfo.environment["UV_PATH"] != nil),
    .downloadModel(modelId: Utils.ModelId.modernBertNomic, downloadBase: Utils.modelPath)
)
struct ModernBertNomicAccuracyTests {
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test("ModernBert Nomic Accuracy", arguments: ["Text to encode", "", "❤️"])
    func modernBertNomicAccuracy(_ text: String) async throws {
        let modelBundle = try await ModernBert.loadModelBundle(
            from: Utils.ModelId.modernBertNomic,
            downloadBase: Utils.modelPath
        )
        let encoded = try modelBundle.encode(text, postProcess: .meanPoolAndNormalize)
        let swiftData = await encoded.cast(to: Float.self).scalars(of: Float.self)
        let modelPath = modelPath(
            modelId: Utils.ModelId.modernBertNomic, cacheDirectory: Utils.modelPath)
        let pythonData = try await generateUsingTransformers(
            modelPath: modelPath,
            texts: [text],
            modelType: .modernbert
        )

        #expect(allClose(pythonData, swiftData, absoluteTolerance: 1e-5) == true)
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test("ModernBert Nomic Batch Accuracy")
    func modernBertNomicBatchAccuracy() async throws {
        let modelBundle = try await ModernBert.loadModelBundle(
            from: Utils.ModelId.modernBertNomic,
            downloadBase: Utils.modelPath
        )
        let encoded = try modelBundle.batchEncode(
            Utils.batchTextsToTests, postProcess: .meanPoolAndNormalize)
        let swiftData = await encoded.cast(to: Float.self).scalars(of: Float.self)
        let modelPath = modelPath(
            modelId: Utils.ModelId.modernBertNomic, cacheDirectory: Utils.modelPath)
        let pythonData = try await generateUsingTransformers(
            modelPath: modelPath,
            texts: Utils.batchTextsToTests,
            modelType: .modernbert
        )

        #expect(allClose(pythonData, swiftData, absoluteTolerance: 1e-5) == true)
    }
}
