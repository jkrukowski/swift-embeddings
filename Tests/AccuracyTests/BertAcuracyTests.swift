import CoreML
import MLTensorUtils
import Testing
import TestingUtils

@testable import Embeddings

@Suite(
    .enabled(if: ProcessInfo.processInfo.environment["UV_PATH"] != nil),
    .downloadModel(modelId: Utils.ModelId.bert, downloadBase: Utils.modelPath)
)
struct BertAccuracyTests {
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test("Bert Accuracy", .serialized, arguments: ["Text to encode", "", "❤️"])
    func bertAccuracy(_ text: String) async throws {
        let modelBundle = try await Bert.loadModelBundle(
            from: Utils.ModelId.bert,
            downloadBase: Utils.modelPath,
            loadConfig: .googleBert
        )
        let encoded = try modelBundle.encode(text)
        let swiftData = await encoded.cast(to: Float.self).scalars(of: Float.self)
        let modelPath = modelPath(modelId: Utils.ModelId.bert, cacheDirectory: Utils.modelPath)
        let pythonData = try await generateUsingTransformers(
            modelPath: modelPath,
            texts: [text],
            modelType: .bert
        )

        #expect(allClose(pythonData, swiftData, absoluteTolerance: 1e-5) == true)
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test("Bert Batch Accuracy")
    func bertBatchAccuracy() async throws {
        let modelBundle = try await Bert.loadModelBundle(
            from: Utils.ModelId.bert,
            downloadBase: Utils.modelPath,
            loadConfig: .googleBert
        )
        let encoded = try modelBundle.batchEncode(Utils.batchTextsToTests)
        let swiftData = await encoded.cast(to: Float.self).scalars(of: Float.self)
        let modelPath = modelPath(modelId: Utils.ModelId.bert, cacheDirectory: Utils.modelPath)
        let pythonData = try await generateUsingTransformers(
            modelPath: modelPath,
            texts: Utils.batchTextsToTests,
            modelType: .bert
        )

        #expect(allClose(pythonData, swiftData, absoluteTolerance: 1e-5) == true)
    }
}
