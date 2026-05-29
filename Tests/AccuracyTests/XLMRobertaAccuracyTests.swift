import CoreML
import MLTensorUtils
import Testing
import TestingUtils

@testable import Embeddings

@Suite(
    .enabled(if: ProcessInfo.processInfo.environment["UV_PATH"] != nil),
    .downloadModel(modelId: Utils.ModelId.xlmRoberta, downloadBase: Utils.modelPath)
)
struct XLMRobertaAccuracyTests {
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test("XLM Roberta Accuracy", arguments: ["Text to encode", "", "❤️"])
    func xlmRobertaAccuracy(_ text: String) async throws {
        let modelBundle = try await XLMRoberta.loadModelBundle(
            from: Utils.ModelId.xlmRoberta,
            downloadBase: Utils.modelPath,
            loadConfig: .addWeightKeyPrefix("roberta.")
        )
        let encoded = try modelBundle.encode(text)
        let swiftData = await encoded.cast(to: Float.self).scalars(of: Float.self)
        let modelPath = modelPath(
            modelId: Utils.ModelId.xlmRoberta, cacheDirectory: Utils.modelPath)
        let pythonData = try await generateUsingTransformers(
            modelPath: modelPath,
            texts: [text],
            modelType: .xlmRoberta
        )

        #expect(allClose(pythonData, swiftData, absoluteTolerance: 1e-5) == true)
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test("XLM Roberta Batch Accuracy")
    func xlmRobertaBatchAccuracy() async throws {
        let modelBundle = try await XLMRoberta.loadModelBundle(
            from: Utils.ModelId.xlmRoberta,
            downloadBase: Utils.modelPath,
            loadConfig: .addWeightKeyPrefix("roberta.")
        )
        let encoded = try modelBundle.batchEncode(Utils.batchTextsToTests)
        let swiftData = await encoded.cast(to: Float.self).scalars(of: Float.self)
        let modelPath = modelPath(
            modelId: Utils.ModelId.xlmRoberta, cacheDirectory: Utils.modelPath)
        let pythonData = try await generateUsingTransformers(
            modelPath: modelPath,
            texts: Utils.batchTextsToTests,
            modelType: .xlmRoberta
        )

        #expect(allClose(pythonData, swiftData, absoluteTolerance: 1e-5) == true)
    }
}
