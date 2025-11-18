import CoreML
import MLTensorUtils
import Testing
import TestingUtils

@testable import Embeddings

@Suite(
    .enabled(if: ProcessInfo.processInfo.environment["UV_PATH"] != nil),
    .downloadModel(modelId: Utils.ModelId.roberta, downloadBase: Utils.modelPath)
)
struct RobertaAccuracyTests {
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test("Roberta Accuracy", arguments: ["Text to encode", "", "❤️"])
    func robertaAccuracy(_ text: String) async throws {
        let modelBundle = try await Roberta.loadModelBundle(
            from: Utils.ModelId.roberta,
            downloadBase: Utils.modelPath,
            loadConfig: .addWeightKeyPrefix("roberta.")
        )
        let encoded = try modelBundle.encode(text)
        let swiftData = await encoded.cast(to: Float.self).scalars(of: Float.self)
        let modelPath = modelPath(modelId: Utils.ModelId.roberta, cacheDirectory: Utils.modelPath)
        let pythonData = try await generateUsingTransformers(
            modelPath: modelPath,
            texts: [text],
            modelType: .roberta
        )

        #expect(allClose(pythonData, swiftData, absoluteTolerance: 1e-5) == true)
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test("Roberta Batch Accuracy")
    func robertaBatchAccuracy() async throws {
        let modelBundle = try await Roberta.loadModelBundle(
            from: Utils.ModelId.roberta,
            downloadBase: Utils.modelPath,
            loadConfig: .addWeightKeyPrefix("roberta.")
        )
        let encoded = try modelBundle.batchEncode(Utils.batchTextsToTests)
        let swiftData = await encoded.cast(to: Float.self).scalars(of: Float.self)
        let modelPath = modelPath(modelId: Utils.ModelId.roberta, cacheDirectory: Utils.modelPath)
        let pythonData = try await generateUsingTransformers(
            modelPath: modelPath,
            texts: Utils.batchTextsToTests,
            modelType: .roberta
        )

        #expect(allClose(pythonData, swiftData, absoluteTolerance: 1e-5) == true)
    }
}
