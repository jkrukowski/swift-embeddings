import CoreML
import MLTensorUtils
import Testing
import TestingUtils

@testable import Embeddings

@Suite(
    .enabled(if: ProcessInfo.processInfo.environment["UV_PATH"] != nil),
    .downloadModel(modelId: Utils.ModelId.nomicEmbedTextV15, downloadBase: Utils.modelPath)
)
struct NomicBertAccuracyTests {
    private func prefixed(_ text: String) -> String {
        "search_document: \(text)"
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test("NomicBert Accuracy", .serialized, arguments: ["Text to encode", "", "❤️"])
    func nomicBertAccuracy(_ text: String) async throws {
        let modelBundle = try await NomicBert.loadModelBundle(
            from: Utils.ModelId.nomicEmbedTextV15,
            downloadBase: Utils.modelPath
        )
        var options = EncodeOptions.nomicBert
        options.postProcess = .meanPool(normalize: false)
        options.computePolicy = .cpuOnly
        let encoded = try modelBundle.encode(prefixed(text), options: options)
        let swiftData = await encoded.cast(to: Float.self).scalars(of: Float.self)
        let modelPath = modelPath(
            modelId: Utils.ModelId.nomicEmbedTextV15,
            cacheDirectory: Utils.modelPath
        )
        let pythonData = try await generateUsingTransformers(
            modelPath: modelPath,
            texts: [prefixed(text)],
            modelType: .nomic
        )

        #expect(allClose(pythonData, swiftData, absoluteTolerance: 1e-5) == true)
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test("NomicBert Batch Accuracy")
    func nomicBertBatchAccuracy() async throws {
        let modelBundle = try await NomicBert.loadModelBundle(
            from: Utils.ModelId.nomicEmbedTextV15,
            downloadBase: Utils.modelPath
        )
        let prefixedTexts = Utils.batchTextsToTests.map(prefixed)
        var options = EncodeOptions.nomicBert
        options.postProcess = .meanPool(normalize: false)
        options.computePolicy = .cpuOnly
        let encoded = try modelBundle.batchEncode(prefixedTexts, options: options)
        let swiftData = await encoded.cast(to: Float.self).scalars(of: Float.self)
        let modelPath = modelPath(
            modelId: Utils.ModelId.nomicEmbedTextV15,
            cacheDirectory: Utils.modelPath
        )
        let pythonData = try await generateUsingTransformers(
            modelPath: modelPath,
            texts: prefixedTexts,
            modelType: .nomic
        )

        #expect(allClose(pythonData, swiftData, absoluteTolerance: 1e-5) == true)
    }
}
