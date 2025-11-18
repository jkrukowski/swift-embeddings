import CoreML
import MLTensorUtils
import Testing
import TestingUtils

@testable import Embeddings

@Suite(
    .enabled(if: ProcessInfo.processInfo.environment["UV_PATH"] != nil),
    .downloadModel(modelId: Utils.ModelId.clip, downloadBase: Utils.modelPath)
)
struct ClipAccuracyTests {
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test("Clip Accuracy")
    func clipAccuracy() async throws {
        let text = "a photo of a dog"
        let modelBundle = try await Clip.loadModelBundle(
            from: Utils.ModelId.clip,
            downloadBase: Utils.modelPath
        )
        let tokens = try modelBundle.tokenizer.tokenizeText(text, maxLength: 77)
        let inputIds = MLTensor(shape: [1, tokens.count], scalars: tokens)
        let modelOutput = modelBundle.textModel(inputIds: inputIds)
        let swiftData =
            await modelOutput
            .poolerOutput
            .cast(to: Float.self)
            .scalars(of: Float.self)
        let modelPath = modelPath(modelId: Utils.ModelId.clip, cacheDirectory: Utils.modelPath)
        let pythonData = try await generateUsingTransformers(
            modelPath: modelPath,
            texts: [text],
            modelType: .clip
        )

        #expect(allClose(pythonData, swiftData, absoluteTolerance: 1e-5) == true)
    }
}
