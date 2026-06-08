import ArgumentParser
import Embeddings
import Foundation

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
struct ModernBertCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "modern-bert",
        abstract: "Encode text using ModernBERT model"
    )
    @Option var modelId: String = "answerdotai/ModernBERT-base"
    @Option var text: String = "Text to encode"
    @Option var maxLength: Int?

    func run() async throws {
        let modelBundle = try await ModernBert.loadModelBundle(
            from: modelId,
            loadConfig: .addWeightKeyPrefix("model.")
        )
        var options = modelBundle.defaultEncodeOptions
        if let maxLength {
            options.maxLength = maxLength
        }
        if options.postProcess == nil {
            options.postProcess = .meanPool(normalize: false)
        }
        let encoded = try modelBundle.encode(text, options: options)
        let result = await encoded.cast(to: Float.self).shapedArray(of: Float.self).scalars
        print(result)
    }
}
