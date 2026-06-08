import ArgumentParser
import Embeddings
import Foundation

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
struct BertCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "bert",
        abstract: "Encode text using BERT model"
    )
    @Option var modelId: String = "google-bert/bert-base-uncased"
    @Option var text: String = "Text to encode"
    @Option var maxLength: Int?

    func run() async throws {
        let modelBundle = try await Bert.loadModelBundle(
            from: modelId,
            loadConfig: LoadConfig.googleBert
        )
        var options = modelBundle.defaultEncodeOptions
        if let maxLength {
            options.maxLength = maxLength
        }
        let encoded = try modelBundle.encode(text, options: options)
        let result = await encoded.cast(to: Float.self).shapedArray(of: Float.self).scalars
        print(result)
    }
}
