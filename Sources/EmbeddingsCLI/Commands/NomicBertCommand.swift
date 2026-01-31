import ArgumentParser
import Embeddings
import Foundation

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
struct NomicBertCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "nomic-bert",
        abstract: "Encode text using Nomic embedding model"
    )
    @Option var modelId: String = "nomic-ai/nomic-embed-text-v1.5"
    @Option var text: String = "Text to encode"
    @Option var maxLength: Int = 2048

    func run() async throws {
        let modelBundle = try await NomicBert.loadModelBundle(from: modelId)
        let encoded = try modelBundle.encode(text, maxLength: maxLength, postProcess: .meanPool)
        let result = await encoded.cast(to: Float.self).shapedArray(of: Float.self).scalars
        print(result)
    }
}
