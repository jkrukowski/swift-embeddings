import ArgumentParser
import Embeddings
import Foundation

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
struct Model2VecCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "model2vec",
        abstract: "Encode text using Model2Vec model"
    )
    @Option var modelId: String = "minishlab/potion-base-2M"
    @Option var text: String = "Text to encode"

    func run() async throws {
        let modelBundle = try await Model2Vec.loadModelBundle(from: modelId)
        let encoded = try modelBundle.encode(text)
        let result = await encoded.cast(to: Float.self).shapedArray(of: Float.self).scalars
        print(result)
    }
}
