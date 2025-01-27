import ArgumentParser
import Embeddings
import Foundation

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
struct ClipCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "clip",
        abstract: "Encode text using CLIP model"
    )
    @Option var modelId: String = "jkrukowski/clip-vit-base-patch16"
    @Option var text: String = "a photo of a dog"
    @Option var maxLength: Int = 77

    func run() async throws {
        let modelBundle = try await Clip.loadModelBundle(from: modelId)
        let encoded = try modelBundle.encode(text, maxLength: maxLength)
        let result = await encoded.cast(to: Float.self).shapedArray(of: Float.self).scalars
        print(result)
    }
}
