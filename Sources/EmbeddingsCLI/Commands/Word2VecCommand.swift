import ArgumentParser
import Embeddings
import Foundation

struct Word2VecCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "word2vec",
        abstract: "Encode word using Word2Vec model"
    )
    @Option var modelId: String = "jkrukowski/glove-twitter-25"
    @Option var modelFile: String = "glove-twitter-25.txt"
    @Option var word: String = "queen"

    func run() async throws {
        let modelBundle = try await Word2Vec.loadModelBundle(
            from: modelId, modelFileName: modelFile)
        guard let encoded = modelBundle.encode(word) else {
            print("Word '\(word)' not found")
            return
        }
        let result = await encoded.cast(to: Float.self).shapedArray(of: Float.self).scalars
        print("\(word): \(result)")
        let mostSimilar = await modelBundle.mostSimilar(to: word, topK: 5)
        print("Most similar to \(word): \(mostSimilar)")
    }
}
