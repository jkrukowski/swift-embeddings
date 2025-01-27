import CoreML
import Foundation
import Hub
import MLTensorUtils

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Word2Vec {
    public static func loadModelBundle(
        from hubRepoId: String,
        downloadBase: URL? = nil,
        useBackgroundSession: Bool = false,
        loadConfig: LoadConfig = LoadConfig()
    ) async throws -> Word2Vec.ModelBundle {
        let modelFolder = try await downloadModelFromHub(
            from: hubRepoId,
            downloadBase: downloadBase,
            useBackgroundSession: useBackgroundSession,
            globs: [loadConfig.modelFileName]
        )
        let modelFile = modelFolder.appendingPathComponent(loadConfig.modelFileName)
        return try await loadModelBundle(from: modelFile)
    }

    public static func loadModelBundle(
        from modelFile: URL
    ) async throws -> Word2Vec.ModelBundle {
        let data = try Data(contentsOf: modelFile, options: .mappedIfSafe)
        let lines = String(decoding: data, as: UTF8.self).components(separatedBy: .newlines)
        var lineCount: Int?
        var vectorSize: Int?
        for line in lines.prefix(1) {
            let parts = line.components(separatedBy: .whitespaces)
            if parts.count == 2 {
                lineCount = Int(parts[0])
                vectorSize = Int(parts[1])
            }
        }
        guard let lineCount, let vectorSize else {
            throw EmbeddingsError.invalidFile
        }
        var keyToIndex = [String: Int]()
        keyToIndex.reserveCapacity(lineCount)
        var indexToKey = [Int: String]()
        indexToKey.reserveCapacity(lineCount)
        var vectors = [Float]()
        vectors.reserveCapacity(lineCount * vectorSize)
        var currentIndex = 0
        for line in lines.dropFirst() where !line.isEmpty {
            let parts = line.components(separatedBy: .whitespaces)
            let word = parts[0]
            let vector = parts.dropFirst().map { Float($0)! }
            if vector.count != vectorSize {
                throw EmbeddingsError.invalidFile
            }
            keyToIndex[word] = currentIndex
            indexToKey[currentIndex] = word
            vectors.append(contentsOf: vector)
            currentIndex += 1
        }
        let embeddings = MLTensor(shape: [lineCount, vectorSize], scalars: vectors)
        return Word2Vec.ModelBundle(
            keyToIndex: keyToIndex,
            indexToKey: indexToKey,
            embeddings: embeddings
        )
    }
}
