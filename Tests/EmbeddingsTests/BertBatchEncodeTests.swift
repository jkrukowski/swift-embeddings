import CoreML
import MLTensorUtils
import Testing
import TestingUtils
import XCTest

@testable import Embeddings

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
final class BertBatchEncodeTests: XCTestCase {

    func testBatchEncodeMatchesIndividualEncode() async throws {
        print("\n" + String(repeating: "=", count: 80))
        print("BERT BATCH ENCODE CORRECTNESS TEST")
        print(String(repeating: "=", count: 80))

        let modelBundle = try await Bert.loadModelBundle(
            from: "sentence-transformers/all-MiniLM-L6-v2"
        )

        let testTexts = [
            "Hello",
            "The quick brown fox jumps over the lazy dog and continues running through the meadow",
            "Good morning everyone",
            "Machine learning enables computers to learn from data without being explicitly programmed for every scenario"
        ]

        print("\nTest Texts (varying lengths):")
        for (i, text) in testTexts.enumerated() {
            let wordCount = text.split(separator: " ").count
            print("  [\(i)] \(wordCount) words: \(text.prefix(60))\(text.count > 60 ? "..." : "")")
        }

        func l2Normalize(_ vector: [Float]) -> [Float] {
            let norm = sqrt(vector.map { $0 * $0 }.reduce(0, +))
            return vector.map { $0 / norm }
        }

        func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
            return zip(a, b).map { $0 * $1 }.reduce(0, +)
        }

        print("\n" + String(repeating: "-", count: 80))
        print("METHOD 1: Individual encode() calls")
        print(String(repeating: "-", count: 80))

        var individualEmbeddings: [[Float]] = []
        for (i, text) in testTexts.enumerated() {
            let encoded = try modelBundle.encode(text)
            let rawVector = await encoded.cast(to: Float.self)
                .shapedArray(of: Float.self).scalars
            let normalized = l2Normalize(Array(rawVector))
            individualEmbeddings.append(normalized)

            print("\n[\(i)] \"\(text.prefix(40))\(text.count > 40 ? "..." : "")\"")
            print("    First 10 values: ", terminator: "")
            print(normalized.prefix(10).map { String(format: "%.6f", $0) }.joined(separator: ", "))
        }

        print("\n" + String(repeating: "-", count: 80))
        print("METHOD 2: batchEncode()")
        print(String(repeating: "-", count: 80))

        let batchTensor = try modelBundle.batchEncode(testTexts, maxLength: 512)
        let batchVectors = await batchTensor.cast(to: Float.self)
            .shapedArray(of: Float.self).scalars

        let embeddingDim = batchVectors.count / testTexts.count

        var batchEmbeddings: [[Float]] = []
        for i in 0..<testTexts.count {
            let startIdx = i * embeddingDim
            let endIdx = startIdx + embeddingDim
            let rawVector = Array(batchVectors[startIdx..<endIdx])
            let normalized = l2Normalize(rawVector)
            batchEmbeddings.append(normalized)

            print("\n[\(i)] \"\(testTexts[i].prefix(40))\(testTexts[i].count > 40 ? "..." : "")\"")
            print("    First 10 values: ", terminator: "")
            print(normalized.prefix(10).map { String(format: "%.6f", $0) }.joined(separator: ", "))
        }

        print("\n" + String(repeating: "=", count: 80))
        print("COMPARISON")
        print(String(repeating: "=", count: 80))

        var allMatch = true
        var maxDifferences: [Float] = []

        for i in 0..<testTexts.count {
            let differences = zip(individualEmbeddings[i], batchEmbeddings[i])
                .map { abs($0 - $1) }
            let maxDiff = differences.max() ?? 0
            maxDifferences.append(maxDiff)

            let cosineScore = cosineSimilarity(individualEmbeddings[i], batchEmbeddings[i])

            print("\n[\(i)] \"\(testTexts[i].prefix(30))\(testTexts[i].count > 30 ? "..." : "")\"")
            print("    Max element difference: \(String(format: "%.8f", maxDiff))")
            print("    Cosine similarity:      \(String(format: "%.8f", cosineScore))")

            if maxDiff < 0.0001 && cosineScore > 0.9999 {
                print("    ✅ PASS - Vectors match")
            } else {
                print("    ❌ FAIL - Vectors differ significantly")
                print("    Individual first 10: \(individualEmbeddings[i].prefix(10).map { String(format: "%.6f", $0) }.joined(separator: ", "))")
                print("    Batch first 10:      \(batchEmbeddings[i].prefix(10).map { String(format: "%.6f", $0) }.joined(separator: ", "))")
                allMatch = false
            }
        }

        print("\n" + String(repeating: "=", count: 80))
        if allMatch {
            print("✅ SUCCESS: All vectors match")
        } else {
            print("❌ FAILURE: Vectors differ - attention mask bug present")
        }
        print(String(repeating: "=", count: 80) + "\n")

        for i in 0..<testTexts.count {
            XCTAssertLessThan(maxDifferences[i], 0.0001,
                "Text [\(i)] max diff should be < 0.0001, got \(maxDifferences[i])")

            let cosineScore = cosineSimilarity(individualEmbeddings[i], batchEmbeddings[i])
            XCTAssertGreaterThan(cosineScore, 0.9999,
                "Text [\(i)] cosine should be > 0.9999, got \(cosineScore)")
        }
    }

    func testSemanticSearchRanking() async throws {
        print("\n" + String(repeating: "=", count: 80))
        print("SEMANTIC SEARCH TEST")
        print(String(repeating: "=", count: 80))

        let modelBundle = try await Bert.loadModelBundle(
            from: "sentence-transformers/all-MiniLM-L6-v2"
        )

        let query = "How do neural networks learn from examples and improve their performance over time"
        let documents = [
            "Dogs are loyal pets",
            "Neural networks learn by adjusting weights through backpropagation",
            "The weather forecast",
            "Machine learning"
        ]

        print("\nQuery: \(query)")
        print("\nDocuments:")
        for (i, doc) in documents.enumerated() {
            print("  [\(i)] \(doc)")
        }

        func l2Normalize(_ vector: [Float]) -> [Float] {
            let norm = sqrt(vector.map { $0 * $0 }.reduce(0, +))
            return vector.map { $0 / norm }
        }

        func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
            return zip(a, b).map { $0 * $1 }.reduce(0, +)
        }

        let queryEncoded = try modelBundle.encode(query)
        let queryVector = await queryEncoded.cast(to: Float.self)
            .shapedArray(of: Float.self).scalars
        let queryNormalized = l2Normalize(Array(queryVector))

        let docTensor = try modelBundle.batchEncode(documents, maxLength: 512)
        let docVectors = await docTensor.cast(to: Float.self)
            .shapedArray(of: Float.self).scalars

        let embeddingDim = docVectors.count / documents.count
        var results: [(index: Int, similarity: Float)] = []

        for i in 0..<documents.count {
            let startIdx = i * embeddingDim
            let endIdx = startIdx + embeddingDim
            let rawVector = Array(docVectors[startIdx..<endIdx])
            let normalized = l2Normalize(rawVector)
            let similarity = cosineSimilarity(queryNormalized, normalized)
            results.append((index: i, similarity: similarity))
        }

        results.sort { $0.similarity > $1.similarity }

        print("\n" + String(repeating: "-", count: 80))
        print("Ranked Results:")
        for (rank, result) in results.enumerated() {
            print("\(rank + 1). [\(result.index)] \(String(format: "%.4f", result.similarity)) - \(documents[result.index])")
        }

        print("\n" + String(repeating: "=", count: 80))
        let correctRanking = results[0].index == 1
        if correctRanking {
            print("✅ Document about neural networks ranked first")
        } else {
            print("❌ Wrong document ranked first (expected doc 1, got doc \(results[0].index))")
        }
        print(String(repeating: "=", count: 80) + "\n")

        XCTAssertEqual(results[0].index, 1,
            "Neural networks doc should rank first for machine learning query")
    }
}
