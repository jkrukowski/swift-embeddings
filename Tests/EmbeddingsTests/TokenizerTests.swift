import Foundation
import Testing

@testable import Embeddings

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
@Test func clipTokenizer() throws {
    let bundleUrl = Bundle.module
        .url(forResource: "merges", withExtension: "txt", subdirectory: "Resources")?
        .deletingLastPathComponent()
    let url = try #require(bundleUrl, "Wrong bundle URL")
    let tokenizer = try loadClipTokenizer(at: url)

    #expect(tokenizer.tokenize("", maxLength: 128, addSpecialTokens: true) == [49406, 49407])
    #expect(tokenizer.tokenize("", maxLength: 128, addSpecialTokens: false) == [])
    #expect(
        tokenizer.tokenize("a photo of a cat", maxLength: 128, addSpecialTokens: true)
            == [49406, 320, 1125, 539, 320, 2368, 49407])
    #expect(
        tokenizer.tokenize("a photo of a cat", maxLength: 128, addSpecialTokens: false)
            == [320, 1125, 539, 320, 2368])
    #expect(
        tokenizer.tokenize("a photo of a cat", maxLength: 5, addSpecialTokens: true)
            == [49406, 320, 1125, 539, 49407])
    #expect(
        tokenizer.tokenize("a photo of a cat", maxLength: 5, addSpecialTokens: false)
            == [320, 1125, 539, 320, 2368])
    #expect(
        tokenizer.tokenize(
            "a photo of a cat", maxLength: 128, padToLength: 10, addSpecialTokens: true)
            == [49406, 320, 1125, 539, 320, 2368, 49407, 0, 0, 0])
    #expect(
        tokenizer.tokenize(
            "a photo of a cat", maxLength: 128, padToLength: 10, addSpecialTokens: false)
            == [320, 1125, 539, 320, 2368, 0, 0, 0, 0, 0])
    #expect(
        tokenizer.tokenize(
            "a photo of a cat", maxLength: 5, padToLength: 10, addSpecialTokens: true)
            == [49406, 320, 1125, 539, 49407])
    #expect(
        tokenizer.tokenize(
            "a photo of a cat", maxLength: 5, padToLength: 10, addSpecialTokens: false)
            == [320, 1125, 539, 320, 2368])
    #expect(
        tokenizer.tokenize("a photo of a cat", maxLength: 128, addSpecialTokens: true)
            == tokenizer.tokenize(
                "    a    photo  of  a cat    ", maxLength: 128, addSpecialTokens: true)
    )
    #expect(
        tokenizer.tokenize("a photo of a cat", maxLength: 128, addSpecialTokens: true)
            == tokenizer.tokenize("A pHotO of a CaT", maxLength: 128, addSpecialTokens: true)
    )
}
