import Foundation
import Testing

@testable import Embeddings

@Test func clipTokenizer() throws {
    let bundleUrl = Bundle.module
        .url(forResource: "merges", withExtension: "txt", subdirectory: "Resources")?
        .deletingLastPathComponent()
    let url = try #require(bundleUrl, "Wrong bundle URL")
    let tokenizer = try loadClipTokenizer(at: url)

    #expect(tokenizer.tokenize("", maxLength: 128) == [49406, 49407])
    #expect(
        tokenizer.tokenize("a photo of a cat", maxLength: 128)
            == [49406, 320, 1125, 539, 320, 2368, 49407])
    #expect(
        tokenizer.tokenize("a photo of a cat", maxLength: 5)
            == [49406, 320, 1125, 539, 49407])
    #expect(
        tokenizer.tokenize("a photo of a cat", maxLength: 128, padToLength: 10)
            == [49406, 320, 1125, 539, 320, 2368, 49407, 0, 0, 0])
    #expect(
        tokenizer.tokenize("a photo of a cat", maxLength: 5, padToLength: 10)
            == [49406, 320, 1125, 539, 49407])
    #expect(
        tokenizer.tokenize("a photo of a cat", maxLength: 128)
            == tokenizer.tokenize("    a    photo  of  a cat    ", maxLength: 128)
    )
    #expect(
        tokenizer.tokenize("a photo of a cat", maxLength: 128)
            == tokenizer.tokenize("A pHotO of a CaT", maxLength: 128)
    )
}
