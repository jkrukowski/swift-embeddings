import Foundation
import Tokenizers

public protocol TextTokenizer: Sendable {
    var unknownTokenId: Int? { get }

    func tokenizeText(_ text: String) throws -> [Int32]
    func tokenizeText(_ text: String, maxLength: Int?) throws -> [Int32]
    func tokenizeText(_ text: String, maxLength: Int?, addSpecialTokens: Bool) throws -> [Int32]
    func tokenizeTextsPaddingToLongest(
        _ texts: [String], padTokenId: Int
    ) throws -> BatchTokenizeResult
    func tokenizeTextsPaddingToLongest(
        _ texts: [String], padTokenId: Int, maxLength: Int?
    ) throws -> BatchTokenizeResult
    func tokenizeTextsPaddingToLongest(
        _ texts: [String], padTokenId: Int, maxLength: Int?, addSpecialTokens: Bool
    ) throws -> BatchTokenizeResult
}

extension TextTokenizer {
    public func tokenizeText(_ text: String) throws -> [Int32] {
        try tokenizeText(text, maxLength: nil, addSpecialTokens: true)
    }

    public func tokenizeText(_ text: String, maxLength: Int?) throws -> [Int32] {
        try tokenizeText(text, maxLength: maxLength, addSpecialTokens: true)
    }

    public func tokenizeTextsPaddingToLongest(
        _ texts: [String],
        padTokenId: Int
    ) throws -> BatchTokenizeResult {
        try tokenizeTextsPaddingToLongest(
            texts, padTokenId: padTokenId, maxLength: nil, addSpecialTokens: true)
    }

    public func tokenizeTextsPaddingToLongest(
        _ texts: [String],
        padTokenId: Int,
        maxLength: Int?
    ) throws -> BatchTokenizeResult {
        try tokenizeTextsPaddingToLongest(
            texts, padTokenId: padTokenId, maxLength: maxLength, addSpecialTokens: true)
    }

    public func tokenizeTextsPaddingToLongest(
        _ texts: [String],
        padTokenId: Int,
        maxLength: Int?,
        addSpecialTokens: Bool
    ) throws -> BatchTokenizeResult {
        var longest = 0
        var tokenizedTexts = [[Int32]]()
        tokenizedTexts.reserveCapacity(texts.count)
        for text in texts {
            let encoded = try tokenizeText(
                text,
                maxLength: maxLength,
                addSpecialTokens: addSpecialTokens
            )
            longest = max(longest, encoded.count)
            tokenizedTexts.append(encoded)
        }
        var tokens = [Int32]()
        tokens.reserveCapacity(longest * tokenizedTexts.count)
        var attentionMask = [Float]()
        attentionMask.reserveCapacity(longest * tokenizedTexts.count)
        for item in tokenizedTexts {
            tokens.append(contentsOf: item)
            attentionMask.append(contentsOf: Array(repeating: 1.0, count: item.count))
            if item.count < longest {
                tokens.append(
                    contentsOf: Array(repeating: Int32(padTokenId), count: longest - item.count))
                attentionMask.append(
                    contentsOf: Array(repeating: 0.0, count: longest - item.count))
            }
        }
        return BatchTokenizeResult(
            tokens: tokens,
            attentionMask: attentionMask,
            shape: [tokenizedTexts.count, longest]
        )
    }
}

public struct BatchTokenizeResult {
    public let tokens: [Int32]
    public let attentionMask: [Float]
    public let shape: [Int]

    public init(tokens: [Int32], attentionMask: [Float], shape: [Int]) {
        self.tokens = tokens
        self.attentionMask = attentionMask
        self.shape = shape
    }
}

public struct TokenizerWrapper {
    private let tokenizer: any Tokenizers.Tokenizer

    public var unknownTokenId: Int? {
        tokenizer.unknownTokenId
    }

    public init(_ tokenizer: any Tokenizers.Tokenizer) {
        self.tokenizer = tokenizer
    }
}

extension TokenizerWrapper: TextTokenizer {
    public func tokenizeText(
        _ text: String,
        maxLength: Int?,
        addSpecialTokens: Bool
    ) throws -> [Int32] {
        var encoded = tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)
        if let maxLength, encoded.count > maxLength {
            encoded.removeLast(encoded.count - maxLength)
        }
        return encoded.map { Int32($0) }
    }
}
