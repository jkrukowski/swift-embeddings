import Foundation
import SentencepieceTokenizer
import Synchronization

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
final class XLMRobetaTokenizer: Sendable {
    /// TODO: Make `SentencepieceTokenizer` conform to `Sendable`
    private let tokenizer: Mutex<SentencepieceTokenizer>
    private let addedTokens: [String: Int]

    init(
        tokenizerModelUrl: URL,
        addedTokens: [String: Int],
        tokenOffset: Int = 1
    ) throws {
        let sentencepieceTokenizer = try SentencepieceTokenizer(
            modelPath: tokenizerModelUrl.path,
            tokenOffset: tokenOffset
        )
        self.tokenizer = Mutex(sentencepieceTokenizer)
        self.addedTokens = addedTokens
    }

    func tokenize(
        _ text: String,
        maxLength: Int?,
        padToLength: Int? = nil,
        addSpecialTokens: Bool
    ) throws -> [Int] {
        let tokenIds = try tokenizer.withLock {
            try $0.encode(text)
        }
        var result = addSpecialTokens ? [bosTokenId] : []
        if let maxLength {
            if addSpecialTokens {
                precondition(
                    maxLength >= 2, "maxLength must be at least 2 to accommodate BOS and EOS tokens"
                )
                // Truncate to maxLength - 2 to make space for bos and eos tokens
                result.append(contentsOf: tokenIds.prefix(maxLength - 2))
            } else {
                result.append(contentsOf: tokenIds.prefix(maxLength))
            }
        } else {
            result.append(contentsOf: tokenIds)
        }
        if addSpecialTokens {
            result.append(eosTokenId)
        }
        // If padToLength is provided, pad the tokenIds with padTokenId
        if let padToLength {
            precondition(padToLength - 2 >= 0, "padToLength must be greater than or equal to 2")
            if let maxLength {
                if padToLength <= maxLength {
                    result.append(
                        contentsOf: Array(repeating: 0, count: padToLength - result.count))
                }
            } else {
                result.append(
                    contentsOf: Array(repeating: 0, count: padToLength - result.count))
            }
        }
        return result
    }

    var bosTokenId: Int {
        if let bosTokenId = addedTokens["<s>"] {
            return bosTokenId
        }
        return tokenizer.withLock {
            $0.bosTokenId
        }
    }

    var eosTokenId: Int {
        if let eosTokenId = addedTokens["</s>"] {
            return eosTokenId
        }
        return tokenizer.withLock {
            $0.eosTokenId
        }
    }

    var padTokenId: Int {
        if let padTokenId = addedTokens["<pad>"] {
            return padTokenId
        }
        return tokenizer.withLock {
            $0.padTokenId
        }
    }

    var unkTokenId: Int {
        if let unkTokenId = addedTokens["<unk>"] {
            return unkTokenId
        }
        return tokenizer.withLock {
            $0.unkTokenId
        }
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension XLMRobetaTokenizer: TextTokenizer {
    var unknownTokenId: Int? {
        unkTokenId
    }

    func tokenizeText(
        _ text: String,
        maxLength: Int?,
        addSpecialTokens: Bool
    ) throws -> [Int32] {
        try tokenize(
            text,
            maxLength: maxLength,
            padToLength: nil,
            addSpecialTokens: addSpecialTokens
        ).map { Int32($0) }
    }
}
