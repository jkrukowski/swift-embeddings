import CoreML

/// Options shared by the transformer model bundles (`Bert`, `ModernBert`,
/// `NomicBert`, `Qwen3`, `Roberta`, `XLMRoberta`) for `encode` / `batchEncode`.
///
/// Use the per-model static defaults (e.g. `.bert`, `.qwen3`) as a starting
/// point and override the fields you need:
///
/// ```swift
/// var options = EncodeOptions.bert
/// options.postProcess = .meanPool(normalize: true)
/// let encoded = try bundle.encode("The cat is black", options: options)
/// ```
@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
public struct EncodeOptions: Sendable {
    /// Maximum number of tokens per input sequence.
    public var maxLength: Int
    /// Token id used to right-pad sequences in `batchEncode`. When `nil`, each
    /// model uses its own padding id (read from the model config where
    /// applicable). Ignored by single-sequence `encode`.
    public var padTokenId: Int?
    /// How the token-level output is pooled into a single embedding.
    public var postProcess: PostProcess?
    /// Compute policy for the underlying `MLTensor` work.
    public var computePolicy: MLComputePolicy

    public init(
        maxLength: Int,
        padTokenId: Int? = nil,
        postProcess: PostProcess? = nil,
        computePolicy: MLComputePolicy = .cpuAndGPU
    ) {
        self.maxLength = maxLength
        self.padTokenId = padTokenId
        self.postProcess = postProcess
        self.computePolicy = computePolicy
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension EncodeOptions {
    public static let bert = Self(maxLength: 512, postProcess: .clsTokenPool(normalize: false))
    public static let modernBert = Self(maxLength: 8192, postProcess: nil)
    public static let nomicBert = Self(maxLength: 2048, postProcess: nil)
    public static let qwen3 = Self(maxLength: 32_768, postProcess: .lastTokenPool(normalize: true))
    public static let roberta = Self(maxLength: 512, postProcess: .clsTokenPool(normalize: false))
    public static let xlmRoberta = Self(
        maxLength: 512,
        postProcess: .clsTokenPool(normalize: false)
    )
    // `postProcess` is ignored by Clip; its pooling/projection/normalize is fixed.
    public static let clip = Self(maxLength: 77, postProcess: nil)
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension EncodeOptions {
    func applying(_ defaults: SentenceTransformersDefaults?) -> Self {
        guard let defaults else { return self }
        var options = self
        if let postProcess = defaults.postProcess {
            options.postProcess = postProcess
        } else if let normalize = defaults.normalize {
            if let current = options.postProcess {
                // Layout requested normalization but not a recognized pooling module; apply it
                // on top of the bundle's default pooling.
                options.postProcess = current.withNormalize(normalize)
            } else {
                // The layout asked for normalization but neither it nor the bundle provides a
                // pooling step to attach it to, so the request can't be honored. Surface it
                // rather than dropping it silently.
                logger.error(
                    """
                    sentence-transformers layout requested normalization but no pooling is \
                    available to apply it to; ignoring the normalization request.
                    """
                )
            }
        }
        if let maxLength = defaults.maxLength {
            options.maxLength = maxLength
        }
        return options
    }
}
