import CoreML
import MLTensorUtils
import Testing
import TestingUtils

@testable import Embeddings

struct Qwen3Tests {
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test func rmsNorm() async {
        // variance = mean([3^2, 4^2]) = 12.5, scale = 1 / sqrt(12.5)
        let weight = MLTensor(shape: [2], scalars: [1, 1] as [Float])
        let layer = MLTensorUtils.rmsNorm(weight: weight, epsilon: 1e-6)
        let result = layer(MLTensor(shape: [1, 2], scalars: [3, 4] as [Float]))
        let data = await result.scalars(of: Float.self)

        #expect(result.shape == [1, 2])
        #expect(allClose(data, [0.8485281, 1.1313708], absoluteTolerance: 1e-5) == true)
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test func swiGLUMLP() async {
        // gate, up, down are all identity, so output = silu(x) * x
        // silu(1) = 1 * sigmoid(1) = 0.7310586, then * up(1) = 0.7310586
        let identity = MLTensor(
            shape: [2, 2],
            scalars: [
                1, 0,
                0, 1,
            ] as [Float]
        )
        let mlp = Qwen3.MLP(
            gate: MLTensorUtils.linear(weight: identity),
            up: MLTensorUtils.linear(weight: identity),
            down: MLTensorUtils.linear(weight: identity)
        )
        let result = mlp(MLTensor(shape: [1, 1, 2], scalars: [1, 1] as [Float]))
        let data = await result.scalars(of: Float.self)

        #expect(result.shape == [1, 1, 2])
        #expect(allClose(data, [0.7310586, 0.7310586], absoluteTolerance: 1e-5) == true)
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test func repeatKeyValue() async {
        // [bs=1, kvHeads=2, seq=1, headDim=2]
        // head0 = [1, 2], head1 = [3, 4]
        let x = MLTensor(shape: [1, 2, 1, 2], scalars: [1, 2, 3, 4] as [Float])
        let result = Qwen3.Attention.repeatKeyValue(x, nRep: 2)
        let data = await result.scalars(of: Float.self)

        // Each kv head is repeated consecutively: head0, head0, head1, head1
        #expect(result.shape == [1, 4, 1, 2])
        #expect(allClose(data, [1, 2, 1, 2, 3, 4, 3, 4]) == true)
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test func repeatKeyValueNoRepeat() async {
        let x = MLTensor(shape: [1, 2, 1, 2], scalars: [1, 2, 3, 4] as [Float])
        let result = Qwen3.Attention.repeatKeyValue(x, nRep: 1)
        let data = await result.scalars(of: Float.self)

        #expect(result.shape == [1, 2, 1, 2])
        #expect(allClose(data, [1, 2, 3, 4]) == true)
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test func makeAttentionMaskCausalOnly() async {
        let mask = Qwen3.Model.makeAttentionMask(seqLen: 3, attentionMask: nil)
        let data = await mask.scalars(of: Float.self)

        #expect(mask.shape == [1, 1, 3, 3])
        #expect(
            allClose(
                data,
                [
                    0, -1e9, -1e9,
                    0, 0, -1e9,
                    0, 0, 0,
                ]
            )
        )
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test func makeAttentionMaskWithPadding() async {
        // last position is padding
        let attentionMask = MLTensor(shape: [1, 3], scalars: [1, 1, 0] as [Float])
        let mask = Qwen3.Model.makeAttentionMask(seqLen: 3, attentionMask: attentionMask)
        let data = await mask.scalars(of: Float.self)

        #expect(mask.shape == [1, 1, 3, 3])
        #expect(
            allClose(
                data,
                [
                    0, -1e9, -2e9,
                    0, 0, -2e9,
                    0, 0, -1e9,
                ]
            )
        )
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test func lastTokenPoolWithMask() async {
        // tensor [bs=1, seq=3, hidden=2], rows: [1,1], [2,2], [3,3]
        let tensor = MLTensor(shape: [1, 3, 2], scalars: [1, 1, 2, 2, 3, 3] as [Float])
        // attention mask [1, 1, 0] -> last non-pad token is index 1 -> [2, 2]
        let attentionMask = MLTensor(shape: [1, 3], scalars: [1, 1, 0] as [Float])
        let result = lastTokenPool(tensor, attentionMask: attentionMask)
        let data = await result.scalars(of: Float.self)

        #expect(result.shape == [1, 2])
        #expect(allClose(data, [2, 2]) == true)
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test func lastTokenPoolWithoutMask() async {
        let tensor = MLTensor(shape: [1, 3, 2], scalars: [1, 1, 2, 2, 3, 3] as [Float])
        // no mask -> last token (index 2) -> [3, 3]
        let result = lastTokenPool(tensor)
        let data = await result.scalars(of: Float.self)

        #expect(result.shape == [1, 2])
        #expect(allClose(data, [3, 3]) == true)
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test func attentionOutputShapeWithGroupedQueryAttention() async {
        let hidden = 8
        let numHeads = 4
        let numKeyValueHeads = 2
        let headDim = 2
        let qWeight = MLTensor.float(shape: [numHeads * headDim, hidden]) / 100.0
        let kWeight = MLTensor.float(shape: [numKeyValueHeads * headDim, hidden]) / 100.0
        let vWeight = MLTensor.float(shape: [numKeyValueHeads * headDim, hidden]) / 100.0
        let oWeight = MLTensor.float(shape: [hidden, numHeads * headDim]) / 100.0
        let normWeight = MLTensor(ones: [headDim], scalarType: Float.self)

        let attention = Qwen3.Attention(
            qProj: MLTensorUtils.linear(weight: qWeight),
            kProj: MLTensorUtils.linear(weight: kWeight),
            vProj: MLTensorUtils.linear(weight: vWeight),
            oProj: MLTensorUtils.linear(weight: oWeight),
            qNorm: MLTensorUtils.rmsNorm(weight: normWeight, epsilon: 1e-6),
            kNorm: MLTensorUtils.rmsNorm(weight: normWeight, epsilon: 1e-6),
            rotaryEmbeddings: MLTensorUtils.roPE(dims: headDim, base: 1_000_000),
            numHeads: numHeads,
            numKeyValueHeads: numKeyValueHeads,
            headDim: headDim
        )

        let seqLen = 3
        let hiddenStates = MLTensor.float(shape: [1, seqLen, hidden]) / 100.0
        let mask = Qwen3.Model.makeAttentionMask(seqLen: seqLen, attentionMask: nil)
        let result = attention(hiddenStates, attentionMask: mask)

        #expect(result.shape == [1, seqLen, hidden])
    }
}
