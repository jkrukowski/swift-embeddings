import CoreML
import MLTensorUtils
import Testing
import TestingUtils

@testable import Embeddings

struct ModernBertTests {
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test func updateAttentionMaskAllUnmasked() async {
        let batchSize = 1
        let seqLen = 6
        let localAttention = 4
        let attentionMask = MLTensor(ones: [batchSize, seqLen], scalarType: Float.self)

        let (globalMask, slidingMask) = ModernBert.Model.updateAttentionMask(
            attentionMask,
            localAttention: localAttention
        )

        let globalData = await globalMask.scalars(of: Float.self)
        let slidingData = await slidingMask.scalars(of: Float.self)

        #expect(globalData.allSatisfy { $0 == 0.0 })
        #expect(
            allClose(
                slidingData,
                [
                    0.0, 0.0, 0.0, -1e+09, -1e+09, -1e+09,
                    0.0, 0.0, 0.0, 0.0, -1e+09, -1e+09,
                    0.0, 0.0, 0.0, 0.0, 0, -1e+09,
                    -1e+09, 0.0, 0.0, 0.0, 0.0, 0.0,
                    -1e+09, -1e+09, 0.0, 0.0, 0.0, 0.0,
                    -1e+09, -1e+09, -1e+09, 0.0, 0.0, 0.0,
                ]
            )
        )
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test func updateAttentionMaskAllMasked() async {
        let batchSize = 1
        let seqLen = 4
        let localAttention = 4
        let attentionMask = MLTensor(zeros: [batchSize, seqLen], scalarType: Float.self)

        let (globalMask, slidingMask) = ModernBert.Model.updateAttentionMask(
            attentionMask,
            localAttention: localAttention
        )

        let globalData = await globalMask.scalars(of: Float.self)
        let slidingData = await slidingMask.scalars(of: Float.self)

        #expect(globalData.allSatisfy { $0 == -1e9 })
        #expect(slidingData.allSatisfy { $0 == -1e9 })
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test func updateAttentionMaskWithMaskedPositions() async {
        let batchSize = 1
        let seqLen = 6
        let localAttention = 4
        var attentionMaskData = [Float](repeating: 1.0, count: seqLen)
        attentionMaskData[4] = 0.0  // Mask position 4
        let attentionMask = MLTensor(
            shape: [batchSize, seqLen],
            scalars: attentionMaskData
        )

        let (globalMask, slidingMask) = ModernBert.Model.updateAttentionMask(
            attentionMask,
            localAttention: localAttention
        )

        let globalData = await globalMask.scalars(of: Float.self)
        let slidingData = await slidingMask.scalars(of: Float.self)

        #expect(
            allClose(
                globalData,
                [
                    0.0, 0.0, 0.0, 0.0, -1e+09, 0.0,
                    0.0, 0.0, 0.0, 0.0, -1e+09, 0.0,
                    0.0, 0.0, 0.0, 0.0, -1e+09, 0.0,
                    0.0, 0.0, 0.0, 0.0, -1e+09, 0.0,
                    0.0, 0.0, 0.0, 0.0, -1e+09, 0.0,
                    0.0, 0.0, 0.0, 0.0, -1e+09, 0.0,
                ]
            )
        )

        #expect(
            allClose(
                slidingData,
                [
                    0.0, 0.0, 0.0, -1e+09, -1e+09, -1e+09,
                    0.0, 0.0, 0.0, 0.0, -1e+09, -1e+09,
                    0.0, 0.0, 0.0, 0.0, -1e+09, -1e+09,
                    -1e+09, 0.0, 0.0, 0.0, -1e+09, 0.0,
                    -1e+09, -1e+09, 0.0, 0.0, -1e+09, 0.0,
                    -1e+09, -1e+09, -1e+09, 0.0, -1e+09, 0.0,
                ]
            )
        )
    }
}
