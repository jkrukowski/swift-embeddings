import CoreML
import Numerics
import Testing
import TestingUtils

@testable import MLTensorUtils

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
@Test func embeddingLayer1D() async {
    let embedding = embedding(weight: MLTensor.float(shape: [12]))
    let result = embedding(MLTensor([0, 2, 4] as [Int32]))

    #expect(result.shape == [3])
    let resultArray = await result.scalars(of: Float.self)
    #expect(resultArray == [0, 2, 4])
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
@Test func embeddingLayer2D() async {
    let embedding = embedding(weight: MLTensor.float(shape: [6, 2]))
    let result = embedding(MLTensor([0, 2, 4] as [Int32]))

    #expect(result.shape == [3, 2])
    let resultArray = await result.scalars(of: Float.self)
    #expect(resultArray == [0, 1, 4, 5, 8, 9])
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
@Test func embeddingLayer3D() async {
    let embedding = embedding(weight: MLTensor.float(shape: [2, 2, 2]))
    let result = embedding(MLTensor([0, 1] as [Int32]))

    #expect(result.shape == [2, 2, 2])
    let resultArray = await result.scalars(of: Float.self)
    #expect(resultArray == [0, 1, 2, 3, 4, 5, 6, 7])
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
@Test func layerNorm1D() async {
    let weight = MLTensor(
        shape: [3],
        scalars: [1, 2, 3],
        scalarType: Float.self
    )
    let bias = MLTensor(
        shape: [3],
        scalars: [4, 5, 6],
        scalarType: Float.self
    )
    let layerNorm = layerNorm(weight: weight, bias: bias, epsilon: 1e-5)
    let input = MLTensor(
        shape: [2, 3],
        scalars: [1, 2, 3, 4, 5, 6],
        scalarType: Float.self
    )
    let result = layerNorm(input)

    #expect(result.shape == [2, 3])
    let resultArray = await result.scalars(of: Float.self)
    let expectedArray: [Float] = [2.7753, 5.0000, 9.6742, 2.7753, 5.0000, 9.6742]
    #expect(allClose(resultArray, expectedArray) == true)
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
@Test func layerNorm2D() async {
    let weight = MLTensor(
        shape: [2, 3],
        scalars: [1, 2, 3, 4, 5, 6],
        scalarType: Float.self
    )
    let bias = MLTensor(
        shape: [2, 3],
        scalars: [4, 5, 6, 7, 8, 9],
        scalarType: Float.self
    )
    let layerNorm = layerNorm(weight: weight, bias: bias, epsilon: 1e-5)
    let input = MLTensor(
        shape: [1, 2, 3],
        scalars: [1, 2, 3, 4, 5, 6],
        scalarType: Float.self
    )
    let result = layerNorm(input)

    #expect(result.shape == [1, 2, 3])
    let resultArray = await result.shapedArray(of: Float.self).scalars
    let expectedArray: [Float] = [2.7753, 5.0000, 9.6742, 2.1011, 8.0000, 16.3484]
    #expect(allClose(resultArray, expectedArray) == true)
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
@Test func layerNorm3D() async {
    let weight = MLTensor(
        shape: [1, 2, 3],
        scalars: [1, 2, 3, 4, 5, 6],
        scalarType: Float.self
    )
    let bias = MLTensor(
        shape: [1, 2, 3],
        scalars: [4, 5, 6, 7, 8, 9],
        scalarType: Float.self
    )
    let layerNorm = layerNorm(weight: weight, bias: bias, epsilon: 1e-5)
    let input = MLTensor(
        shape: [1, 2, 3],
        scalars: [1, 2, 3, 4, 5, 6],
        scalarType: Float.self
    )
    let result = layerNorm(input)

    #expect(result.shape == [1, 2, 3])
    let resultArray = await result.shapedArray(of: Float.self).scalars
    let expectedArray: [Float] = [2.7753, 5.0000, 9.6742, 2.1011, 8.0000, 16.3484]
    #expect(allClose(resultArray, expectedArray) == true)
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
@Test func roPE() async {
    let input = MLTensor(
        shape: [3, 4],
        scalars: [1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10],
        scalarType: Float.self
    )
    let roPE = roPE(dims: 4, base: 10_000)
    let result = roPE(input)

    #expect(result.shape == [3, 4])
    let resultArray = await result.shapedArray(of: Float.self).scalars
    let expectedArray: [Float] =
        [
            1.0, 2.0, 3.0, 4.0,
            -2.88762, 4.92975, 6.6077, 7.04965,
            -11.0967, 7.79841, 2.61976, 10.158,
        ]
    #expect(allClose(resultArray, expectedArray) == true)
}
