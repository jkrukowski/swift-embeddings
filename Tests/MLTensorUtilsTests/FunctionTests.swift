import CoreML
import Numerics
import Testing
import TestingUtils

@testable import MLTensorUtils

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
@Test func additiveCasualMask() async {
    let result = additiveCausalMask(3)

    #expect(result.shape == [3, 3])
    let resultArray = await result.scalars(of: Float.self)
    let expectedArray: [Float] = [0, -1e9, -1e9, 0, 0, -1e9, 0, 0, 0]
    #expect(allClose(resultArray, expectedArray) == true)
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
@Test func norm() async {
    let x = MLTensor(
        shape: [2, 3],
        scalars: [1, 2, 3, 4, 5, 6],
        scalarType: Float.self
    )
    let result = norm(x)

    #expect(result.shape == [2])
    let resultArray = await result.scalars(of: Float.self)
    let expectedArray: [Float] = [3.7417, 8.7749]
    #expect(allClose(resultArray, expectedArray) == true)
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
@Test func cosineSimilarity1D() async {
    let x = MLTensor(
        shape: [1, 6],
        scalars: [1, 2, 3, 4, 5, 6],
        scalarType: Float.self
    )
    let y = MLTensor(
        shape: [1, 6],
        scalars: [4, 5, 6, 7, 8, 9],
        scalarType: Float.self
    )
    let result = cosineSimilarity(x, y)

    #expect(result.shape == [1, 1])
    let resultArray = await result.shapedArray(of: Float.self).scalars
    let expectedArray: [Float] = [0.980653]
    #expect(allClose(resultArray, expectedArray) == true)
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
@Test func cosineSimilarity2D() async {
    let x = MLTensor(
        shape: [2, 3],
        scalars: [1, 2, 3, 4, 5, 6],
        scalarType: Float.self
    )
    let y = MLTensor(
        shape: [2, 3],
        scalars: [4, 5, 6, 7, 8, 9],
        scalarType: Float.self
    )
    let result = cosineSimilarity(x, y)

    #expect(result.shape == [2, 2])
    let resultArray = await result.shapedArray(of: Float.self).scalars
    let expectedArray: [Float] = [0.9746318, 0.4090946, 2.3452077, 0.9981909]
    #expect(allClose(resultArray, expectedArray) == true)
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
@Test func cosineSimilaritySameTensor() async {
    let x = MLTensor(
        shape: [2, 3],
        scalars: [1, 2, 3, 1, 2, 3],
        scalarType: Float.self
    )
    let result = cosineSimilarity(x, x)

    #expect(result.shape == [2, 2])
    let resultArray = await result.shapedArray(of: Float.self).scalars
    let expectedArray: [Float] = [1.0, 1.0, 1.0, 1.0]
    #expect(allClose(resultArray, expectedArray) == true)
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
@Test func dotProduct1D() async {
    let x = MLTensor(
        shape: [6],
        scalars: [1, 2, 3, 4, 5, 6],
        scalarType: Float.self
    )
    let y = MLTensor(
        shape: [6],
        scalars: [4, 5, 6, 7, 8, 9],
        scalarType: Float.self
    )
    let result = dotProduct(x, y)

    #expect(result.shape == [])
    let resultArray = await result.shapedArray(of: Float.self).scalars
    let expectedArray: [Float] = [154]
    #expect(allClose(resultArray, expectedArray) == true)
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
@Test func dotProduct2D() async {
    let x = MLTensor(
        shape: [2, 2],
        scalars: [1, 0, 0, 1],
        scalarType: Float.self
    )
    let y = MLTensor(
        shape: [2, 2],
        scalars: [4, 5, 6, 7],
        scalarType: Float.self
    )
    let result = dotProduct(x, y)

    #expect(result.shape == [2, 2])
    let resultArray = await result.shapedArray(of: Float.self).scalars
    let expectedArray: [Float] = [4, 5, 6, 7]
    #expect(allClose(resultArray, expectedArray) == true)
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
@Test func euclideanDistance1D() async {
    let x = MLTensor(
        shape: [6],
        scalars: [1, 2, 3, 4, 5, 6],
        scalarType: Float.self
    )
    let y = MLTensor(
        shape: [6],
        scalars: [4, 5, 6, 7, 8, 9],
        scalarType: Float.self
    )
    let result = euclideanDistance(x, y, alongAxes: 0)

    #expect(result.shape == [])
    let resultArray = await result.shapedArray(of: Float.self).scalars
    let expectedArray: [Float] = [7.34846]
    #expect(allClose(resultArray, expectedArray) == true)
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
@Test func euclideanDistance2D() async {
    let x = MLTensor(
        shape: [2, 2],
        scalars: [1, 0, 0, 1],
        scalarType: Float.self
    )
    let y = MLTensor(
        shape: [2, 2],
        scalars: [4, 5, 6, 7],
        scalarType: Float.self
    )
    let result = euclideanDistance(x, y, alongAxes: 1)

    #expect(result.shape == [2])
    let resultArray = await result.shapedArray(of: Float.self).scalars
    let expectedArray: [Float] = [5.83095, 8.48528]
    #expect(allClose(resultArray, expectedArray) == true)
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
@Test func euclideanDistanceSameTensor() async {
    let x = MLTensor(
        shape: [6],
        scalars: [1, 2, 3, 4, 5, 6],
        scalarType: Float.self
    )
    let result = euclideanDistance(x, x, alongAxes: 0)

    #expect(result.shape == [])
    let resultArray = await result.shapedArray(of: Float.self).scalars
    let expectedArray: [Float] = [0.0]
    #expect(allClose(resultArray, expectedArray) == true)
}
