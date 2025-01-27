import CoreML
import Numerics
import Testing
import TestingUtils

@testable import MLTensorUtils

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
@Test func erf() async {
    let input1 = MLTensor(shape: [1], scalars: [0], scalarType: Float.self)
    let result1 = await erf(input1).shapedArray(of: Float.self).scalars
    #expect(result1 == [0])

    let input2 = MLTensor(shape: [1], scalars: [.infinity], scalarType: Float.self)
    let result2 = await erf(input2).shapedArray(of: Float.self).scalars
    #expect(result2 == [1])

    let input3 = MLTensor(shape: [1], scalars: [-.infinity], scalarType: Float.self)
    let result3 = await erf(input3).shapedArray(of: Float.self).scalars
    #expect(result3 == [-1])

    let input4 = MLTensor(
        shape: [6], scalars: [0.9, 0.5, 0.1, -0.1, -0.5, -0.9], scalarType: Float.self)
    let result4 = await erf(input4).shapedArray(of: Float.self).scalars
    let expected4: [Float] = [
        0.7969082124228322,
        0.5204998778130465,
        0.1124629160182849,
        -0.1124629160182849,
        -0.5204998778130465,
        -0.7969082124228322,
    ]
    #expect(allClose(result4, expected4) == true)
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
@Test func gelu() async {
    let input1 = MLTensor(shape: [1], scalars: [0], scalarType: Float.self)
    let result1 = await gelu(input1).shapedArray(of: Float.self).scalars
    #expect(result1 == [0])

    let input2 = MLTensor(shape: [1], scalars: [100], scalarType: Float.self)
    let result2 = await gelu(input2).shapedArray(of: Float.self).scalars
    #expect(result2 == [100])

    let input3 = MLTensor(shape: [1], scalars: [-100], scalarType: Float.self)
    let result3 = await gelu(input3).shapedArray(of: Float.self).scalars
    #expect(result3 == [0])

    let input4 = MLTensor(
        shape: [6], scalars: [0.9, 0.5, 0.1, -0.1, -0.5, -0.9], scalarType: Float.self)
    let result4 = await gelu(input4).shapedArray(of: Float.self).scalars
    let expected4: [Float] = [
        0.734346, 0.345731, 0.0539828, -0.0460172, -0.154269, -0.165654,
    ]
    #expect(allClose(result4, expected4) == true)
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
@Test func geluApproximationFast() async {
    let input1 = MLTensor(shape: [1], scalars: [0], scalarType: Float.self)
    let result1 = await gelu(input1, approximation: .fast).shapedArray(of: Float.self).scalars
    #expect(result1 == [0])

    let input2 = MLTensor(shape: [1], scalars: [100], scalarType: Float.self)
    let result2 = await gelu(input2, approximation: .fast).shapedArray(of: Float.self).scalars
    #expect(result2 == [100])

    let input3 = MLTensor(shape: [1], scalars: [-100], scalarType: Float.self)
    let result3 = await gelu(input3, approximation: .fast).shapedArray(of: Float.self).scalars
    #expect(result3 == [0])

    let input4 = MLTensor(
        shape: [6], scalars: [0.9, 0.5, 0.1, -0.1, -0.5, -0.9], scalarType: Float.self)
    let result4 = await gelu(input4, approximation: .fast).shapedArray(of: Float.self).scalars
    let expected4: [Float] = [
        0.740043, 0.350388, 0.0542448, -0.0457552, -0.149612, -0.159957,
    ]
    #expect(allClose(result4, expected4) == true)
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
@Test func geluApproximationPrecise() async {
    let input1 = MLTensor(shape: [1], scalars: [0], scalarType: Float.self)
    let result1 = await gelu(input1, approximation: .precise).shapedArray(of: Float.self).scalars
    #expect(result1 == [0])

    let input2 = MLTensor(shape: [1], scalars: [100], scalarType: Float.self)
    let result2 = await gelu(input2, approximation: .precise).shapedArray(of: Float.self).scalars
    #expect(result2 == [100])

    let input3 = MLTensor(shape: [1], scalars: [-100], scalarType: Float.self)
    let result3 = await gelu(input3, approximation: .precise).shapedArray(of: Float.self).scalars
    #expect(result3 == [0])

    let input4 = MLTensor(
        shape: [6], scalars: [0.9, 0.5, 0.1, -0.1, -0.5, -0.9], scalarType: Float.self)
    let result4 = await gelu(input4, approximation: .precise).shapedArray(of: Float.self).scalars
    let expected4: [Float] = [
        0.734228, 0.345714, 0.0539828, -0.0460172, -0.154286, -0.165772,
    ]
    #expect(allClose(result4, expected4) == true)
}
