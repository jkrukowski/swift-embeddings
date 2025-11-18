import CoreML
import Numerics
import Testing

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension MLTensor {
    package func scalars<Scalar>(
        of scalarType: Scalar.Type
    ) async -> [Scalar] where Scalar: MLShapedArrayScalar, Scalar: MLTensorScalar {
        await shapedArray(of: scalarType).scalars
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension MLTensor {
    package static func float(shape: [Int]) -> MLTensor {
        let count = shape.reduce(1, *)
        return MLTensor(shape: shape, scalars: (0..<count).map { Float($0) })
    }

    package static func int32(shape: [Int]) -> MLTensor {
        let count = shape.reduce(1, *)
        return MLTensor(shape: shape, scalars: (0..<count).map { Int32($0) })
    }
}

package func allClose<T: Numeric>(
    _ lhs: [T],
    _ rhs: [T],
    absoluteTolerance: T.Magnitude = T.Magnitude.ulpOfOne.squareRoot()
        * T.Magnitude.leastNormalMagnitude,
    relativeTolerance: T.Magnitude = T.Magnitude.ulpOfOne.squareRoot()
) -> Bool where T.Magnitude: FloatingPoint {
    guard lhs.count == rhs.count else {
        Issue.record("Sizes differ: \(lhs.count) vs. \(rhs.count)")
        return false
    }
    for (l, r) in zip(lhs, rhs) {
        guard
            l.isApproximatelyEqual(
                to: r,
                absoluteTolerance: absoluteTolerance,
                relativeTolerance: relativeTolerance
            )
        else {
            Issue.record("Expected \(lhs) to be approximately equal to \(rhs), but \(l) != \(r)")
            return false
        }
    }
    return true
}
