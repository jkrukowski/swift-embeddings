import CoreML
import Numerics
import Testing

extension MLTensor {
    package func scalars<Scalar>(
        of scalarType: Scalar.Type
    ) async -> [Scalar] where Scalar: MLShapedArrayScalar, Scalar: MLTensorScalar {
        await shapedArray(of: scalarType).scalars
    }
}

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
        Issue.record("Expected \(lhs) to be approximately equal to \(rhs), but sizes differ")
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
