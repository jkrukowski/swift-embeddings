import Command
import CoreML
import MLTensorUtils
import Testing
import TestingUtils

@testable import Embeddings

struct SDPATensorData: Codable {
    let shape: [Int]
    let data: [Float]
}

struct SDPATestInput: Codable {
    let query: SDPATensorData
    let key: SDPATensorData
    let value: SDPATensorData
    let mask: SDPATensorData?
    let scale: Float?
}

struct SDPATestCase: Codable {
    let name: String
    let input: SDPATestInput
    let output: SDPATensorData
}

struct SDPATestData: Codable {
    let testCases: [SDPATestCase]
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
func loadSDPATestData() throws -> SDPATestData {
    let jsonUrl = try #require(
        Bundle.module.path(forResource: "sdpa", ofType: "json", inDirectory: "Data"),
        "sdpa.json not found"
    )
    let jsonData = try Data(contentsOf: URL(fileURLWithPath: jsonUrl))
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(SDPATestData.self, from: jsonData)
}

struct SDPAAccuracyTests {
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test("SDPA Accuracy", arguments: ["basic", "with_scale", "with_mask", "multi_head"])
    func sdpaAccuracy(testName: String) async throws {
        // Load test data from JSON
        let testData = try loadSDPATestData()
        let testCase = try #require(
            testData.testCases.first { $0.name == testName },
            "Test case '\(testName)' not found"
        )

        // Create input tensors from test data
        let query = MLTensor(
            shape: testCase.input.query.shape,
            scalars: testCase.input.query.data
        )
        let key = MLTensor(
            shape: testCase.input.key.shape,
            scalars: testCase.input.key.data
        )
        let value = MLTensor(
            shape: testCase.input.value.shape,
            scalars: testCase.input.value.data
        )

        let mask: MLTensor? =
            if let maskData = testCase.input.mask {
                MLTensor(shape: maskData.shape, scalars: maskData.data)
            } else {
                nil
            }

        // Run Swift SDPA
        let swiftOutput = sdpa(
            query: query,
            key: key,
            value: value,
            mask: mask,
            scale: testCase.input.scale
        )
        let swiftData = await swiftOutput.cast(to: Float.self).scalars(of: Float.self)

        // Compare with PyTorch output
        let pythonData = testCase.output.data

        #expect(allClose(pythonData, swiftData, absoluteTolerance: 1e-5) == true)
    }
}
