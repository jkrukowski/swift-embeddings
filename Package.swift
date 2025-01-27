// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "swift-embeddings",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .tvOS(.v17),
        .visionOS(.v1),
        .watchOS(.v10),
    ],
    products: [
        .executable(
            name: "embeddings-cli",
            targets: ["EmbeddingsCLI"]
        ),
        .library(
            name: "Embeddings",
            targets: ["Embeddings"]),
        .library(
            name: "MLTensorUtils",
            targets: ["MLTensorUtils"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/apple/swift-numerics.git",
            from: "1.0.2"
        ),
        .package(
            url: "https://github.com/huggingface/swift-transformers.git",
            from: "0.1.14"
        ),
        .package(
            url: "https://github.com/jkrukowski/swift-safetensors.git",
            from: "0.0.7"
        ),
        .package(
            url: "https://github.com/apple/swift-argument-parser.git",
            from: "1.5.0"
        ),
        .package(
            url: "https://github.com/jkrukowski/swift-sentencepiece",
            from: "0.0.5"
        ),
        .package(
            url: "https://github.com/tuist/Command.git",
            from: "0.11.16"
        ),
    ],
    targets: [
        .executableTarget(
            name: "EmbeddingsCLI",
            dependencies: [
                "Embeddings",
                "MLTensorUtils",
                .product(name: "Safetensors", package: "swift-safetensors"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .target(
            name: "Embeddings",
            dependencies: [
                "MLTensorUtils",
                .product(name: "Safetensors", package: "swift-safetensors"),
                .product(name: "Transformers", package: "swift-transformers"),
                .product(name: "SentencepieceTokenizer", package: "swift-sentencepiece"),
            ]
        ),
        .target(
            name: "MLTensorUtils"),
        .target(
            name: "TestingUtils",
            dependencies: [
                .product(name: "Numerics", package: "swift-numerics")
            ]
        ),
        .testTarget(
            name: "EmbeddingsTests",
            dependencies: [
                "Embeddings",
                "MLTensorUtils",
                "TestingUtils",
                .product(name: "Safetensors", package: "swift-safetensors"),
            ],
            resources: [
                .copy("Resources")
            ]
        ),
        .testTarget(
            name: "AccuracyTests",
            dependencies: [
                "Embeddings",
                "TestingUtils",
                .product(name: "Command", package: "Command"),
            ],
            resources: [
                .copy("Scripts")
            ]
        ),
        .testTarget(
            name: "MLTensorUtilsTests",
            dependencies: [
                "MLTensorUtils",
                "TestingUtils",
            ]
        ),
    ]
)
