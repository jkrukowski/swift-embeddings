// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "benchmarks",
    platforms: [
        .macOS(.v15),
    ],
    dependencies: [
        .package(name: "swift-embeddings", path: ".."),
        .package(url: "https://github.com/ordo-one/package-benchmark", from: "1.29.7"),
    ],
    targets: [
        .executableTarget(
            name: "EmbeddingsBenchmarks",
            dependencies: [
                .product(name: "Embeddings", package: "swift-embeddings"),
                .product(name: "Benchmark", package: "package-benchmark"),
            ],
            path: "Benchmarks/EmbeddingsBenchmarks",
            plugins: [
                .plugin(name: "BenchmarkPlugin", package: "package-benchmark")
            ]
        )
    ]
)
