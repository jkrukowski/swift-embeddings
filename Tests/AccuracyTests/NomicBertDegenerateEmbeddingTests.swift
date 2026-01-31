import CoreML
import MLTensorUtils
import Testing
import TestingUtils

@testable import Embeddings

@Suite(
    .enabled(if: ProcessInfo.processInfo.environment["UV_PATH"] != nil),
    .downloadModel(modelId: Utils.ModelId.nomicEmbedTextV15, downloadBase: Utils.modelPath)
)
struct NomicBertDegenerateEmbeddingTests {
    private static let suspiciousThreshold: Float = 0.999

    private func prefixed(_ text: String) -> String {
        "search_document: \(text)"
    }

    @available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
    @Test("NomicBert Degenerate Embedding Detection (Swift vs Python)")
    func nomicBertDegenerateEmbeddings() async throws {
        let texts = Self.affectedEntityNames.map(prefixed)

        let modelBundle = try await NomicBert.loadModelBundle(
            from: Utils.ModelId.nomicEmbedTextV15,
            downloadBase: Utils.modelPath
        )
        let encoded = try modelBundle.batchEncode(
            texts,
            postProcess: .meanPool,
            computePolicy: .cpuOnly
        )
        let swiftData = await encoded.cast(to: Float.self).scalars(of: Float.self)
        let swiftEmbeddings = try reshape(
            data: swiftData,
            count: texts.count,
            dimension: encoded.shape.last ?? 0,
            label: "Swift"
        )

        let modelPath = modelPath(
            modelId: Utils.ModelId.nomicEmbedTextV15, cacheDirectory: Utils.modelPath)
        let pythonData = try await generateUsingTransformers(
            modelPath: modelPath,
            texts: texts,
            modelType: .nomic
        )
        let pythonEmbeddings = try reshape(
            data: pythonData,
            count: texts.count,
            dimension: swiftEmbeddings.first?.count ?? 0,
            label: "Python"
        )

        try assertNoDegeneratePairs(
            embeddings: swiftEmbeddings,
            names: Self.affectedEntityNames,
            label: "Swift"
        )
        try assertNoDegeneratePairs(
            embeddings: pythonEmbeddings,
            names: Self.affectedEntityNames,
            label: "Python"
        )
    }

    private func reshape(
        data: [Float],
        count: Int,
        dimension: Int,
        label: String
    ) throws -> [[Float]] {
        guard dimension > 0 else {
            #expect(
                Bool(false),
                Comment(rawValue: "\(label) embeddings have invalid dimension \(dimension)")
            )
            return []
        }
        let expected = count * dimension
        guard data.count == expected else {
            #expect(
                Bool(false),
                Comment(rawValue:
                    "\(label) embeddings shape mismatch: got \(data.count) values " +
                        "(expected \(expected) for \(count) x \(dimension))"
                )
            )
            return []
        }
        var result: [[Float]] = []
        result.reserveCapacity(count)
        for i in 0..<count {
            let start = i * dimension
            result.append(Array(data[start..<start + dimension]))
        }
        return result
    }

    private func assertNoDegeneratePairs(
        embeddings: [[Float]],
        names: [String],
        label: String
    ) throws {
        for i in 0..<embeddings.count {
            for j in (i + 1)..<embeddings.count {
                let isIdentical = embeddings[i].elementsEqual(embeddings[j])
                let similarity = cosineSimilarity(embeddings[i], embeddings[j])
                if isIdentical || similarity > Self.suspiciousThreshold {
                    let simString = String(format: "%.6f", similarity)
                    #expect(
                        Bool(false),
                        Comment(rawValue:
                            "\(label) degenerate embeddings: '\(names[i])' ↔ '\(names[j])' " +
                                "sim=\(simString)"
                        )
                    )
                }
            }
        }
    }

    private func cosineSimilarity(_ x: [Float], _ y: [Float]) -> Float {
        var dot: Float = 0
        var normX: Float = 0
        var normY: Float = 0
        for i in 0..<min(x.count, y.count) {
            let xv = x[i]
            let yv = y[i]
            dot += xv * yv
            normX += xv * xv
            normY += yv * yv
        }
        let denom = (sqrt(normX) * sqrt(normY))
        if denom == 0 { return 0 }
        return dot / denom
    }
}

extension NomicBertDegenerateEmbeddingTests {
    static let affectedEntityNames: [String] = [
        "Dr. Tian Jishun", "Dr. Melanie Hoenig", "Zhang Chao", "Jack Ma",
        "DeepSeek R1", "Baichuan AI", "Wei Lijia", "Real Kuang",
        "Wang Xiaochuan", "Shreya Johri", "Zhang Jiansheng", "Andrew Bean",
        "Lu Tang", "Synyi AI", "Influencers on WeChat", "MRI scans",
        "Qt Creator", "Prt Sc", "Mouse Wheel", "Desktop Environment",
        "System Settings", "Custom Shortcuts", "Keyboard Shortcuts",
        "Screenshot History", "Prebuilt Packages", "Right Click",
        "Flameshot GUI", "Plasma Wayland", "Gnome Wayland",
        "Microsoft Windows", "Open Anyway",
        "C2 Server", "Contabo GmbH", "C2 Backdoor", "Contabo VPS",
        "Coalfire Labs", "Raw JSON", "Tenuo Warrants",
        "Infrared Sounder", "OHB Systems", "Middle East", "Sahara Desert",
        "Simonetta Cheli", "Second Imager", "European Commission",
        "Tango Dark", "Tango Light", "Solarized Dark", "Solarized Light",
        "Ethan Schoonover",
        "Cloudflare Workers", "Browser Rendering", "Google Maps",
        "AI Search", "Cloudflare Access", "Agents SDK", "Puppeteer APIs",
        "Admin UI", "Cloudflare Containers", "AI Agents",
        "AWS SES", "Prompt Injection", "O365 GCC", "Mail Agent",
        "Subscription Billing", "Developer Platform", "Sandbox SDK",
        "OTEL SDK", "Universal SDKs", "AI Gateway", "Unified Billing",
        "Claude Code", "Madhu Gottumukkala", "Waymo AV", "James Champion",
        "Google DeepMind", "José Ralat", "Rodrigo Bravo", "Instagram Reels",
        "Project Genie", "SignPath Foundation", "Void Linux",
        "EmulatorJS", "Sega CD", "Mega Drive", "Game Gear", "Master System",
        "Atari Lynx", "PlayStation Portable", "Sega 32X", "Nintendo DS",
        "Virtual Boy", "Sega Saturn", "Game Boy", "Atari Jaguar",
        "Commodore PET", "Commodore Amiga", "Code Generator",
        "Trey Harris", "LoongArch ISA", "Loongson 3A6000", "AMD GPU",
        "C++20 Modules", "Generative AI", "Sendmail 8", "Big 8",
        "Agile teams", "Luleå University of Technology",
        "Concordia University", "Valery Fabrikant", "Alexander Abian",
        "EA App", "Rockstar Launcher", "Joe Talmadge", "Xiaolin Li",
    ]
}
