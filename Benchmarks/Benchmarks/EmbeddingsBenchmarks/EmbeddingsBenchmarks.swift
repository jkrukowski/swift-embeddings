import Benchmark
import Embeddings

private enum ModelId {
    static let bert = "sentence-transformers/all-MiniLM-L6-v2"
    static let qwen3 = "jkrukowski/Qwen3-Embedding-0.6B-F32"
    static let modernBert = "answerdotai/ModernBERT-base"
    static let nomicBert = "nomic-ai/nomic-embed-text-v1.5"
    static let xlmRoberta = "FacebookAI/xlm-roberta-base"
    static let roberta = "FacebookAI/roberta-base"
    static let clip = "jkrukowski/clip-vit-base-patch32"
    static let model2Vec = "minishlab/potion-base-2M"
    static let staticEmbeddings = "sentence-transformers/static-retrieval-mrl-en-v1"
}

private let sampleText = "The quick brown fox jumps over the lazy dog."

let benchmarks: @Sendable () -> Void = {
    Benchmark.defaultConfiguration = .init(
        metrics: [
            .wallClock,
            .throughput,
            .peakMemoryResident,
            .mallocCountTotal,
        ],
        maxIterations: 100
    )

    Benchmark(
        "Load Bert ModelBundle"
    ) { benchmark in
        for _ in benchmark.scaledIterations {
            blackHole(
                try await Bert.loadModelBundle(from: ModelId.bert)
            )
        }
    }

    Benchmark("Encode (Bert)") { benchmark, bundle in
        for _ in benchmark.scaledIterations {
            blackHole(try bundle.encode(sampleText))
        }
    } setup: {
        try await Bert.loadModelBundle(from: ModelId.bert)
    }

    Benchmark("Batch encode (Bert)") { benchmark, bundle in
        let sampleBatch = [
            "The quick brown fox jumps over the lazy dog.",
            "A journey of a thousand miles begins with a single step.",
            "To be or not to be, that is the question.",
            "All that glitters is not gold.",
            "The early bird catches the worm.",
            "Actions speak louder than words.",
            "Better late than never, but never late is better.",
            "A picture is worth a thousand words.",
        ]
        for _ in benchmark.scaledIterations {
            blackHole(try bundle.batchEncode(sampleBatch))
        }
    } setup: {
        try await Bert.loadModelBundle(from: ModelId.bert)
    }

    Benchmark(
        "Encode (Qwen3)",
        configuration: .init(
            maxDuration: .seconds(20),
            maxIterations: 50
        )
    ) { benchmark, bundle in
        for _ in benchmark.scaledIterations {
            blackHole(try bundle.encode(sampleText))
        }
    } setup: {
        try await Qwen3.loadModelBundle(from: ModelId.qwen3)
    }

    Benchmark("Encode (ModernBert)") { benchmark, bundle in
        for _ in benchmark.scaledIterations {
            blackHole(try bundle.encode(sampleText))
        }
    } setup: {
        try await ModernBert.loadModelBundle(
            from: ModelId.modernBert,
            loadConfig: .addWeightKeyPrefix("model.")
        )
    }

    Benchmark("Encode (NomicBert)") { benchmark, bundle in
        for _ in benchmark.scaledIterations {
            blackHole(try bundle.encode(sampleText))
        }
    } setup: {
        try await NomicBert.loadModelBundle(from: ModelId.nomicBert)
    }

    Benchmark("Encode (XLMRoberta)") { benchmark, bundle in
        for _ in benchmark.scaledIterations {
            blackHole(try bundle.encode(sampleText))
        }
    } setup: {
        try await XLMRoberta.loadModelBundle(
            from: ModelId.xlmRoberta,
            loadConfig: .addWeightKeyPrefix("roberta.")
        )
    }

    Benchmark("Encode (Roberta)") { benchmark, bundle in
        for _ in benchmark.scaledIterations {
            blackHole(try bundle.encode(sampleText))
        }
    } setup: {
        try await Roberta.loadModelBundle(
            from: ModelId.roberta,
            loadConfig: .addWeightKeyPrefix("roberta.")
        )
    }

    Benchmark("Encode (Clip)") { benchmark, bundle in
        for _ in benchmark.scaledIterations {
            blackHole(try bundle.encode(sampleText))
        }
    } setup: {
        try await Clip.loadModelBundle(from: ModelId.clip)
    }

    Benchmark("Encode (Model2Vec)") { benchmark, bundle in
        for _ in benchmark.scaledIterations {
            blackHole(try bundle.encode(sampleText))
        }
    } setup: {
        try await Model2Vec.loadModelBundle(from: ModelId.model2Vec)
    }

    Benchmark("Encode (StaticEmbeddings)") { benchmark, bundle in
        for _ in benchmark.scaledIterations {
            blackHole(try bundle.encode(sampleText))
        }
    } setup: {
        try await StaticEmbeddings.loadModelBundle(
            from: ModelId.staticEmbeddings,
            loadConfig: .staticEmbeddings
        )
    }
}
