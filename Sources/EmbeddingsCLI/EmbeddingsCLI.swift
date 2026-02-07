import ArgumentParser
import Foundation

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension CommandConfiguration {
    fileprivate static let embeddingsCLISupported = CommandConfiguration(
        abstract: "Encode text using embedding model",
        subcommands: [
            BertCommand.self,
            ClipCommand.self,
            Model2VecCommand.self,
            ModernBertCommand.self,
            NomicBertCommand.self,
            RobertaCommand.self,
            StaticEmbeddingsCommand.self,
            Word2VecCommand.self,
            XLMRobertaCommand.self,
        ]
    )
}

extension CommandConfiguration {
    fileprivate static let embeddingsCLIUnsupported = CommandConfiguration(
        abstract: "Encode text using embedding model (requires macOS 15 or later)",
        subcommands: [
            UnsupportedStub.self
        ]
    )
}

@main
struct EmbeddingsCLI: AsyncParsableCommand {
    static let configuration: CommandConfiguration = {
        if #available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *) {
            CommandConfiguration.embeddingsCLISupported
        } else {
            CommandConfiguration.embeddingsCLIUnsupported
        }
    }()
}

private struct UnsupportedStub: AsyncParsableCommand {
    func run() async throws {
        fputs("This command requires macOS 15 or later.\n", stderr)
        throw ExitCode.failure
    }
}
