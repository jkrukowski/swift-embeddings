# Benchmarks

Benchmarks for `swift-embeddings`, built with
[package-benchmark](https://github.com/ordo-one/package-benchmark).

This is a separate Swift package (it depends on the parent package via a local
path) so that the benchmarking toolchain and its dependencies are kept out of
the main package graph.

## Running

From this directory:

```bash
swift package --disable-sandbox benchmark
```

The first run downloads a small static embedding model
(`minishlab/potion-base-8M`) from the Hugging Face Hub and caches it locally,
so subsequent runs do not hit the network during measurement.

### Useful commands

```bash
# List available benchmarks
swift package benchmark list

# Run and store a baseline
swift package --disable-sandbox benchmark baseline update main

# Compare against a stored baseline
swift package --disable-sandbox benchmark baseline compare main
```
