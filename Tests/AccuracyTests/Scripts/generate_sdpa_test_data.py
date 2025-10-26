# /// script
# requires-python = "==3.12"
# dependencies = [
#     "torch",
#     "numpy",
# ]
# ///

"""
Script to generate SDPA test data using PyTorch and save to JSON.
"""

import json
import torch
import torch.nn.functional as F


def sdpa_test_case(
    name, query_shape, key_shape, value_shape, mask_shape=None, scale=None
):
    """Generate a single SDPA test case."""
    # Generate input tensors
    query_size = 1
    for dim in query_shape:
        query_size *= dim
    query = torch.arange(query_size, dtype=torch.float32).reshape(query_shape)

    key_size = 1
    for dim in key_shape:
        key_size *= dim
    key = torch.arange(key_size, dtype=torch.float32).reshape(key_shape) * 0.5

    value_size = 1
    for dim in value_shape:
        value_size *= dim
    value = torch.arange(value_size, dtype=torch.float32).reshape(value_shape) * 0.1

    mask = None
    if mask_shape:
        # Create a causal mask for testing
        seq_len = mask_shape[0]
        mask_data = []
        for i in range(seq_len):
            row = [0.0] * (i + 1) + [-1e9] * (seq_len - i - 1)
            mask_data.extend(row)
        mask = torch.tensor(mask_data, dtype=torch.float32).reshape(mask_shape)

    # Run PyTorch SDPA
    output = F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, scale=scale
    )

    # Build test case dictionary
    test_case = {
        "name": name,
        "input": {
            "query": {"shape": list(query_shape), "data": query.flatten().tolist()},
            "key": {"shape": list(key_shape), "data": key.flatten().tolist()},
            "value": {"shape": list(value_shape), "data": value.flatten().tolist()},
        },
        "output": {"shape": list(output.shape), "data": output.flatten().tolist()},
    }

    if mask is not None:
        test_case["input"]["mask"] = {
            "shape": list(mask.shape),
            "data": mask.flatten().tolist(),
        }

    if scale is not None:
        test_case["input"]["scale"] = scale

    return test_case


def main():
    test_cases = []

    # Test 1: Basic SDPA test
    test_cases.append(
        sdpa_test_case(
            name="basic",
            query_shape=[1, 1, 2, 4],
            key_shape=[1, 1, 2, 4],
            value_shape=[1, 1, 2, 4],
        )
    )

    # Test 2: SDPA with custom scale
    test_cases.append(
        sdpa_test_case(
            name="with_scale",
            query_shape=[1, 1, 3, 8],
            key_shape=[1, 1, 3, 8],
            value_shape=[1, 1, 3, 8],
            scale=0.25,
        )
    )

    # Test 3: SDPA with attention mask
    test_cases.append(
        sdpa_test_case(
            name="with_mask",
            query_shape=[1, 1, 4, 8],
            key_shape=[1, 1, 4, 8],
            value_shape=[1, 1, 4, 8],
            mask_shape=[4, 4],
        )
    )

    # Test 4: Multi-head SDPA
    test_cases.append(
        sdpa_test_case(
            name="multi_head",
            query_shape=[2, 4, 3, 8],
            key_shape=[2, 4, 3, 8],
            value_shape=[2, 4, 3, 8],
        )
    )

    output = {"test_cases": test_cases}
    with open("sdpa.json", "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
