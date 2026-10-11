# NVIDIA nccl-tests source

`nccl-tests-v2.17.6.tar.gz` is the unmodified NVIDIA source archive at commit
`da0b547b1b9c6e3b1d4c15578087874522ae3761` (tag `v2.17.6`).

- Source: https://github.com/NVIDIA/nccl-tests/tree/da0b547b1b9c6e3b1d4c15578087874522ae3761
- Download: https://codeload.github.com/NVIDIA/nccl-tests/tar.gz/da0b547b1b9c6e3b1d4c15578087874522ae3761
- SHA256: `d589c597110da30b52d462269ae0d0381c46811a7b266db09fe865d9aaff31c1`
- Copyright: NVIDIA CORPORATION. BSD license in the archive's `LICENSE.txt`.

The image build verifies the checksum, retains the source and license under
`/opt/nccl-tests`, and compiles against the NCCL already loaded by PyTorch, or
an existing shared library of exactly the same version when it is statically linked.
The source commit is fixed; the reference publisher did not disclose its nccl-tests
version. This difference is visible in the report.
