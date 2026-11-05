# Candidate pretrained models for the GPU experiment

These are candidates, not a verified matched checkpoint set.
These checkpoints have not been evaluated in this project.

## Original and retrained control

[jaeunglee/resnet18-cifar10-unlearning](https://huggingface.co/jaeunglee/resnet18-cifar10-unlearning)
documents a full CIFAR-10 ResNet-18 and class-excluded models trained from
scratch. The file listing confirms `resnet18_cifar10_full.pth` and
`resnet18_cifar10_no_airplane.pth`, each about 44.8 MB. The documented network
uses a 3x3 stride-1 first convolution, no initial max-pool, and ten outputs.
Normalization is mean `(0.4914, 0.4822, 0.4465)`, std
`(0.2023, 0.1994, 0.2010)`; this differs from the CIFAR-10 training configuration
used here.

These class-excluded files are retrained controls, not approximate unlearning
checkpoints. Their original initialization seeds and detailed dataset indices
still need verification.

## Candidate unlearned model

[Yurim0507/SALUN](https://huggingface.co/Yurim0507/SALUN) documents CIFAR-10
ResNet-18 class-unlearned checkpoints, including
`cifar10_resnet18_SALUN_0.pth` for airplane. Its model tree links to the above
base-model repository. That is evidence of intended provenance, not proof
of a particular original checkpoint hash. The card describes a short
unlearning procedure and reports class-specific results. Exact preprocessing,
checkpoint format, the claimed SalUn implementation, and forgotten-example
semantics need verification in the source and by reproduction. The available
weight files also need verification.

## Practical next step

For airplane, investigate the original, `SALUN_0`, and `no_airplane` files as
one candidate trio. First inspect state-dict names/shapes and load tensor data
with `torch.load(..., weights_only=True)`, then reproduce ordinary inference
metrics under the published preprocessing. Record revision IDs and SHA-256
hashes. Check the unlearned model's actual base checkpoint and keep its
label mapping unchanged. A successful load alone does not establish a match.

`targeted_flips_gpu.py --stage search` expects the project's checkpoint schema
and split metadata, so these files cannot be used directly. An explicit
import/evaluation adapter is required. These models could save original and
control training time if provenance and baseline reproduction succeed.
