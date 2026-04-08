# Reconstructed CKPFCM Code

This package is a clean-room reconstruction written from the legacy manuscript narrative. It exists so the project can run end-to-end again.

## Scope

- synthetic benchmark generator for CAO, GU, and WEI-style cases
- segmentation accuracy with label-permutation matching
- lightweight NumPy implementations of:
  - FCM
  - PFCM
  - KPFCM
  - C-PFCM
  - C-KPFCM

## Important limitation

This is not the original implementation from the paper workspace. Therefore:

- the numerical results produced by this package are useful for testing and future redevelopment;
- they should not be presented as the original manuscript's exact results;
- they are a reconstruction baseline from which a closer reproduction can be developed.
