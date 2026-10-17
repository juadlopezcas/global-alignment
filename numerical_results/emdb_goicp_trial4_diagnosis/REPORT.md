# Trial 4 (index 3): native Go-ICP rotation roundoff

Only Go-ICP at trial index 3 was reproduced, using seed 42 and the exact
configuration and thread environment in the interrupted benchmark metadata.
All 12 saved benchmark rows (indices 0-2, all four methods) remain untouched.
The full 20-trial benchmark was not started.

## Raw capture before strict SO(3) validation

```text
R_raw =
[[-0.314992674441463, -0.000403696961494, -0.949093957965121],
 [ 0.551196810026136,  0.813995560619373, -0.183281756191368],
 [ 0.772632796364006, -0.580870041112020, -0.256180311446252]]
```

The wrapper return is a list of lists of Python `float`; its inferred NumPy
dtype is `float64`. Native computation nevertheless includes `ICP3D<float>` and
float-valued branch-and-bound rotation entries.

| Diagnostic | Raw value |
| --- | --- |
| Orthogonality Frobenius error | 1.193220147751071e-6 |
| Maximum absolute orthogonality error | 1.0596592356915124e-6 |
| Determinant | 0.9999992909318903 |
| Singular values | 0.9999999523320925, 0.9999999290558964, 0.9999994095438277 |

The original strict check failed as reported. Its maximum residual exceeded
the absolute 1e-6 orthogonality threshold by approximately 5.97%. These
residuals support a tiny numerical cleanup; they do not suggest a convention
error. The actual diagnostics were reported before choosing the guard.

## Explicit SVD cleanup

The new policy is Go-ICP-specific. Finite 3x3 raw matrices must have positive
determinant and Frobenius/max orthogonality residuals, absolute determinant
deviation, and all singular-value deviations at most **5e-6**. Only a matrix
that fails the original strict check is projected. The projection distance
must also be at most **5e-6**; the result then passes the original SO(3) check.
Matrices already accepted by that check are preserved exactly. Reflections,
nonfinite matrices, and material scale/shear are rejected, with raw diagnostics.
This policy has no ground-truth inputs and never changes the native search.

For this raw matrix, `U @ diag(1,1,sign(det(U@Vt))) @ Vt` gives:

| Quantity | Before | After |
| --- | --- | --- |
| Frobenius registration error | 0.029475347585678893 | 0.029475350530646358 |
| Normalized Frobenius error | 0.017017599863049494 | 0.017017601563327253 |
| Angular diagnostic / benchmark error (degrees) | 1.1951670298705068 | 1.1941928707868792 |
| Orthogonality Frobenius error | 1.193220147751071e-6 | 1.1267533089989422e-15 |
| Maximum orthogonality error | 1.0596592356915124e-6 | 8.881784197001252e-16 |
| Determinant | 0.9999992909318903 | 1.0000000000000009 |

**Projection distance: 5.966102471354767e-7.** No transpose or inverse is used.
The raw angle is the benchmark trace expression applied diagnostically, with
cosine 0.999782446437173 (inside [-1,1]); it is not a rigorous SO(3) geodesic
angle for a non-orthogonal input. The projected angle is the unchanged common
benchmark evaluation on a proper rotation.

The native final SSE remains 3.1291933059692383. Native DT construction took
11.0802343 seconds and search took 112.2295286 seconds. The diagnostic invocation
took 123.3341911 seconds including capture. Cleanup validation reused the
captured native result; no second native fit was needed.

## Preservation, tests, and manual continuation

All 109 protected pre-existing data, result and source files matched their
pre-diagnosis hashes. The C++ registration implementation and Windows build
patches were not changed. Ours, ICP, CPD, RNG, preprocessing, shared error
metrics, strict validation tolerance, and all Go-ICP search parameters remain
unchanged. Raw diagnostics and cleanup decisions are now part of Go-ICP CSV
rows, with explicit failure JSON for rejected matrices.

All 15 targeted cleanup/adapter/resume tests passed, using saved or mocked
native returns only. Tests cover the exact trial-4 matrix, preserving accepted
rotations, rejecting reflections/scale/shear/nonfinite matrices, fixed guard
limits, and resuming only missing trials without modifying the input CSV.

The actual 12-row prefix was also checked: Go-ICP indices 0, 1 and 2 require
no projection, so their existing common metrics remain unchanged. It is safe
to resume under the guarded policy; future larger deviations still stop for
review. `GOICP_SETUP.md` contains the exact manual PowerShell command with
`--resume-from`, fixed search parameters, the saved convention validation, and
a new output directory. That command resumes at index 3 and was not executed.

Files here include the full raw capture, projection analysis, a one-row
diagnostic CSV, and the policy check on the three previously saved rotations.
