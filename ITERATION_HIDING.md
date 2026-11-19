# Iteration-hiding Nova

This branch modifies Nova so that a proof does not reveal how many steps of the
computation it covers.

In standard Nova the verifier is given the step count `i` and checks a proof of
"`z_i = F^i(z_0)`". Here the verifier learns only that **some** number of steps
below a fixed bound `N` was executed: the statement becomes
"there exists `i < N` such that `z_i = F^i(z_0)`".

The construction follows the iteration-hiding IVC of
[ePrint 2025/767](https://eprint.iacr.org/2025/767).

It is based on `nova-snark` 0.76.0 (upstream `microsoft/Nova` at `01b1e2d`) and
changes only the `nova` module. The `neutron` module is untouched.

> **Research code.** This has not been audited. Read [Limitations](#limitations)
> before relying on it.

## How it works

Standard Nova's augmented circuit carries the step count in its running hash,
`H(pp, i, z_0, z_i, U, r)`, and the verifier recomputes that hash from the
claimed `i`.

This branch replaces `i` in that hash with a hiding commitment:

```text
com  = H(i, rho)                      rho fresh and uniform at every step
hash = H(pp, z_0, z_i, U, r, N, com)
```

- `i` and `rho` are private witnesses of the augmented circuit. The circuit
  still enforces `i_next = i + 1` and the base case `i = 0`.
- The circuit enforces `i < N`. `N` is absorbed into the hash, so the verifier
  fixes its value.
- The proof carries `com` for each curve of the cycle. The verifier recomputes
  the hashes from `com` and never sees `i`.

`H` is the engine's circuit-friendly hash (Poseidon), truncated to 250 bits.

## Using it

Add the branch as a git dependency:

```toml
[dependencies]
nova-snark = { git = "https://github.com/jiwonkimpark/Nova", branch = "iter-hiding" }
```

The API is the same as upstream Nova's. Step circuits need no changes.

```rust
use nova_snark::{
  nova::{CompressedSNARK, PublicParams, RecursiveSNARK},
  provider::{PallasEngine, VestaEngine},
  traits::snark::default_ck_hint,
};

type E1 = PallasEngine;
type E2 = VestaEngine;
type EE<E> = nova_snark::provider::ipa_pc::EvaluationEngine<E>;
type S<E> = nova_snark::spartan::snark::RelaxedR1CSSNARK<E, EE<E>>;

// `circuit` implements `StepCircuit`, `z0` is the initial input
let pp = PublicParams::<E1, E2, _>::setup(&circuit, &*default_ck_hint(), &*default_ck_hint())?;

let mut recursive_snark = RecursiveSNARK::new(&pp, &circuit, &z0)?;
for _ in 0..num_steps {
  recursive_snark.prove_step(&pp, &circuit)?;
}

// The compressed proof is the one to publish: it hides the step count.
let (pk, vk) = CompressedSNARK::<_, _, _, S<E1>, S<E2>>::setup(&pp)?;
let proof = CompressedSNARK::<_, _, _, S<E1>, S<E2>>::prove(&pp, &pk, &recursive_snark)?;

// The verifier does not need to know the step count.
let zn = proof.verify(&vk, 1, &z0)?;
```

### Differences from upstream

| | Upstream Nova | This branch |
|---|---|---|
| `RecursiveSNARK::verify(pp, num_steps, z0)` | checks `num_steps` | ignores `num_steps` |
| `CompressedSNARK::verify(vk, num_steps, z0)` | checks `num_steps` | only rejects `num_steps == 0` |
| `RecursiveSNARK::num_steps()` | available | removed |
| Proof contents | step count is public | one extra field element per curve (`com`) |

The `num_steps` parameters are kept so existing call sites compile. Pass any
non-zero value.

Public parameters and proofs are **not** compatible with upstream Nova: the
circuit, the parameter digest and the proof layout all differ.

## Cost

The comparison `i < N` and the extra hash add about 3,900 to 4,400 constraints
to each augmented circuit:

| Curve cycle | Primary circuit | Secondary circuit |
|---|---|---|
| Pallas/Vesta | 9,818 → 13,759 | 10,349 → 14,296 |
| BN254/Grumpkin | 9,986 → 14,032 | 10,538 → 14,604 |
| secp/secq | 10,265 → 14,511 | 10,961 → 15,323 |

## Limitations

- **Only `CompressedSNARK` hides the step count.** A `RecursiveSNARK` contains
  the running witnesses, including `i`. Never publish it.
- **Very short chains are distinguishable.** `CompressedSNARK` publishes the
  running instance of the secondary curve. After 1 step that instance is the
  default (all-zero) instance, and after 2 steps it is exactly one fold away
  from the default instance, which anyone can test. From 3 steps on this test
  no longer applies.
- **Proofs of the same chain taken one step apart are linkable.** The published
  running instances of two such proofs satisfy a public relation, which reveals
  that they are consecutive states of one computation. Leave at least two steps
  between published proofs of the same chain.
- **The bound is fixed at `N = 2^254`.** It is hard-coded, not a public
  parameter. On BN254/Grumpkin and secp/secq the value is reduced modulo the
  field order.
- **Not audited**, and there is no written security proof for this
  implementation.

## Performance note: Pasta curves

On x86_64 and aarch64, multi-scalar multiplications over Pallas and Vesta with
32 to 2^17 terms go through [`pasta-msm`](https://crates.io/crates/pasta-msm).
The library still uses the `halo2curves` types everywhere else; points and
scalars are converted at that one boundary. Other sizes and other targets use
Nova's generic implementation.

Measured on an Apple M2 Pro, Pallas/Vesta, against the same code with this path
disabled:

| Step circuit size | `prove_step` | Compress | Verify |
|---|---|---|---|
| 0 constraints | 75.6 → 49.8 ms | 2,222 → 2,089 ms | 50.3 → 43.8 ms |
| 20,000 | 105.7 → 92.0 ms | 5,237 → 5,185 ms | 91.7 → 85.8 ms |
| 100,000 | 262.9 → 220.0 ms | 9,418 → 9,273 ms | 171.0 → 138.2 ms |

## Tests

```text
cargo test --release
```

The IVC tests in `src/nova/mod.rs` run on all three curve cycles.
`test_fast_msm_matches_generic` in `src/provider/pasta.rs` checks the
`pasta-msm` path against the generic implementation.

## License

Same as upstream Nova; see [LICENSE](LICENSE).
