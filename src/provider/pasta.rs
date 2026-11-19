//! This module implements the Nova traits for `pallas::Point`, `pallas::Scalar`, `vesta::Point`, `vesta::Scalar`.
use crate::{
  impl_traits_no_dlog_ext,
  provider::{
    msm::{msm, msm_small, msm_small_with_max_num_bits},
    traits::{DlogGroup, DlogGroupExt},
  },
  traits::{Group, PrimeFieldExt, TranscriptReprTrait},
};
use digest::{ExtendableOutput, Update};
use ff::FromUniformBytes;
use halo2curves::{
  group::{cofactor::CofactorCurveAffine, Curve, Group as AnotherGroup},
  pasta::{Pallas, PallasAffine, Vesta, VestaAffine},
  CurveAffine, CurveExt,
};
use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{Num, ToPrimitive};
use rayon::prelude::*;
use sha3::Shake256;

/// Re-exports that give access to the standard aliases used in the code base, for pallas
pub mod pallas {
  pub use halo2curves::pasta::{Fp as Base, Fq as Scalar, Pallas as Point, PallasAffine as Affine};
}

/// Re-exports that give access to the standard aliases used in the code base, for vesta
pub mod vesta {
  pub use halo2curves::pasta::{Fp as Scalar, Fq as Base, Vesta as Point, VestaAffine as Affine};
}

impl_traits_no_dlog_ext!(
  pallas,
  Pallas,
  PallasAffine,
  "40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001",
  "40000000000000000000000000000000224698fc094cf91b992d30ed00000001"
);

impl_traits_no_dlog_ext!(
  vesta,
  Vesta,
  VestaAffine,
  "40000000000000000000000000000000224698fc094cf91b992d30ed00000001",
  "40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001"
);

/// Multi-scalar multiplication through `pasta-msm`, an assembly-backed Pippenger.
///
/// The rest of the library works with the `halo2curves` Pasta types, so the inputs are
/// converted to their `pasta_curves` counterparts at this boundary and the result is converted
/// back. The conversion goes through canonical encodings, never through memory layout, so it
/// does not depend on how either crate represents a field element.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod fast_msm {
  use ff::PrimeField;
  use halo2curves::{group::Curve, CurveAffine};
  use rayon::prelude::*;

  /// Smallest input for which this path is used; below it the conversion is not worth it.
  pub(super) const MIN_LEN: usize = 1 << 5;
  /// Largest input for which this path is used. It is 2-2.5x faster than the generic
  /// implementation up to 2^12 and about 1.2x faster up to 2^16; beyond that the two are on par.
  pub(super) const MAX_LEN: usize = 1 << 17;

  macro_rules! impl_fast_msm {
    ($fn_name:ident, $h2_affine:ty, $h2_point:ty, $h2_scalar:ty, $pc:ident) => {
      /// Returns `None` when an input has no `pasta-msm` equivalent (a base at infinity), in
      /// which case the caller falls back to the generic implementation.
      pub(super) fn $fn_name(scalars: &[$h2_scalar], bases: &[$h2_affine]) -> Option<$h2_point> {
        use pasta_curves::arithmetic::CurveAffine as PastaCurveAffine;

        assert_eq!(scalars.len(), bases.len());

        let bases_pc = bases
          .par_iter()
          .map(|b| {
            if bool::from(halo2curves::group::prime::PrimeCurveAffine::is_identity(b)) {
              return None;
            }
            let c = Option::<halo2curves::Coordinates<$h2_affine>>::from(b.coordinates())?;
            let x = Option::from(<pasta_curves::$pc::Base as PrimeField>::from_repr(
              c.x().to_repr().as_ref().try_into().ok()?,
            ))?;
            let y = Option::from(<pasta_curves::$pc::Base as PrimeField>::from_repr(
              c.y().to_repr().as_ref().try_into().ok()?,
            ))?;
            Option::from(pasta_curves::$pc::Affine::from_xy(x, y))
          })
          .collect::<Option<Vec<pasta_curves::$pc::Affine>>>()?;

        let scalars_pc = scalars
          .par_iter()
          .map(|s| {
            Option::from(<pasta_curves::$pc::Scalar as PrimeField>::from_repr(
              s.to_repr().as_ref().try_into().ok()?,
            ))
          })
          .collect::<Option<Vec<pasta_curves::$pc::Scalar>>>()?;

        let res = pasta_msm::$pc(&bases_pc, &scalars_pc).to_affine();

        // back to the halo2curves representation
        let coords =
          match Option::<pasta_curves::arithmetic::Coordinates<_>>::from(res.coordinates()) {
            Some(coords) => coords,
            None => return Some(<$h2_point as halo2curves::group::Group>::identity()),
          };
        let mut x_repr = <<$h2_affine as CurveAffine>::Base as PrimeField>::Repr::default();
        x_repr
          .as_mut()
          .copy_from_slice(coords.x().to_repr().as_ref());
        let mut y_repr = <<$h2_affine as CurveAffine>::Base as PrimeField>::Repr::default();
        y_repr
          .as_mut()
          .copy_from_slice(coords.y().to_repr().as_ref());
        let x = Option::from(<<$h2_affine as CurveAffine>::Base as PrimeField>::from_repr(x_repr))?;
        let y = Option::from(<<$h2_affine as CurveAffine>::Base as PrimeField>::from_repr(y_repr))?;
        let p: $h2_affine = Option::from(<$h2_affine as CurveAffine>::from_xy(x, y))?;
        Some(p.into())
      }
    };
  }

  impl_fast_msm!(
    pallas,
    super::pallas::Affine,
    super::pallas::Point,
    super::pallas::Scalar,
    pallas
  );
  impl_fast_msm!(
    vesta,
    super::vesta::Affine,
    super::vesta::Point,
    super::vesta::Scalar,
    vesta
  );
}

macro_rules! impl_dlog_ext {
  ($name:ident) => {
    impl DlogGroupExt for $name::Point {
      fn vartime_multiscalar_mul(
        scalars: &[Self::Scalar],
        bases: &[Self::AffineGroupElement],
      ) -> Self {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        if (fast_msm::MIN_LEN..=fast_msm::MAX_LEN).contains(&scalars.len()) {
          if let Some(res) = fast_msm::$name(scalars, bases) {
            return res;
          }
        }
        msm(scalars, bases)
      }

      fn vartime_multiscalar_mul_small<T: Integer + Into<u64> + Copy + Sync + ToPrimitive>(
        scalars: &[T],
        bases: &[Self::AffineGroupElement],
      ) -> Self {
        msm_small(scalars, bases)
      }

      fn vartime_multiscalar_mul_small_with_max_num_bits<
        T: Integer + Into<u64> + Copy + Sync + ToPrimitive,
      >(
        scalars: &[T],
        bases: &[Self::AffineGroupElement],
        max_num_bits: usize,
      ) -> Self {
        msm_small_with_max_num_bits(scalars, bases, max_num_bits)
      }
    }
  };
}

impl_dlog_ext!(pallas);
impl_dlog_ext!(vesta);

#[cfg(all(test, any(target_arch = "x86_64", target_arch = "aarch64")))]
mod tests {
  use super::*;
  use ff::Field;
  use rand_core::OsRng;

  macro_rules! check_fast_msm {
    ($name:ident) => {
      for n in [1usize, 2, 31, 32, 33, 1000, 5000] {
        let bases = <$name::Point as DlogGroup>::from_label(b"fast_msm_test", n);
        let mut scalars = (0..n)
          .map(|_| $name::Scalar::random(&mut OsRng))
          .collect::<Vec<_>>();
        // edge values
        scalars[0] = $name::Scalar::ZERO;
        scalars[n - 1] = -$name::Scalar::ONE;
        let expected = msm(&scalars, &bases);
        assert_eq!(fast_msm::$name(&scalars, &bases), Some(expected));
        assert_eq!(
          <$name::Point as DlogGroupExt>::vartime_multiscalar_mul(&scalars, &bases),
          expected
        );

        // all-zero scalars give the identity
        let zeros = vec![$name::Scalar::ZERO; n];
        assert_eq!(
          fast_msm::$name(&zeros, &bases),
          Some(<$name::Point as DlogGroup>::zero())
        );

        // a base at infinity falls back to the generic implementation
        let mut bases_inf = bases.clone();
        bases_inf[n / 2] = <$name::Point as DlogGroup>::zero().affine();
        assert_eq!(fast_msm::$name(&scalars, &bases_inf), None);
        assert_eq!(
          <$name::Point as DlogGroupExt>::vartime_multiscalar_mul(&scalars, &bases_inf),
          msm(&scalars, &bases_inf)
        );
      }
    };
  }

  #[test]
  fn test_fast_msm_matches_generic() {
    check_fast_msm!(pallas);
    check_fast_msm!(vesta);
  }
}
