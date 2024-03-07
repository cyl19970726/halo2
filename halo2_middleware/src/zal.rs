//! This module provides "ZK Acceleration Layer" traits
//! to abstract away the execution engine for performance-critical primitives.
//!
//! Terminology
//! -----------
//!
//! We use the name Backend+Engine for concrete implementations of ZalEngine.
//! For example H2cEngine for pure Halo2curves implementation.
//!
//! Alternative names considered were Executor or Driver however
//! - executor is already used in Rust (and the name is long)
//! - driver will be confusing as we work quite low-level with GPUs and FPGAs.
//!
//! Unfortunately the "Engine" name is used in bn256 for pairings.
//! Fortunately a ZalEngine is only used in the prover (at least for now)
//! while "pairing engine" is only used in the verifier
//!
//! Initialization design space
//! ---------------------------
//!
//! It is recommended that ZAL backends provide:
//! - an initialization function:
//!   - either "fn new() -> ZalEngine" for simple libraries
//!   - or a builder pattern for complex initializations
//! - a shutdown function or document when it is not needed (when it's a global threadpool like Rayon for example).
//!
//! Backends might want to add as an option:
//! - The number of threads (CPU)
//! - The device(s) to run on (multi-sockets machines, multi-GPUs machines, ...)
//! - The curve (JIT-compiled backend)
//!
//! Descriptors
//! ---------------------------
//!
//! Descriptors enable providers to configure opaque details on data
//! when doing repeated computations with the same input(s).
//! For example:
//! - Pointer(s) caching to limit data movement between CPU and GPU, FPGAs
//! - Length of data
//! - data in layout:
//!    - canonical or Montgomery fields, unsaturated representation, endianness
//!    - jacobian or projective coordinates or maybe even Twisted Edwards for faster elliptic curve additions,
//!    - FFT: canonical or bit-reversed permuted
//! - data out layout
//! - Device(s) ID
//!
//! They are required to be Plain Old Data (Copy trait), so no custom `Drop` is required.
//! If a specific resource is needed, it can be stored in the engine in a hashmap for example
//! and an integer ID or a pointer can be opaquely given as a descriptor.

// The ZK Accel Layer API
// ---------------------------------------------------
pub mod traits {
    use halo2curves::CurveAffine;

    pub trait MsmAccel<C: CurveAffine> {
        fn msm(&self, coeffs: &[C::Scalar], base: &[C]) -> C::Curve;

        // Caching API
        // -------------------------------------------------
        // From here we propose an extended API
        // that allows reusing coeffs and/or the base points
        //
        // This is inspired by CuDNN API (Nvidia GPU)
        // and oneDNN API (CPU, OpenCL) https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnn-ops-infer-so-opaque
        // usage of descriptors
        //
        // https://github.com/oneapi-src/oneDNN/blob/master/doc/programming_model/basic_concepts.md
        //
        // Descriptors are opaque pointers that hold the input in a format suitable for the accelerator engine.
        // They may be:
        // - Input moved on accelerator device (only once for repeated calls)
        // - Endianess conversion
        // - Converting from Montgomery to Canonical form
        // - Input changed from Projective to Jacobian coordinates or even to a Twisted Edwards curve.
        // - other form of expensive preprocessing
        type CoeffsDescriptor<'c>: Copy;
        type BaseDescriptor<'b>: Copy;

        fn get_coeffs_descriptor<'c>(&self, coeffs: &'c [C::Scalar]) -> Self::CoeffsDescriptor<'c>;
        fn get_base_descriptor<'b>(&self, base: &'b [C]) -> Self::BaseDescriptor<'b>;

        fn msm_with_cached_scalars(
            &self,
            coeffs: &Self::CoeffsDescriptor<'_>,
            base: &[C],
        ) -> C::Curve;

        fn msm_with_cached_base(
            &self,
            coeffs: &[C::Scalar],
            base: &Self::BaseDescriptor<'_>,
        ) -> C::Curve;

        fn msm_with_cached_inputs(
            &self,
            coeffs: &Self::CoeffsDescriptor<'_>,
            base: &Self::BaseDescriptor<'_>,
        ) -> C::Curve;
        // Execute MSM according to descriptors
        // Unsure of naming, msm_with_cached_inputs, msm_apply, msm_cached, msm_with_descriptors, ...
    }
}

// ZAL using Halo2curves as a backend
// ---------------------------------------------------

pub mod impls {
    use std::marker::PhantomData;

    use crate::zal::traits::MsmAccel;
    use halo2curves::msm::best_multiexp;
    use halo2curves::CurveAffine;

    // Halo2curve Backend
    // ---------------------------------------------------
    #[derive(Default)]
    pub struct H2cEngine;

    #[derive(Clone, Copy)]
    pub struct H2cMsmCoeffsDesc<'c, C: CurveAffine> {
        raw: &'c [C::Scalar],
    }

    #[derive(Clone, Copy)]
    pub struct H2cMsmBaseDesc<'b, C: CurveAffine> {
        raw: &'b [C],
    }

    impl H2cEngine {
        pub fn new() -> Self {
            Self {}
        }
    }

    impl<C: CurveAffine> MsmAccel<C> for H2cEngine {
        fn msm(&self, coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
            best_multiexp(coeffs, bases)
        }

        // Caching API
        // -------------------------------------------------

        type CoeffsDescriptor<'c> = H2cMsmCoeffsDesc<'c, C>;
        type BaseDescriptor<'b> = H2cMsmBaseDesc<'b, C>;

        fn get_coeffs_descriptor<'c>(&self, coeffs: &'c [C::Scalar]) -> Self::CoeffsDescriptor<'c> {
            // Do expensive device/library specific preprocessing here
            Self::CoeffsDescriptor { raw: coeffs }
        }
        fn get_base_descriptor<'b>(&self, base: &'b [C]) -> Self::BaseDescriptor<'b> {
            Self::BaseDescriptor { raw: base }
        }

        fn msm_with_cached_scalars(
            &self,
            coeffs: &Self::CoeffsDescriptor<'_>,
            base: &[C],
        ) -> C::Curve {
            best_multiexp(coeffs.raw, base)
        }

        fn msm_with_cached_base(
            &self,
            coeffs: &[C::Scalar],
            base: &Self::BaseDescriptor<'_>,
        ) -> C::Curve {
            best_multiexp(coeffs, base.raw)
        }

        fn msm_with_cached_inputs(
            &self,
            coeffs: &Self::CoeffsDescriptor<'_>,
            base: &Self::BaseDescriptor<'_>,
        ) -> C::Curve {
            best_multiexp(coeffs.raw, base.raw)
        }
    }

    // Backend-agnostic engine objects
    // ---------------------------------------------------
    #[derive(Debug)]
    pub struct PlonkEngine<C: CurveAffine, MsmEngine: MsmAccel<C>> {
        pub msm_backend: MsmEngine,
        _marker: PhantomData<C>, // compiler complains about unused C otherwise
    }

    #[derive(Default)]
    pub struct PlonkEngineConfig<C, M> {
        curve: PhantomData<C>,
        msm_backend: Option<M>,
    }

    #[derive(Default)]
    pub struct NoCurve;

    #[derive(Default)]
    pub struct HasCurve<C: CurveAffine>(PhantomData<C>);

    #[derive(Default)]
    pub struct NoMsmEngine;

    pub struct HasMsmEngine<C: CurveAffine, M: MsmAccel<C>>(M, PhantomData<C>);

    impl PlonkEngineConfig<NoCurve, NoMsmEngine> {
        pub fn new() -> PlonkEngineConfig<NoCurve, NoMsmEngine> {
            Default::default()
        }

        pub fn set_curve<C: CurveAffine>(self) -> PlonkEngineConfig<HasCurve<C>, NoMsmEngine> {
            Default::default()
        }

        pub fn build_default<C: CurveAffine>() -> PlonkEngine<C, H2cEngine> {
            PlonkEngine {
                msm_backend: H2cEngine::new(),
                _marker: Default::default(),
            }
        }
    }

    impl<C: CurveAffine, M> PlonkEngineConfig<HasCurve<C>, M> {
        pub fn set_msm<MsmEngine: MsmAccel<C>>(
            self,
            engine: MsmEngine,
        ) -> PlonkEngineConfig<HasCurve<C>, HasMsmEngine<C, MsmEngine>> {
            // Copy all other parameters
            let Self { curve, .. } = self;
            // Return with modified MSM engine
            PlonkEngineConfig {
                curve,
                msm_backend: Some(HasMsmEngine(engine, Default::default())),
            }
        }
    }

    impl<C: CurveAffine, M: MsmAccel<C>> PlonkEngineConfig<HasCurve<C>, HasMsmEngine<C, M>> {
        pub fn build(self) -> PlonkEngine<C, M> {
            PlonkEngine {
                msm_backend: self.msm_backend.unwrap().0,
                _marker: Default::default(),
            }
        }
    }
}

#[cfg(feature = "icicle_gpu")]
pub mod impl_icicle{
    // ZAL using Icicle as a backend
    // ---------------------------------------------------

    use ff::PrimeField;
    use icicle::{
        curves::bn254::{Point_BN254, ScalarField_BN254},
        test_bn254::{commit_batch_bn254, commit_bn254},
    };
    use std::sync::{Arc, Once};

    pub use icicle::curves::bn254::PointAffineNoInfinity_BN254;
    use rustacuda::memory::CopyDestination;
    use rustacuda::prelude::*;

    pub use halo2curves::CurveAffine;
    use log::info;
    use std::{env, mem};

    use std::marker::PhantomData;
    use crate::zal::traits::MsmAccel;

    static mut GPU_CONTEXT: Option<Context> = None;
    static mut GPU_G: Option<DeviceBuffer<PointAffineNoInfinity_BN254>> = None;
    static mut GPU_G_LAGRANGE: Option<DeviceBuffer<PointAffineNoInfinity_BN254>> = None;
    static GPU_INIT: Once = Once::new();

     // Halo2curve Backend
    // ---------------------------------------------------
    #[derive(Default)]
    pub struct GpuEngine;

    #[derive(Clone, Copy)]
    pub struct GpuMsmCoeffsDesc<'c> {
        raw:&'c Vec<ScalarField_BN254>,
    }

    #[derive(Clone, Copy)]
    pub struct GpuMsmBaseDesc<'b> {
        raw:&'b DeviceBuffer<PointAffineNoInfinity_BN254>,
    }

    impl GpuEngine {
        pub fn new() -> Self {
            unsafe {
                GPU_INIT.call_once(|| {
                    GPU_CONTEXT = Some(rustacuda::quick_init().unwrap());
                    // GPU_G = Some(copy_points_to_device(g));
                    // GPU_G_LAGRANGE = Some(copy_points_to_device(g_lagrange));
                    info!("GPU initialized");
                });
            }
            Self {}
        }
    }

    impl<C: CurveAffine> MsmAccel<C> for GpuEngine {
        fn msm(&self, coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {

            let mut gpu_coeffs: DeviceBuffer<ScalarField_BN254> = copy_scalars_to_device(coeffs);
            let mut gpu_bases: DeviceBuffer<PointAffineNoInfinity_BN254> = copy_points_to_device(bases);
            let d_commit_result = commit_bn254(gpu_bases, gpu_coeffs, 10);
            let mut h_commit_result = Point_BN254::zero();
            d_commit_result.copy_to(&mut h_commit_result).unwrap();
            c_from_icicle_point::<C>(h_commit_result)
        }

        // Caching API
        // -------------------------------------------------

        type CoeffsDescriptor<'c> = GpuMsmCoeffsDesc;
        type BaseDescriptor<'b> = GpuMsmBaseDesc;

        fn get_coeffs_descriptor<'c>(&self, coeffs: &'c [C::Scalar]) -> Self::CoeffsDescriptor<'c> {
            // Do expensive device/library specific preprocessing here
            let scalars = icicle_scalars_from_c::<C>(coeffs);
            Self::CoeffsDescriptor { raw: DeviceBuffer::from_slice(scalars.as_slice()).unwrap() }
        }

        fn get_base_descriptor<'b>(&self, base: &'b [C]) -> Self::BaseDescriptor<'b> {
            let points = icicle_points_from_c(bases);
            Self::BaseDescriptor { raw: DeviceBuffer::from_slice(points.as_slice()).unwrap() }
        }

        fn msm_with_cached_scalars(
            &self,
            coeffs: &Self::CoeffsDescriptor<'_>,
            base: &[C],
        ) -> C::Curve {
            let mut gpu_bases: DeviceBuffer<PointAffineNoInfinity_BN254> = copy_points_to_device(bases);
            let d_commit_result = commit_bn254(gpu_bases, coeffs, 10);
            let mut h_commit_result = Point_BN254::zero();
            d_commit_result.copy_to(&mut h_commit_result).unwrap();
            c_from_icicle_point::<C>(h_commit_result)
        }

        fn msm_with_cached_base(
            &self,
            coeffs: &[C::Scalar],
            base: &Self::BaseDescriptor<'_>,
        ) -> C::Curve {
            let mut gpu_coeffs: DeviceBuffer<ScalarField_BN254> = copy_scalars_to_device(coeffs);
            let d_commit_result = commit_bn254(gpu_bases, coeffs, 10);
            let mut h_commit_result = Point_BN254::zero();
            d_commit_result.copy_to(&mut h_commit_result).unwrap();
            c_from_icicle_point::<C>(h_commit_result)
        }

        fn msm_with_cached_inputs(
            &self,
            coeffs: &Self::CoeffsDescriptor<'_>,
            base: &Self::BaseDescriptor<'_>,
        ) -> C::Curve {
            let d_commit_result = commit_bn254(base, coeffs, 10);
            let mut h_commit_result = Point_BN254::zero();
            d_commit_result.copy_to(&mut h_commit_result).unwrap();
            c_from_icicle_point::<C>(h_commit_result)
        }
    }   

    // Gpu utils function

    pub fn is_small_circuit(size: usize) -> bool {
        size <= (1
            << u8::from_str_radix(
                &env::var("ICICLE_SMALL_CIRCUIT").unwrap_or("8".to_string()),
                10,
            )
            .unwrap())
    }

    fn u32_from_u8(u8_arr: &[u8; 32]) -> [u32; 8] {
        let mut t = [0u32; 8];
        for i in 0..8 {
            t[i] = u32::from_le_bytes([
                u8_arr[4 * i],
                u8_arr[4 * i + 1],
                u8_arr[4 * i + 2],
                u8_arr[4 * i + 3],
            ]);
        }
        return t;
    }

    fn repr_from_u32<C: CurveAffine>(u32_arr: &[u32; 8]) -> <C as CurveAffine>::Base {
        let t: &[<<C as CurveAffine>::Base as PrimeField>::Repr] =
            unsafe { mem::transmute(&u32_arr[..]) };
        return PrimeField::from_repr(t[0]).unwrap();
    }

    fn is_infinity_point(point: Point_BN254) -> bool {
        let inf_point = Point_BN254::infinity();
        point.z.s.eq(&inf_point.z.s)
    }

    fn icicle_scalars_from_c<C: CurveAffine>(coeffs: &[C::Scalar]) -> Vec<ScalarField_BN254> {
        let _coeffs = [Arc::new(
            coeffs.iter().map(|x| x.to_repr()).collect::<Vec<_>>(),
        )];

        let _coeffs: &Arc<Vec<[u32; 8]>> = unsafe { mem::transmute(&_coeffs) };
        _coeffs
            .iter()
            .map(|x| ScalarField_BN254::from_limbs(x))
            .collect::<Vec<_>>()
    }

    pub fn copy_scalars_to_device<C: CurveAffine>(
        coeffs: &[C::Scalar],
    ) -> DeviceBuffer<ScalarField_BN254> {
        let scalars = icicle_scalars_from_c::<C>(coeffs);

        DeviceBuffer::from_slice(scalars.as_slice()).unwrap()
    }

    fn icicle_points_from_c<C: CurveAffine>(bases: &[C]) -> Vec<PointAffineNoInfinity_BN254> {
        let _bases = [Arc::new(
            bases
                .iter()
                .map(|p| {
                    let coordinates = p.coordinates().unwrap();
                    [coordinates.x().to_repr(), coordinates.y().to_repr()]
                })
                .collect::<Vec<_>>(),
        )];

        let _bases: &Arc<Vec<[[u8; 32]; 2]>> = unsafe { mem::transmute(&_bases) };
        _bases
            .iter()
            .map(|x| {
                let tx = u32_from_u8(&x[0]);
                let ty = u32_from_u8(&x[1]);
                PointAffineNoInfinity_BN254::from_limbs(&tx, &ty)
            })
            .collect::<Vec<_>>()
    }

    pub fn copy_points_to_device<C: CurveAffine>(
        bases: &[C],
    ) -> DeviceBuffer<PointAffineNoInfinity_BN254> {
        let points = icicle_points_from_c(bases);

        DeviceBuffer::from_slice(points.as_slice()).unwrap()
    }

    fn c_from_icicle_point<C: CurveAffine>(commit_res: Point_BN254) -> C::Curve {
        let (x, y) = if is_infinity_point(commit_res) {
            (
                repr_from_u32::<C>(&[0u32; 8]),
                repr_from_u32::<C>(&[0u32; 8]),
            )
        } else {
            let affine_res_from_cuda = commit_res.to_affine();
            (
                repr_from_u32::<C>(&affine_res_from_cuda.x.s),
                repr_from_u32::<C>(&affine_res_from_cuda.y.s),
            )
        };

        let affine = C::from_xy(x, y).unwrap();
        return affine.to_curve();
    }

    pub fn multiexp_on_device<C: CurveAffine>(
        mut coeffs: DeviceBuffer<ScalarField_BN254>,
        is_lagrange: bool,
    ) -> C::Curve {
        let base_ptr: &mut DeviceBuffer<PointAffineNoInfinity_BN254>;
        unsafe {
            if is_lagrange {
                base_ptr = GPU_G_LAGRANGE.as_mut().unwrap();
            } else {
                base_ptr = GPU_G.as_mut().unwrap();
            };
        }

        let d_commit_result = commit_bn254(base_ptr, &mut coeffs, 10);

        let mut h_commit_result = Point_BN254::zero();
        d_commit_result.copy_to(&mut h_commit_result).unwrap();

        c_from_icicle_point::<C>(h_commit_result)
    }

    pub fn batch_multiexp_on_device<C: CurveAffine>(
        mut coeffs: DeviceBuffer<ScalarField_BN254>,
        mut bases: DeviceBuffer<PointAffineNoInfinity_BN254>,
        batch_size: usize,
    ) -> Vec<C::Curve> {
        let d_commit_result = commit_batch_bn254(&mut bases, &mut coeffs, batch_size);
        let mut h_commit_result: Vec<Point_BN254> =
            (0..batch_size).map(|_| Point_BN254::zero()).collect();
        d_commit_result.copy_to(&mut h_commit_result[..]).unwrap();

        h_commit_result
            .iter()
            .map(|commit_result| c_from_icicle_point::<C>(*commit_result))
            .collect()
    }

}

// Testing
// ---------------------------------------------------

#[cfg(test)]
mod test {
    use crate::zal::impls::{H2cEngine, PlonkEngineConfig};
    use crate::zal::traits::MsmAccel;
    use halo2curves::bn256::G1Affine;
    use halo2curves::msm::best_multiexp;
    use halo2curves::CurveAffine;

    use ark_std::{end_timer, start_timer};
    use ff::Field;
    use group::{Curve, Group};
    use rand_core::OsRng;

    fn run_msm_zal<C: CurveAffine>(min_k: usize, max_k: usize) {
        let points = (0..1 << max_k)
            .map(|_| C::Curve::random(OsRng))
            .collect::<Vec<_>>();
        let mut affine_points = vec![C::identity(); 1 << max_k];
        C::Curve::batch_normalize(&points[..], &mut affine_points[..]);
        let points = affine_points;

        let scalars = (0..1 << max_k)
            .map(|_| C::Scalar::random(OsRng))
            .collect::<Vec<_>>();

        for k in min_k..=max_k {
            let points = &points[..1 << k];
            let scalars = &scalars[..1 << k];

            let t0 = start_timer!(|| format!("freestanding msm k={}", k));
            let e0 = best_multiexp(scalars, points);
            end_timer!(t0);

            let engine = PlonkEngineConfig::new()
                .set_curve::<G1Affine>()
                .set_msm(H2cEngine::new())
                .build();
            let t1 = start_timer!(|| format!("H2cEngine msm k={}", k));
            let e1 = engine.msm_backend.msm(scalars, points);
            end_timer!(t1);

            assert_eq!(e0, e1);

            // Caching API
            // -----------
            let t2 = start_timer!(|| format!("H2cEngine msm cached base k={}", k));
            let base_descriptor = engine.msm_backend.get_base_descriptor(points);
            let e2 = engine
                .msm_backend
                .msm_with_cached_base(scalars, &base_descriptor);
            end_timer!(t2);

            assert_eq!(e0, e2)
        }
    }

    #[test]
    fn test_msm_zal() {
        run_msm_zal::<G1Affine>(3, 14);
    }
}
