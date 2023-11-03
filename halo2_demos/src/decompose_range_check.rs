use core::num;
use ff::{PrimeField, PrimeFieldBits};
use halo2_proofs::{circuit::*, plonk::*, poly::Rotation};
use std::marker::PhantomData;

pub mod table;
use table::RangeTableConfig;

/// This gadget range-constrains an element witnessed in the circuit to be N bits.
///
/// Internally, this gadget uses the `range_check` helper, which provides a K-bit
/// lookup table.
///
/// Given an element `value`, we use a running sum to break it into K-bit chunks.
/// Assume for now that N | K, and define C = N / K.
///
///     value = [b_0, b_1, ..., b_{N-1}]   (little-endian)
///           = c_0 + 2^K * c_1  + 2^{2K} * c_2 + ... + 2^{(C-1)K} * c_{C-1}
///
/// Initialise the running sum at
///                                 value = z_0.
///
/// Consequent terms of the running sum are z_{i+1} = (z_i - c_i) * 2^{-K}:
///
///                           z_1 = (z_0 - c_0) * 2^{-K}   ==> c_0 = z_0 - z_1 * 2^K
///                           z_2 = (z_1 - c_1) * 2^{-K}
///                              ...
///                       z_{C-1} = c_{C-1}
///                           z_C = (z_{C-1} - c_{C-1}) * 2^{-K}
///                               = 0
///
///  example:
///     value = z_0 = 7 =[1,1,1] ; K = 1
///     z_1 = (7 - 1) * 2^{-1} = 3
///     z_2 = (3 - 1) * 2^{-1} = 1
///
///     c_0 = z_0 - z_1 * 2^K
/// One configuration for this gadget could look like:
///
///     | running_sum |  q_decompose  |  table_value  |
///     -----------------------------------------------
///     |     z_0     |       1       |       0       |
///     |     z_1     |       1       |       1       |
///     |     ...     |      ...      |      ...      |
///     |   z_{C-1}   |       1       |      ...      |
///     |     z_C     |       0       |      ...      |
///
/// In the case where `N` is not a multiple of `K`, we will have to handle the
/// final `l`-bit partial chunk separately (where `l` < `K`). In other words, we
/// will have to constrain c_{C-1} to `l` bits.
///
///     |      num_bits    | running_sum |  q_decompose  | q_partial_check | table_num_bits|  table_value  |
///     --------------------------------------------------------------------------------------------
///     |         0        |     z_0     |       1       |        0        |       1       |       0       |
///     |         0        |     z_1     |       1       |        0        |       1       |       1       |
///     |        ...       |     ...     |      ...      |       ...       |      ...      |      ...      |
///     |log2_ceil(c_{C-1})|   z_{C-1}   |       1       |        1        |      ...      |      ...      |
///     |         0        |     z_C     |       0       |        0        |      ...      |      ...      |
///
/// To do this, we can lookup the number of bits of c_{C-1} and check that it
/// equals `l`.
///
/// N = 64, K = 10
/// l = 4
///
/// value: u64 = 0
/// value: u64 = 0xFFFFFFFF
///
/// witness:
///     - final_chunk
///     - witness log_final_chunk = log_2_ceil(final_chunk)
///     - range_check that log_final_chunk \in [0, l]
///
/// lookup: (log_final_chunk, final_chunk) against (table.num_bits, table.value)
///
///

#[derive(Debug, Clone)]
pub struct DecomposeConfig<
    F: PrimeFieldBits + PrimeField,
    const LOOKUP_NUM_BITS: usize,
    const LOOKUP_RANGE: usize,
> {
    // you will need an advice column to witness your running sum
    running_sum: Column<Advice>,
    num_bits: Column<Advice>,
    // a selector to costraint the running sum;
    q_decompose: Selector,
    // a selector to constraint the partial chunk;
    q_partial_check: Selector,
    // And of course, the K-bit table
    table: RangeTableConfig<F, LOOKUP_NUM_BITS, LOOKUP_RANGE>,
    _marker: PhantomData<F>,
}

impl<
        F: PrimeField + PrimeFieldBits,
        const LOOKUP_NUM_BITS: usize,
        const LOOKUP_RANGE: usize,
    > DecomposeConfig<F, LOOKUP_NUM_BITS, LOOKUP_RANGE>
{
    pub fn configure(
        meta: &mut ConstraintSystem<F>,
        num_bits: Column<Advice>,
        running_sum: Column<Advice>,
    ) -> Self {
        // Create the needed columns and internal configs.
        let q_decompose = meta.complex_selector();
        let q_partial_check = meta.complex_selector();
        let table = RangeTableConfig::configure(meta);

        meta.enable_equality(running_sum);

        // Range-constrain each K-bit chunk `c_i = z_i - z_{i+1} * 2^K` derived from the running sum.
        meta.lookup(|meta| {
            let q_decompose = meta.query_selector(q_decompose);

            // z_i
            let z_cur = meta.query_advice(running_sum, Rotation::cur());
            // z_{i+1}
            let z_next = meta.query_advice(running_sum, Rotation::next());
            // c_i = z_i - z_{i+1} * 2^k
            let c_i = z_cur - z_next * F::from(1u64 << LOOKUP_NUM_BITS);

            // Lookup default value 0 when q_decompose = 0
            let not_q_decompose = Expression::Constant(F::ZERO) - q_decompose.clone();
            let default_c_i = Expression::Constant(F::ZERO);

            vec![(
                not_q_decompose * default_c_i + q_decompose * c_i,
                table.value,
            )]
        });

        Self {
            running_sum: running_sum,
            num_bits: num_bits,
            q_decompose: q_decompose,
            q_partial_check: q_partial_check,
            table: table,
            _marker: PhantomData,
        }
    }

    fn assign(
        &self,
        mut layouter: impl Layouter<F>,
        value: AssignedCell<Assigned<F>, F>,
        num_bits: usize,
    ) -> Result<(), Error> {
        // `num_bits` must be a multiple of K.
        assert_eq!(num_bits % LOOKUP_NUM_BITS, 0);

        layouter.assign_region(
            || "Decompose value",
            |mut region| {
                let mut offset = 0;

                // 0. Copy in the witnessed `value` at offset = 0
                let mut z = value.copy_advice(
                    || "Copy in value for decomposition",
                    &mut region,
                    self.running_sum,
                    offset,
                )?;

                // Increase offset after copying `value`
                offset += 1;

                // 1. Compute the interstitial running sum values {z_1, ..., z_C}}
                let running_sum: Vec<_> = value
                    .value()
                    .map(|&v| compute_running_sum::<_, LOOKUP_NUM_BITS>(v, num_bits))
                    .transpose_vec(num_bits / LOOKUP_NUM_BITS);

                // 2. Assign the running sum values
                for z_i in running_sum.into_iter() {
                    z = region.assign_advice(
                        || format!("assign z_{:?}", offset),
                        self.running_sum,
                        offset,
                        || z_i,
                    )?;
                    offset += 1;
                }

                // 3. Make sure to enable the relevant selector on each row of the running sum
                //    (but not on the row where z_C is witnessed)
                for offset in 0..(num_bits / LOOKUP_NUM_BITS) {
                    self.q_decompose.enable(&mut region, offset)?;
                }

                // 4. Constrain the final running sum `z_C` to be 0.
                region.constrain_constant(z.cell(), F::ZERO)
            },
        )
    }
}

fn lebs2ip(bits: &[bool]) -> u64 {
    assert!(bits.len() <= 64);
    bits.iter()
        .enumerate()
        .fold(0u64, |acc, (i, b)| acc + if *b { 1 << i } else { 0 })
}

// Function to compute the interstitial running sum values {z_1, ..., z_C}}
fn compute_running_sum<F: PrimeField + PrimeFieldBits, const LOOKUP_NUM_BITS: usize>(
    value: Assigned<F>,
    num_bits: usize,
) -> Vec<Assigned<F>> {
    let mut running_sum = vec![];
    let mut z = value;

    // Get the little-endian bit representation of `value`.
    let value: Vec<_> = value
        .evaluate()
        .to_le_bits()
        .iter()
        .by_vals()
        .take(num_bits)
        .collect();
    for chunk in value.chunks(LOOKUP_NUM_BITS) {
        let chunk = Assigned::from(F::from(lebs2ip(chunk)));
        // z_{i+1} = (z_i - c_i) * 2^{-K}:
        z = (z - chunk) * Assigned::from(F::from(1u64 << LOOKUP_NUM_BITS)).invert();
        running_sum.push(z);
    }

    assert_eq!(running_sum.len(), num_bits / LOOKUP_NUM_BITS);
    running_sum
}


#[cfg(test)]
mod tests {
    use halo2_proofs::{circuit::floor_planner::V1, dev::MockProver, pasta::Fp};
    use rand;

    use super::*;

    struct MyCircuit<F: PrimeField, const NUM_BITS: usize, const RANGE: usize> {
        value: Value<Assigned<F>>,
        num_bits: usize,
    }

    impl<F: PrimeField + PrimeFieldBits, const NUM_BITS: usize, const RANGE: usize> Circuit<F>
        for MyCircuit<F, NUM_BITS, RANGE>
    {
        type Config = DecomposeConfig<F, NUM_BITS, RANGE>;
        type FloorPlanner = V1;

        fn without_witnesses(&self) -> Self {
            Self {
                value: Value::unknown(),
                num_bits: self.num_bits,
            }
        }

        fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
            // Fixed column for constants
            let constants = meta.fixed_column();
            meta.enable_constant(constants);

            let value = meta.advice_column();
            let num_bits = meta.advice_column();
            DecomposeConfig::configure(meta,num_bits ,value)
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            config.table.load(&mut layouter)?;

            // Witness the value somewhere
            let value = layouter.assign_region(
                || "Witness value",
                |mut region| {
                    region.assign_advice(|| "Witness value", config.running_sum, 0, || self.value)
                },
            )?;

            config.assign(
                layouter.namespace(|| "Decompose value"),
                value,
                self.num_bits,
            )?;

            Ok(())
        }
    }

    #[test]
    fn test_decompose_1() {
        let k = 9;
        const NUM_BITS: usize = 8;
        const RANGE: usize = 256; // 8-bit value

        // Random u64 value
        let value: u64 = rand::random();
        let value = Value::known(Assigned::from(Fp::from(value)));

        let circuit = MyCircuit::<Fp, NUM_BITS, RANGE> {
            value,
            num_bits: 64,
        };

        let prover = MockProver::run(k, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[cfg(feature = "dev-graph")]
    #[test]
    fn print_decompose_1() {
        use plotters::prelude::*;

        let root = BitMapBackend::new("decompose-layout.png", (1024, 3096)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let root = root
            .titled("Decompose Range Check Layout", ("sans-serif", 60))
            .unwrap();

        let circuit = MyCircuit::<Fp, 8, 256> {
            value: Value::unknown(),
            num_bits: 64,
        };
        halo2_proofs::dev::CircuitLayout::default()
            .render(9, &circuit, &root)
            .unwrap();
    }
}

