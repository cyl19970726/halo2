pub mod spread_table;
use std::{marker::PhantomData, ops::Div};

use ff::{PrimeField, PrimeFieldBits};
use spread_table::*;

pub mod util;
use util::*;

pub mod assigned_bits;
use assigned_bits::*;

use halo2_proofs::{
    circuit::{AssignedCell, Chip, Layouter, Region, SimpleFloorPlanner, Value},
    plonk::{
        Advice, Assigned, Circuit, Column, ConstraintSystem, Constraints, Error, Expression, Fixed,
        Instance, Selector, TableColumn,
    },
    poly::Rotation,
};

#[derive(Debug, Clone)]
pub struct VarConfig {
    lhs: Column<Advice>,
    xor_selector: Selector,
}

#[derive(Debug, Clone)]
pub struct XORConfig {
    var_config: VarConfig,
    spread_config: SpreadTableConfig,
}

#[derive(Debug, Clone)]
pub struct XORChip<F: PrimeField + PrimeFieldBits> {
    config: XORConfig,
    _marker: PhantomData<F>,
}

impl<F: PrimeField + PrimeFieldBits> XORChip<F> {
    pub fn construct(config: XORConfig) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    pub fn configure(meta: &mut ConstraintSystem<F>, lhs: Column<Advice>) -> XORConfig {
        let tag = meta.advice_column();
        let dense = meta.advice_column();
        let spread = meta.advice_column();
        let spread_table_config =
            SpreadTableChip::configure(meta, tag.clone(), dense.clone(), spread.clone());

        meta.enable_equality(lhs);
        let xor_selector = meta.complex_selector(); // it need to be complex selector to enforce the lookup argument constraints

        meta.enable_equality(tag);
        meta.enable_equality(dense);
        meta.enable_equality(spread);

        meta.create_gate("xor", |v_cells| {
            //
            // xor_selector|  lhs   | tag | dense | spread|
            //     1          a               a       a'
            //                b               b       b'               z'=a'+b'
            //                z'            z_even  z'_even            z'=z'_even+2*z'_odd
            //           z'_odd_mul2                    
            //
            let xor_selector = v_cells.query_selector(xor_selector);

            // let a = v_cells.query_advice(lhs, Rotation::cur());
            // let a_dense = v_cells.query_advice(dense, Rotation::cur());
            // let b = v_cells.query_advice(lhs, Rotation::next());
            // let b_dense = v_cells.query_advice(dense, Rotation::next());

            let a_spread = v_cells.query_advice(spread, Rotation::cur());
            let b_spread = v_cells.query_advice(spread, Rotation::next());
            let z_spread = v_cells.query_advice(lhs, Rotation(2));

            let z_even_spread = v_cells.query_advice(spread, Rotation(2));
            let z_odd_spread_mul2 = v_cells.query_advice(lhs, Rotation(3));

            // let z_even = v_cells.query_advice(dense, Rotation(2));

            // vec![
            //     xor_selector
            //         * (a_spread + b_spread - z_even_spread - z_odd_spread_mul2) * (z_even - result),
            // ]
            vec![
                xor_selector.clone() * (a_spread + b_spread - z_spread.clone()),
                xor_selector * (z_spread - z_even_spread - z_odd_spread_mul2),
            ]
        });

        XORConfig {
            var_config: VarConfig { lhs, xor_selector },
            spread_config: spread_table_config,
        }
    }

    pub fn assign(
        &self,
        mut layouter: impl Layouter<F>,
        a: u16,
        b: u16,
        result: u16,
    ) -> Result<AssignedCell<F, F>, Error> {
        // assert_eq!(a^b, result);
        layouter.assign_region(
            || "Assign_region::XOR",
            |mut region| {
                self.config.var_config.xor_selector.enable(&mut region, 0)?;

                // assign lhs cloumn
                let a_cell = region.assign_advice(
                    || "a",
                    self.config.var_config.lhs,
                    0,
                    || Value::known(F::from(a as u64)),
                )?;

                let b_cell = region.assign_advice(
                    || "b",
                    self.config.var_config.lhs,
                    1,
                    || Value::known(F::from(b as u64)),
                )?;

                // assign the |tag|dense|spread| of a
                let a_tag = F::from(u64::from(get_tag(a)));
                println!("a_tag:{:b}", u64::from(get_tag(a)));
                println!("a_dense:{:b}", u64::from(u64::from(a)));
                let a_spread = F::from(u64::from(interleave_u16_with_zeros(a)));
                println!(
                    "a_spread:{:b}",
                    u64::from(u64::from(interleave_u16_with_zeros(a)))
                );

                region.assign_advice(
                    || "assign a_spread",
                    self.config.spread_config.input.tag,
                    0,
                    || Value::known(a_tag),
                )?;
                a_cell.copy_advice(||"assign a_dense", &mut region , self.config.spread_config.input.dense, 0)?;

                region.assign_advice(
                    || "assign a_spread",
                    self.config.spread_config.input.spread,
                    0,
                    || Value::known(a_spread),
                )?;

                // assign the |tag|dense|spread| of b
                let b_tag = F::from(u64::from(get_tag(b)));
                // let b_dense = F::from(u64::from(b));
                let b_spread = F::from(u64::from(interleave_u16_with_zeros(b)));
                region.assign_advice(
                    || "assign b_tag",
                    self.config.spread_config.input.tag,
                    1,
                    || Value::known(b_tag),
                )?;
                b_cell.copy_advice(||"assign b_dense", &mut region, self.config.spread_config.input.dense, 1)?;
                region.assign_advice(
                    || "assign b_spread",
                    self.config.spread_config.input.spread,
                    1,
                    || Value::known(b_spread),
                )?;

                // assign the z_even
                let z_32 = a_spread + b_spread;

                let z_even_tag = F::from(u64::from(get_tag(result)));
                let z_even = F::from(u64::from(result));
                let z_even_spread = F::from(u64::from(interleave_u16_with_zeros(result)));
                region.assign_advice(
                    || "assign z_even_tag",
                    self.config.spread_config.input.tag,
                    2,
                    || Value::known(z_even_tag),
                )?;
                region.assign_advice(
                    || "assign z_even_dense",
                    self.config.spread_config.input.dense,
                    2,
                    || Value::known(z_even),
                )?;
                region.assign_advice(
                    || "assign z_even_spread",
                    self.config.spread_config.input.spread,
                    2,
                    || Value::known(z_even_spread),
                )?;

                // assign z_32_odd_double
                let z_32_odd_double = z_32 - z_even_spread;
                region.assign_advice(
                    || "asign z_32_odd_double",
                    self.config.var_config.lhs,
                    3,
                    || Value::known(z_32_odd_double),
                )?;

                 let output = region.assign_advice(
                    || "result",
                    self.config.var_config.lhs,
                    2,
                    || Value::known(a_spread+b_spread),
                )?;
                Ok(output)
            },
        )
    }
}

#[derive(Clone, Debug)]
struct XORCircuits<F: PrimeField + PrimeFieldBits> {
    a: u16,
    b: u16,
    result: u16,
    _marker: PhantomData<F>,
}

impl<F: PrimeField + PrimeFieldBits> Circuit<F> for XORCircuits<F> {
    type Config = XORConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        let lhs = meta.advice_column();
        XORChip::configure(meta, lhs)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let a = self.a;
        let b = self.b;
        let result = self.result;
        SpreadTableChip::load(config.spread_config.clone(), &mut layouter)?;
        let output = XORChip::construct(config).assign(
            layouter.namespace(|| "assigon xor and spread table"),
            a,
            b,
            result,
        )?;

        Ok(())
    }
}

mod tests {

    use super::*;
    use halo2_proofs::{dev::MockProver, pasta::Fp};
    #[test]
    fn test_xor() {
        let a = 0b0011; // 100
        let b = 0b0110; // 101
        // let result = a ^ b; // 101
                            let result = 1111;   // 101
        println!("result:{}", result);

        let circuit: XORCircuits<Fp> = XORCircuits {
            a,
            b,
            result,
            _marker: PhantomData,
        };
        let prover = MockProver::<Fp>::run(17, &circuit, [].to_vec()).unwrap();
        prover.assert_satisfied();
    }

    #[cfg(feature = "dev-graph")]
    #[test]
    fn plot_xor() {
        /// Instance columns have a white background.
        /// Advice columns have a red background.
        /// Fixed columns have a blue background.
        use plotters::prelude::*;

        let root = BitMapBackend::new("xor.png", (2048, 2048)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let root = root.titled("XOR_Layout", ("sans-serif", 60)).unwrap();

        let const_value = Fp::from(3);
        let free_value = Fp::from(7);
        let a = 0b0011; // 100
        let b = 0b0110; // 101
        let result = a ^ b; // 101
                            // let result = 0;   // 100

        let circuit: XORCircuits<Fp> = XORCircuits {
            a,
            b,
            result,
            _marker: PhantomData,
        };
        halo2_proofs::dev::CircuitLayout::default()
            .mark_equality_cells(true)
            .show_equality_constraints(false)
            // You can optionally render only a section of the circuit.
            .view_width(0..11)
            .view_height(0..8)
            // You can hide labels, which can be useful with smaller areas.
            .show_labels(true)
            // Render the circuit onto your area!
            // The first argument is the size parameter for the circuit.
            .render(17, &circuit, &root)
            .unwrap();
    }
}

pub fn interleave_u16_with_zeros(word: u16) -> u32 {
    let mut word: u32 = word.into();
    word = (word ^ (word << 8)) & 0x00ff00ff;
    word = (word ^ (word << 4)) & 0x0f0f0f0f;
    word = (word ^ (word << 2)) & 0x33333333;
    word = (word ^ (word << 1)) & 0x55555555;
    word
}
