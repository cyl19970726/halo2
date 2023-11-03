use std::{marker::PhantomData, usize};

use ff::{Field, PrimeField};
use halo2_proofs::{
    circuit::{layouter, AssignedCell, Chip, Layouter, Region, SimpleFloorPlanner, Value},
    plonk::{
        Advice, Assigned, Circuit, Column, ConstraintSystem, Constraints, Error, Expression, Fixed,
        Instance, Selector, TableColumn,
    },
    poly::Rotation,
};

#[derive(Debug, Clone)]
pub struct RangeCheckConfig<F: PrimeField,const RANGE: usize> {
    pub(super)  range_table: TableColumn,
    _marker: PhantomData<F>,
}

#[derive(Debug, Clone)]
pub struct RangeCheckChip<F: PrimeField, const RANGE: usize> {
    pub(super)  config: RangeCheckConfig<F,RANGE>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField, const RANGE: usize> Chip<F> for RangeCheckChip<F, RANGE> {
    type Config = RangeCheckConfig<F,RANGE>;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

impl<F: PrimeField, const RANGE: usize> RangeCheckChip<F, RANGE> {

    pub(super)  fn configure(meta: &mut ConstraintSystem<F>) -> RangeCheckConfig<F,RANGE> {
        let range_table = meta.lookup_table_column();
        RangeCheckConfig {
            range_table: range_table,
            _marker: PhantomData,
        }
    }

    pub(super)  fn load_range_table(config: RangeCheckConfig<F,RANGE>,mut layouter:impl Layouter<F>) -> Result<(), Error> {
        layouter.assign_table(
            || "Load Range Table",
            |mut table| {
                let mut offset = 0;
                for value in 0..RANGE {
                    table.assign_cell(
                        || "assign range table value",
                        config.range_table,
                        offset,
                        || Value::known(F::from(value as u64)),
                    )?;
                    offset += 1;
                }
                Ok(())
            },
        )
    }
}
