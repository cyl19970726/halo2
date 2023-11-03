use super::{RangeCheckChip, RangeCheckConfig};
use ff::{Field, PrimeField};
use halo2_proofs::{
    circuit::{AssignedCell, Chip, Layouter, Region, SimpleFloorPlanner, Value},
    plonk::{
        Advice, Assigned, Circuit, Column, ConstraintSystem, Constraints, Error, Expression, Fixed,
        Instance, Selector, TableColumn,
    },
    poly::Rotation,
};
use std::{borrow::BorrowMut, marker::PhantomData, ops::Add, usize, vec};
/// A = registers[0] , B = registers[1]
/// A′=A+setA⋅(inA⋅A+inB⋅B+inFREE⋅FREE+CONST−A)
/// B′=B+setB⋅(inA⋅A+inB⋅B+inFREE⋅FREE+CONST−B)
#[derive(Debug, Clone)]
pub(super) struct RegistersConfig<F: PrimeField, const RIGSTER_NUM: usize, const RANGE: usize> {
    // The Global Selector to Enable VM Constraints
    pub(super) enable_vm: Selector,
    // The selector for cur_registers
    pub(super) in_registers: [Selector; RIGSTER_NUM],
    pub(super) cur_registers: [Column<Advice>; RIGSTER_NUM],
    // The selector for next_registers
    pub(super) set_registers: [Selector; RIGSTER_NUM],
    pub(super) next_registers: [Column<Advice>; RIGSTER_NUM],
    // The Selector for free_val
    pub(super) in_free: Selector,
    pub(super) free_val: Column<Fixed>,
    // constant variables
    pub(super) const_val: Column<Fixed>,
    // public input and output
    pub(super) instance: Column<Instance>,
    // lookup range check table
    pub(super) enable_lookup_range: Selector,
    pub(super) range_check_table: RangeCheckConfig<F,RANGE>,
    _marker: PhantomData<F>,
}

#[derive(Debug, Clone)]
struct RegisterChip<F: PrimeField, const RIGSTER_NUM: usize, const RANGE: usize> {
    config: RegistersConfig<F, RIGSTER_NUM,RANGE>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField, const RIGSTER_NUM: usize,const RANGE: usize> Chip<F> for RegisterChip<F, RIGSTER_NUM,RANGE> {
    type Config = RegistersConfig<F, RIGSTER_NUM,RANGE>;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

impl<F: PrimeField, const RIGSTER_NUM: usize,const RANGE: usize> RegisterChip<F, RIGSTER_NUM,RANGE> {
    pub fn construct(config: RegistersConfig<F, RIGSTER_NUM,RANGE>) -> Self {
        Self {
            config: config,
            _marker: PhantomData,
        }
    }

    pub(super) fn configure(meta: &mut ConstraintSystem<F>) -> RegistersConfig<F, RIGSTER_NUM,RANGE> {
        let enable_vm = meta.selector();

        let in_registers: [Selector; RIGSTER_NUM] =
            std::array::from_fn(|i| meta.complex_selector());
        assert!(in_registers.len() == 2);
        let cur_registers: [Column<Advice>; RIGSTER_NUM] = std::array::from_fn(|i| {
            let c = meta.advice_column();
            meta.enable_equality(c);
            c
        });
        let set_registers: [Selector; RIGSTER_NUM] =
            std::array::from_fn(|i| meta.complex_selector());
        let next_registers: [Column<Advice>; RIGSTER_NUM] = std::array::from_fn(|i| {
            let c = meta.advice_column();
            meta.enable_equality(c);
            c
        });

        let free_val = meta.fixed_column();
        let in_free = meta.complex_selector();
        let const_val = meta.fixed_column();
        let instance = meta.instance_column();

        meta.enable_equality(free_val);
        meta.enable_equality(instance);
        meta.enable_equality(const_val);

        meta.create_gate("set specific register", |v_cells| {
            let enable_vm_selector = v_cells.query_selector(enable_vm);
            let in_free_cur = v_cells.query_selector(in_free);
            let free_val_cur = v_cells.query_fixed(free_val);
            let const_val_cur = v_cells.query_fixed(const_val);

            let exps: Vec<(Expression<F>, Expression<F>, Expression<F>, Expression<F>)> =
                in_registers
                    .into_iter()
                    .clone()
                    .enumerate()
                    .map(|item| {
                        let in_register_selector = v_cells.query_selector(item.1);
                        let cur_values =
                            v_cells.query_advice(cur_registers[item.0], Rotation::cur());
                        let set_register_selector = v_cells.query_selector(set_registers[item.0]);
                        let next_value =
                            v_cells.query_advice(next_registers[item.0], Rotation::cur());
                        (
                            in_register_selector,
                            cur_values,
                            set_register_selector,
                            next_value,
                        )
                    })
                    .collect();

            // This creates a slice iterator, producing references to each value.
            // calculate acc = inA⋅A+inB⋅B
            let acc = exps
                .clone()
                .into_iter()
                .fold(Expression::Constant(F::ZERO), |acc, item| {
                    let tmp = item.0 * item.1;
                    let new_acc = tmp + acc;
                    new_acc
                });
            // calculate acc = acc + inFREE⋅FREE+CONST
            let acc = acc + in_free_cur * free_val_cur + const_val_cur;
            // A+setA⋅(inA⋅A+inB⋅B+inFREE⋅FREE+CONST−A) - A' = 0
            // B+setB⋅(inA⋅A+inB⋅B+inFREE⋅FREE+CONST−B) - B' = 0

            let mut constraints: Vec<Expression<F>> = Vec::with_capacity(exps.len());
            for item in exps {
                let constraint = item.2 * (acc.clone() - item.1.clone()) - item.3 + item.1;
                constraints.push(constraint);
            }

            Constraints::with_selector(enable_vm_selector, constraints)
        });

        let range_check_table = RangeCheckChip::<F, RANGE>::configure(meta);
        let enable_lookup_range = meta.complex_selector();

        cur_registers.clone().into_iter().for_each(|item|{
            meta.lookup(|v_cells|{
                let selector = v_cells.query_selector(enable_lookup_range);
                let val = v_cells.query_advice(item, Rotation::cur());
                vec![(selector.clone() * val, range_check_table.range_table)] 
            });
        });

        next_registers.clone().into_iter().for_each(|item|{
            meta.lookup(|v_cells|{
                let selector = v_cells.query_selector(enable_lookup_range);
                let val = v_cells.query_advice(item, Rotation::cur());
                vec![(selector.clone() * val, range_check_table.range_table)] 
            });
        });

        RegistersConfig {
            enable_vm,
            in_registers,
            cur_registers,
            set_registers,
            next_registers,
            free_val,
            in_free,
            const_val,
            instance,
            enable_lookup_range,
            range_check_table,
            _marker: PhantomData,
        }
    }

    // A′=A+1⋅(0⋅A+0⋅B+1⋅7+0−A)=A+(7−A)=7
    // B′=B+0⋅(0⋅A+0⋅B+1⋅7+0−B)=B
    /**
     * The first instruction involves a free input and
     * this free input is moved into registry A, as its the next value.
     * Therefore, by definition of the selectors, inFREE=1 and setA=1. Also, the value of the other selectors is 0
     */
    pub fn set_register_from_getFreeInput_assignment(
        &self,
        mut layouter: impl Layouter<F>,
        free_value: Value<F>,
        register_id: usize,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        assert!(register_id < self.config.cur_registers.len());

        let next_register_cells = layouter.assign_region(
            || "GetFreeInput Region",
            |mut region| {
                // assign enable vm selector
                self.config.enable_vm.enable(&mut region, 0)?;

                // assign cur_registers
                let cur_registers_cells = self
                    .config
                    .cur_registers
                    .clone()
                    .into_iter()
                    .map(|item| {
                        region
                            .assign_advice(
                                || "assign cur_registers",
                                item,
                                0,
                                || Value::known(F::ZERO),
                            )
                            .unwrap()
                    })
                    .collect::<Vec<AssignedCell<F, F>>>();

                // assign free columns
                self.config.in_free.enable(&mut region, 0)?;
                region.assign_fixed(
                    || "assign free_val",
                    self.config.free_val,
                    0,
                    || free_value.clone(),
                )?;

                // assign const column
                region.assign_fixed(
                    || "assign const val",
                    self.config.const_val,
                    0,
                    || Value::known(F::ZERO),
                )?;

                // assign next_registers
                self.config.set_registers[register_id].enable(&mut region, 0)?;
                let next_register_cells = self
                    .config
                    .next_registers
                    .clone()
                    .into_iter()
                    .enumerate()
                    .map(|item| {
                        if item.0 == register_id {
                            region
                                .assign_advice(
                                    || "assign next_registers from free_value",
                                    item.1,
                                    0,
                                    || free_value,
                                )
                                .unwrap()
                        } else {
                            cur_registers_cells[item.0]
                                .copy_advice(
                                    || "assign next_register from cur_register",
                                    &mut region,
                                    item.1,
                                    0,
                                )
                                .unwrap()
                        }
                    })
                    .collect::<Vec<AssignedCell<F, F>>>();
                Ok(next_register_cells)
            },
        )?;

        Ok(next_register_cells)
    }

    // " 3 => B"
    // const = 3, setB = 1
    // A′=A+0⋅(0⋅A+0⋅B+0⋅FREE+3−A)=A
    // B′=B+1⋅(0⋅A+0⋅B+0⋅FREE+3−B)=B+(3−B)=3
    //
    pub fn set_register_from_const_assignment(
        &self,
        mut layouter: impl Layouter<F>,
        const_value: Value<F>,
        register_id: usize,
        prev_register_cells: Vec<AssignedCell<F, F>>,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        assert!(register_id < self.config.cur_registers.len());

        let next_register_value = layouter.assign_region(
            || "Const Region",
            |mut region| {
                // assign enable vm selector
                self.config.enable_vm.enable(&mut region, 0)?;

                // assign self.config.cur_registers from self.config.next_registers of previous row
                let cur_registers_cells = prev_register_cells
                    .clone()
                    .into_iter()
                    .enumerate()
                    .map(|item| {
                        item.1
                            .copy_advice(
                                || "assign cur_register from next_register of prev row",
                                &mut region,
                                self.config.cur_registers[item.0],
                                0,
                            )
                            .unwrap()
                    })
                    .collect::<Vec<AssignedCell<F, F>>>();

                // assign const column
                self.config.set_registers[register_id].enable(&mut region, 0)?;
                region.assign_fixed(
                    || "assign const val",
                    self.config.const_val,
                    0,
                    || const_value.clone(),
                )?;

                // assign free column
                region.assign_fixed(
                    || "assign free val",
                    self.config.free_val,
                    0,
                    || Value::known(F::ZERO),
                )?;

                // assign next_registers columns
                let next_register_value = self
                    .config
                    .next_registers
                    .clone()
                    .into_iter()
                    .enumerate()
                    .map(|item| {
                        if register_id == item.0 {
                            region
                                .assign_advice(
                                    || "assign next_registers",
                                    item.1,
                                    0,
                                    || const_value,
                                )
                                .unwrap()
                        } else {
                            cur_registers_cells[item.0]
                                .copy_advice(
                                    || "assign next_registers from cur_registers",
                                    &mut region,
                                    item.1,
                                    0,
                                )
                                .unwrap()
                        }
                    })
                    .collect::<Vec<AssignedCell<F, F>>>();
                Ok(next_register_value)
            },
        )?;

        Ok(next_register_value)
    }

    // ADD A + B
    // inA = 1 , inB = 1 , SetA = 1
    pub fn add_assignment(
        &self,
        mut layouter: impl Layouter<F>,
        in_register_id1: usize,
        in_register_id2: usize,
        set_register_id: usize,
        prev_register_cells: Vec<AssignedCell<F, F>>,
    ) -> Result<Vec<AssignedCell<F, F>>, Error> {
        assert!(in_register_id1 < self.config.cur_registers.len());
        assert!(in_register_id2 < self.config.cur_registers.len());
        assert!(set_register_id < self.config.cur_registers.len());

        layouter.assign_region(
            || "Add Region",
            |mut region| {
                // assign enable vm selector
                self.config.enable_vm.enable(&mut region, 0)?;

                // assign enable lookup range selector 
                self.config.enable_lookup_range.enable(&mut region, 0)?;

                // enable copy constraints for advice and assign self.config.cur_registers
                let cur_register_cells = prev_register_cells
                    .clone()
                    .into_iter()
                    .enumerate()
                    .map(|item| {
                        item.1
                            .copy_advice(
                                || "assign cur_registers from next_register of prev row",
                                &mut region,
                                self.config.cur_registers[item.0],
                                0,
                            )
                            .unwrap()
                    })
                    .collect::<Vec<AssignedCell<F, F>>>();

                // assign fixed columns
                self.config.in_registers[in_register_id1].enable(&mut region, 0)?;
                self.config.in_registers[in_register_id2].enable(&mut region, 0)?;
                self.config.set_registers[set_register_id].enable(&mut region, 0)?;

                // assign free and const columns
                region.assign_fixed(
                    || "assign free val",
                    self.config.free_val,
                    0,
                    || Value::known(F::ZERO),
                )?;
                region.assign_fixed(
                    || "assign free val",
                    self.config.const_val,
                    0,
                    || Value::known(F::ZERO),
                )?;

                // assign self.config.next_registers
                let next_register_cells = self
                    .config
                    .next_registers
                    .clone()
                    .into_iter()
                    .enumerate()
                    .map(|item| {
                        if item.0 == set_register_id {
                            region
                                .assign_advice(
                                    || "assign next_registers",
                                    item.1,
                                    0,
                                    || {
                                        prev_register_cells[in_register_id1].value().copied()
                                            + prev_register_cells[in_register_id2].value().copied()
                                    },
                                )
                                .unwrap()
                        } else {
                            cur_register_cells[item.0]
                                .copy_advice(
                                    || "assign next_registers from cur_registers",
                                    &mut region,
                                    item.1,
                                    0,
                                )
                                .unwrap()
                        }
                    })
                    .collect::<Vec<AssignedCell<F, F>>>();
                Ok(next_register_cells)
            },
        )
    }

    // END
    pub fn end_assignment(
        &self,
        mut layouter: impl Layouter<F>,
        in_register_id1: usize,
        in_register_id2: usize,
        prev_register_cells: Vec<AssignedCell<F, F>>,
    ) -> Result<(Vec<AssignedCell<F, F>>, Vec<AssignedCell<F, F>>), Error> {
        layouter.assign_region(
            || "End Region",
            |mut region| {
                // assign enable vm selector
                self.config.enable_vm.enable(&mut region, 0)?;

                // assign selector
                for i in 0..self.config.cur_registers.len() {
                    self.config.set_registers[i].enable(&mut region, 0)?;
                }

                // assign self.config.cur_registers from self.config.next_registers of previous row
                let cur_registers_cells = prev_register_cells
                    .clone()
                    .into_iter()
                    .enumerate()
                    .map(|item| {
                        item.1
                            .copy_advice(
                                || "assign cur_registers from next_register of prev row",
                                &mut region,
                                self.config.cur_registers[item.0],
                                0,
                            )
                            .unwrap()
                    })
                    .collect::<Vec<AssignedCell<F, F>>>();

                // assign free and const columns
                region.assign_fixed(
                    || "assign free val",
                    self.config.free_val,
                    0,
                    || Value::known(F::ZERO),
                )?;
                region.assign_fixed(
                    || "assign free val",
                    self.config.const_val,
                    0,
                    || Value::known(F::ZERO),
                )?;

                // assign self.config.next_register
                let next_register_cells = self
                    .config
                    .next_registers
                    .clone()
                    .into_iter()
                    .enumerate()
                    .map(|item| {
                        region
                            .assign_advice(
                                || "assign next_registers as 0",
                                item.1,
                                0,
                                || Value::known(F::ZERO),
                            )
                            .unwrap()
                    })
                    .collect::<Vec<AssignedCell<F, F>>>();

                Ok((prev_register_cells.clone(), next_register_cells))
            },
        )
    }

    pub fn expose_public(
        &self,
        mut layouter: impl Layouter<F>,
        cell: &AssignedCell<F, F>,
        row: usize,
    ) -> Result<(), Error> {
        layouter.constrain_instance(cell.cell(), self.config.instance, row)
    }
}

#[derive(Default)]
struct SimpleInstructionCircuits<F: PrimeField,const RIGSTER_NUM: usize, const RANGE:usize> {
    free_value: Value<F>,
    const_value: Value<F>,
}

impl<F: PrimeField,const RIGSTER_NUM: usize, const RANGE:usize> Circuit<F> for SimpleInstructionCircuits<F,RIGSTER_NUM,RANGE> {
    type Config = RegistersConfig<F, RIGSTER_NUM, RANGE>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        RegisterChip::<F, RIGSTER_NUM,RANGE>::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let chip = RegisterChip::<F,RIGSTER_NUM,RANGE>::construct(config.clone());
        let A_id: usize = 0;
        let B_id: usize = 1;

        // assign the lookup table
        RangeCheckChip::load_range_table(config.range_check_table,layouter.namespace(||"Layouter::range_check_chip load table"))?;
        
        let prev_register_cells = chip.set_register_from_getFreeInput_assignment(
            layouter.namespace(|| "Layouter::set_register_from_getFreeInput_assignment"),
            self.free_value.clone(),
            A_id,
        )?;
        let prev_register_cells = chip.set_register_from_const_assignment(
            layouter.namespace(|| "Layouter::set_register_from_const_assignment"),
            self.const_value,
            B_id,
            prev_register_cells,
        )?;
        let prev_register_cells = chip.add_assignment(
            layouter.namespace(|| "Layouter::add_assignment"),
            A_id,
            B_id,
            A_id,
            prev_register_cells,
        )?;
        let (cur_registers_cell, prev_register_cells) = chip.end_assignment(
            layouter.namespace(|| "Layouter::end_assignment"),
            A_id,
            B_id,
            prev_register_cells,
        )?;

        chip.expose_public(
            layouter.namespace(|| "Layouter::expose_public"),
            &cur_registers_cell[A_id],
            0,
        )?;
        Ok(())
    }
}
mod tests {

    use super::*;
    use halo2_proofs::{dev::MockProver, pasta::Fp};

    #[test]
    fn test_simple_instructions() {
        let const_value = Fp::from(1);
        let free_value = Fp::from(2);
        let circuit_instance = SimpleInstructionCircuits::<Fp,2,16> {
            free_value: Value::known(const_value.clone()),
            const_value: Value::known(free_value.clone()),
        };

        let output = vec![Fp::from(3)];
        let prover = MockProver::run(5, &circuit_instance, vec![output]).unwrap();
        prover.assert_satisfied();
    }

    #[cfg(feature = "dev-graph")]
    #[test]
    fn plot_register_vm() {
        /// Instance columns have a white background.
        /// Advice columns have a red background.
        /// Fixed columns have a blue background.
        use plotters::prelude::*;

        let root = BitMapBackend::new("register_layout.png", (2048, 7680)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let root = root.titled("Register_Layout", ("sans-serif", 60)).unwrap();

        let const_value = Fp::from(3);
        let free_value = Fp::from(7);
        let circuit_instance = SimpleInstructionCircuits::<Fp,2,16> {
            free_value: Value::known(const_value.clone()),
            const_value: Value::known(free_value.clone()),
        };
        halo2_proofs::dev::CircuitLayout::default()
            .mark_equality_cells(true)
            .show_equality_constraints(false)
            // You can optionally render only a section of the circuit.
            .view_width(0..12)
            //  .view_height(0..16)
            // You can hide labels, which can be useful with smaller areas.
            .show_labels(true)
            // Render the circuit onto your area!
            // The first argument is the size parameter for the circuit.
            .render(5, &circuit_instance, &root)
            .unwrap();
    }
}
