import unittest
import numpy as np
import pandas as pd
import datetime as dt

from rivapy.optimization.models.batterystorage import BatteryStorage


class BatteryStorageTest(unittest.TestCase):
    """Test suite for BatteryStorage class."""

    def setUp(self):
        """Set up common test fixtures."""
        self.eff_in = 0.97
        self.eff_out = 0.97
        self.max_capacity = 100.0
        
        # Create simple price arrays
        self.timesteps = 10
        np.random.seed(42)
        self.bid_prices = np.random.uniform(low=1, high=10, size=self.timesteps).astype(np.float32)
        self.ask_prices = np.random.uniform(low=1, high=10, size=self.timesteps).astype(np.float32)
        
        # Create states, actions, and max_charges
        self.states = np.arange(0, 100.5, step=1.0, dtype=np.float32)
        self.actions = np.arange(-10, 11, dtype=np.float32)
        self.max_charges = np.arange(0, 200.5, step=2.0, dtype=np.float32)

    def test_initialization_without_base_dispatch(self):
        """Test BatteryStorage initialization without base_dispatch parameter."""
        bs = BatteryStorage(
            eff_in=self.eff_in,
            eff_out=self.eff_out,
            max_capacity=self.max_capacity,
            bid_prices=self.bid_prices,
            ask_prices=self.ask_prices,
            states=self.states,
            actions=self.actions,
            max_charges=self.max_charges,
            precompile=False
        )
        
        self.assertEqual(bs._eff_in, self.eff_in)
        self.assertEqual(bs._eff_out, self.eff_out)
        self.assertEqual(bs._max_capacity, self.max_capacity)
        np.testing.assert_array_equal(bs._bid_prices, self.bid_prices)
        np.testing.assert_array_equal(bs._ask_prices, self.ask_prices)
        self.assertEqual(bs._start_state, None)
        self.assertEqual(bs._start_charges, None)
        self.assertEqual(bs._end_state, None)
        # Check that base_dispatch was initialized to zeros
        self.assertEqual(len(bs._base_dispatch), len(self.bid_prices) - 1)
        np.testing.assert_array_equal(bs._base_dispatch, np.zeros(len(self.bid_prices) - 1, dtype=np.float32))

    def test_initialization_with_base_dispatch(self):
        """Test BatteryStorage initialization with base_dispatch parameter."""
        base_dispatch = np.ones(self.timesteps - 1, dtype=np.float32) * 2.0
        
        bs = BatteryStorage(
            eff_in=self.eff_in,
            eff_out=self.eff_out,
            max_capacity=self.max_capacity,
            bid_prices=self.bid_prices,
            ask_prices=self.ask_prices,
            states=self.states,
            actions=self.actions,
            max_charges=self.max_charges,
            base_dispatch=base_dispatch,
            precompile=False
        )
        
        np.testing.assert_array_equal(bs._base_dispatch, base_dispatch)

    def test_initialization_with_start_state(self):
        """Test BatteryStorage initialization with start_state parameter."""
        start_state = 50.0
        
        bs = BatteryStorage(
            eff_in=self.eff_in,
            eff_out=self.eff_out,
            max_capacity=self.max_capacity,
            bid_prices=self.bid_prices,
            ask_prices=self.ask_prices,
            states=self.states,
            actions=self.actions,
            max_charges=self.max_charges,
            start_state=start_state,
            precompile=False
        )
        
        self.assertEqual(bs._start_state, start_state)

    def test_initialization_with_start_charges(self):
        """Test BatteryStorage initialization with start_charges parameter."""
        start_charges = 100.0
        
        bs = BatteryStorage(
            eff_in=self.eff_in,
            eff_out=self.eff_out,
            max_capacity=self.max_capacity,
            bid_prices=self.bid_prices,
            ask_prices=self.ask_prices,
            states=self.states,
            actions=self.actions,
            max_charges=self.max_charges,
            start_charges=start_charges,
            precompile=False
        )
        
        self.assertEqual(bs._start_charges, start_charges)

    def test_initialization_with_end_state(self):
        """Test BatteryStorage initialization with end_state parameter."""
        end_state = 75.0
        
        bs = BatteryStorage(
            eff_in=self.eff_in,
            eff_out=self.eff_out,
            max_capacity=self.max_capacity,
            bid_prices=self.bid_prices,
            ask_prices=self.ask_prices,
            states=self.states,
            actions=self.actions,
            max_charges=self.max_charges,
            end_state=end_state,
            precompile=False
        )
        
        self.assertEqual(bs._end_state, end_state)

    def test_initialization_with_all_parameters(self):
        """Test BatteryStorage initialization with all parameters."""
        start_state = 30.0
        start_charges = 50.0
        end_state = 80.0
        base_dispatch = np.ones(self.timesteps - 1, dtype=np.float32)
        
        bs = BatteryStorage(
            eff_in=self.eff_in,
            eff_out=self.eff_out,
            max_capacity=self.max_capacity,
            bid_prices=self.bid_prices,
            ask_prices=self.ask_prices,
            states=self.states,
            actions=self.actions,
            max_charges=self.max_charges,
            base_dispatch=base_dispatch,
            start_state=start_state,
            start_charges=start_charges,
            end_state=end_state,
            precompile=False
        )
        
        self.assertEqual(bs._eff_in, self.eff_in)
        self.assertEqual(bs._eff_out, self.eff_out)
        self.assertEqual(bs._max_capacity, self.max_capacity)
        self.assertEqual(bs._start_state, start_state)
        self.assertEqual(bs._start_charges, start_charges)
        self.assertEqual(bs._end_state, end_state)
        np.testing.assert_array_equal(bs._base_dispatch, base_dispatch)

    def test_initialization_with_custom_penalty_and_tolerance(self):
        """Test BatteryStorage initialization with custom penalty and tolerance."""
        custom_penalty = -1e10
        custom_tolerance = 1e-6
        
        bs = BatteryStorage(
            eff_in=self.eff_in,
            eff_out=self.eff_out,
            max_capacity=self.max_capacity,
            bid_prices=self.bid_prices,
            ask_prices=self.ask_prices,
            states=self.states,
            actions=self.actions,
            max_charges=self.max_charges,
            penalty=custom_penalty,
            tolerance=custom_tolerance,
            precompile=False
        )
        
        self.assertEqual(bs._penalty, custom_penalty)
        self.assertEqual(bs._tolerance, custom_tolerance)

    def test_precompile_true(self):
        """Test BatteryStorage initialization with precompile=True."""
        # This test ensures that precompile doesn't throw an error
        bs = BatteryStorage(
            eff_in=self.eff_in,
            eff_out=self.eff_out,
            max_capacity=self.max_capacity,
            bid_prices=self.bid_prices,
            ask_prices=self.ask_prices,
            states=self.states,
            actions=self.actions,
            max_charges=self.max_charges,
            precompile=True,
            precompile_timefraction=0.1
        )
        
        self.assertIsNotNone(bs)

    def test_optimize_initializes_attributes(self):
        """Test that optimize() method initializes required attributes."""
        bs = BatteryStorage(
            eff_in=self.eff_in,
            eff_out=self.eff_out,
            max_capacity=self.max_capacity,
            bid_prices=self.bid_prices,
            ask_prices=self.ask_prices,
            states=self.states,
            actions=self.actions,
            max_charges=self.max_charges,
            precompile=False
        )
        
        bs.optimize()
        
        # Check that optimization was successful
        self.assertTrue(hasattr(bs, '_state_choices'))
        self.assertTrue(hasattr(bs, '_charges_choices'))
        self.assertTrue(hasattr(bs, '_action_choices'))
        self.assertTrue(hasattr(bs, '_objective'))
        self.assertTrue(hasattr(bs, '_value_matrix'))

    def test_optimize_result_shapes(self):
        """Test that optimize() produces arrays with correct shapes."""
        bs = BatteryStorage(
            eff_in=self.eff_in,
            eff_out=self.eff_out,
            max_capacity=self.max_capacity,
            bid_prices=self.bid_prices,
            ask_prices=self.ask_prices,
            states=self.states,
            actions=self.actions,
            max_charges=self.max_charges,
            precompile=False
        )
        
        bs.optimize()
        
        # Check shapes
        self.assertEqual(len(bs._state_choices), len(self.bid_prices))
        self.assertEqual(len(bs._charges_choices), len(self.bid_prices))
        self.assertEqual(len(bs._action_choices), len(self.bid_prices) - 1)
        self.assertEqual(len(bs._objective), len(self.bid_prices) - 1)

    def test_optimize_result_types(self):
        """Test that optimize() produces arrays with correct data types."""
        bs = BatteryStorage(
            eff_in=self.eff_in,
            eff_out=self.eff_out,
            max_capacity=self.max_capacity,
            bid_prices=self.bid_prices,
            ask_prices=self.ask_prices,
            states=self.states,
            actions=self.actions,
            max_charges=self.max_charges,
            precompile=False
        )
        
        bs.optimize()
        
        # Check types
        self.assertIsInstance(bs._state_choices, np.ndarray)
        self.assertIsInstance(bs._charges_choices, np.ndarray)
        self.assertIsInstance(bs._action_choices, np.ndarray)
        self.assertIsInstance(bs._objective, np.ndarray)

    def test_create_output_before_optimize_raises_error(self):
        """Test that create_output() raises ValueError if optimize() hasn't been called."""
        bs = BatteryStorage(
            eff_in=self.eff_in,
            eff_out=self.eff_out,
            max_capacity=self.max_capacity,
            bid_prices=self.bid_prices,
            ask_prices=self.ask_prices,
            states=self.states,
            actions=self.actions,
            max_charges=self.max_charges,
            precompile=False
        )
        
        with self.assertRaises(ValueError) as context:
            bs.create_output()
        
        self.assertIn("Not able to create an output", str(context.exception))

    def test_create_output_returns_dataframe(self):
        """Test that create_output() returns a DataFrame after optimization."""
        bs = BatteryStorage(
            eff_in=self.eff_in,
            eff_out=self.eff_out,
            max_capacity=self.max_capacity,
            bid_prices=self.bid_prices,
            ask_prices=self.ask_prices,
            states=self.states,
            actions=self.actions,
            max_charges=self.max_charges,
            precompile=False
        )
        
        bs.optimize()
        output = bs.create_output()
        
        self.assertIsInstance(output, pd.DataFrame)

    def test_create_output_dataframe_shape(self):
        """Test that create_output() DataFrame has correct shape."""
        bs = BatteryStorage(
            eff_in=self.eff_in,
            eff_out=self.eff_out,
            max_capacity=self.max_capacity,
            bid_prices=self.bid_prices,
            ask_prices=self.ask_prices,
            states=self.states,
            actions=self.actions,
            max_charges=self.max_charges,
            precompile=False
        )
        
        bs.optimize()
        output = bs.create_output()
        
        # Check shape
        self.assertEqual(len(output), len(self.bid_prices))

    def test_create_output_dataframe_columns(self):
        """Test that create_output() DataFrame has correct columns."""
        bs = BatteryStorage(
            eff_in=self.eff_in,
            eff_out=self.eff_out,
            max_capacity=self.max_capacity,
            bid_prices=self.bid_prices,
            ask_prices=self.ask_prices,
            states=self.states,
            actions=self.actions,
            max_charges=self.max_charges,
            precompile=False
        )
        
        bs.optimize()
        output = bs.create_output()
        
        # Check columns
        expected_columns = ["Bid Price", "Ask Price", "SOC", "ChargeCycle", "Charging", "Value"]
        self.assertEqual(list(output.columns), expected_columns)

    def test_create_output_dataframe_values_are_numeric(self):
        """Test that create_output() DataFrame contains numeric values."""
        bs = BatteryStorage(
            eff_in=self.eff_in,
            eff_out=self.eff_out,
            max_capacity=self.max_capacity,
            bid_prices=self.bid_prices,
            ask_prices=self.ask_prices,
            states=self.states,
            actions=self.actions,
            max_charges=self.max_charges,
            precompile=False
        )
        
        bs.optimize()
        output = bs.create_output()
        
        # Check that all columns contain numeric values
        for col in output.columns:
            self.assertTrue(np.issubdtype(output[col].dtype, np.number))

    def test_get_dispatch_before_optimize_raises_error(self):
        """Test that get_dispatch() returns None or raises error if optimize() hasn't been called."""
        bs = BatteryStorage(
            eff_in=self.eff_in,
            eff_out=self.eff_out,
            max_capacity=self.max_capacity,
            bid_prices=self.bid_prices,
            ask_prices=self.ask_prices,
            states=self.states,
            actions=self.actions,
            max_charges=self.max_charges,
            precompile=False
        )
        
        # get_dispatch will try to access _action_choices which doesn't exist
        with self.assertRaises(AttributeError):
            bs.get_dispatch()

    def test_get_dispatch_after_optimize(self):
        """Test that get_dispatch() returns correct array after optimization."""
        bs = BatteryStorage(
            eff_in=self.eff_in,
            eff_out=self.eff_out,
            max_capacity=self.max_capacity,
            bid_prices=self.bid_prices,
            ask_prices=self.ask_prices,
            states=self.states,
            actions=self.actions,
            max_charges=self.max_charges,
            precompile=False
        )
        
        bs.optimize()
        dispatch = bs.get_dispatch()
        
        # Check that dispatch is an array
        self.assertIsInstance(dispatch, np.ndarray)
        # Check length
        self.assertEqual(len(dispatch), len(self.bid_prices) - 1)

    def test_get_dispatch_returns_action_choices(self):
        """Test that get_dispatch() returns the same as _action_choices."""
        bs = BatteryStorage(
            eff_in=self.eff_in,
            eff_out=self.eff_out,
            max_capacity=self.max_capacity,
            bid_prices=self.bid_prices,
            ask_prices=self.ask_prices,
            states=self.states,
            actions=self.actions,
            max_charges=self.max_charges,
            precompile=False
        )
        
        bs.optimize()
        dispatch = bs.get_dispatch()
        
        # Check that dispatch equals _action_choices
        np.testing.assert_array_equal(dispatch, bs._action_choices)

    def test_different_efficiency_values(self):
        """Test BatteryStorage with different efficiency values."""
        for eff in [0.80, 0.90, 0.95, 0.99]:
            bs = BatteryStorage(
                eff_in=eff,
                eff_out=eff,
                max_capacity=self.max_capacity,
                bid_prices=self.bid_prices,
                ask_prices=self.ask_prices,
                states=self.states,
                actions=self.actions,
                max_charges=self.max_charges,
                precompile=False
            )
            
            self.assertEqual(bs._eff_in, eff)
            self.assertEqual(bs._eff_out, eff)

    def test_different_max_capacity_values(self):
        """Test BatteryStorage with different max_capacity values."""
        for capacity in [50.0, 100.0, 200.0, 500.0]:
            bs = BatteryStorage(
                eff_in=self.eff_in,
                eff_out=self.eff_out,
                max_capacity=capacity,
                bid_prices=self.bid_prices,
                ask_prices=self.ask_prices,
                states=self.states,
                actions=self.actions,
                max_charges=self.max_charges,
                precompile=False
            )
            
            self.assertEqual(bs._max_capacity, capacity)

    def test_optimize_with_state_start_and_end(self):
        """Test optimization with both start_state and end_state specified."""
        start_state = 25.0
        end_state = 75.0
        
        bs = BatteryStorage(
            eff_in=self.eff_in,
            eff_out=self.eff_out,
            max_capacity=self.max_capacity,
            bid_prices=self.bid_prices,
            ask_prices=self.ask_prices,
            states=self.states,
            actions=self.actions,
            max_charges=self.max_charges,
            start_state=start_state,
            end_state=end_state,
            precompile=False
        )
        
        bs.optimize()
        
        # Check that optimization completed successfully
        self.assertTrue(hasattr(bs, '_state_choices'))

    def test_states_array_constraints(self):
        """Test BatteryStorage respects state array constraints."""
        # Verify that states are stored correctly
        bs = BatteryStorage(
            eff_in=self.eff_in,
            eff_out=self.eff_out,
            max_capacity=self.max_capacity,
            bid_prices=self.bid_prices,
            ask_prices=self.ask_prices,
            states=self.states,
            actions=self.actions,
            max_charges=self.max_charges,
            precompile=False
        )
        
        np.testing.assert_array_equal(bs._states, self.states)

    def test_actions_array_constraints(self):
        """Test BatteryStorage respects actions array constraints."""
        # Verify that actions are stored correctly
        bs = BatteryStorage(
            eff_in=self.eff_in,
            eff_out=self.eff_out,
            max_capacity=self.max_capacity,
            bid_prices=self.bid_prices,
            ask_prices=self.ask_prices,
            states=self.states,
            actions=self.actions,
            max_charges=self.max_charges,
            precompile=False
        )
        
        np.testing.assert_array_equal(bs._actions, self.actions)

    def test_max_charges_array_constraints(self):
        """Test BatteryStorage respects max_charges array constraints."""
        # Verify that max_charges are stored correctly
        bs = BatteryStorage(
            eff_in=self.eff_in,
            eff_out=self.eff_out,
            max_capacity=self.max_capacity,
            bid_prices=self.bid_prices,
            ask_prices=self.ask_prices,
            states=self.states,
            actions=self.actions,
            max_charges=self.max_charges,
            precompile=False
        )
        
        np.testing.assert_array_equal(bs._max_charges, self.max_charges)


if __name__ == '__main__':
    unittest.main()
