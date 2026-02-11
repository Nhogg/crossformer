from __future__ import annotations

import jax
import jax.numpy as jnp
from rich import print

from crossformer.data.grain.pipelines import dummy_data
from crossformer.utils.spec import spec
from crossformer.utils.type_checking import Action, Batch, Chunked, jtyped, Step, Windowed


def create_step_sample():
    """Generate a single dummy sample reshaped to match Step type requirements."""
    sample = dummy_data()

    # Add window dimension (win=1) to observation
    sample["observation"] = jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), sample["observation"])

    # stack action from (8,) to (w, c, a) for (win, chunk, act)
    win, chunk = 1, 20
    # stack copies
    sample["action"] = jnp.stack([sample["action"]] * chunk, axis=0)  # (chunk, act)
    sample["action"] = jnp.expand_dims(sample["action"], axis=0)  # (win, chunk, act)

    return sample


def create_batch_sample(batch_size=4):
    """Generate multiple samples and stack to create Batch with batch dimension."""
    samples = [create_step_sample() for _ in range(batch_size)]

    # Stack along batch dimension (axis=0)
    batch = jax.tree.map(lambda *x: jnp.stack(x, axis=0), *samples)

    return batch


@jtyped
def process_step(step: Step) -> Step:
    """Simple @jtyped function that processes a Step and returns it."""
    # For demo: just return the step (validates types/shapes)

    return step


@jtyped
def process_step_action(action: Windowed[Chunked[Action]]) -> jax.Array:
    return action


@jtyped
def process_batch(batch: Batch) -> jax.Array:
    """@jtyped function that processes a Batch and returns mean action."""
    # Return mean of action across batch dimension
    return jnp.mean(batch["action"], axis=0)


@jtyped
def process_step_wrong(step: Step) -> Step:
    """Intentionally incorrect @jtyped function to demonstrate error handling."""
    # Transpose action to wrong shape: (1,8,1) -> (1,1,8), but expected (1,8,1)
    step["action"] = jnp.transpose(step["action"], (0, 2, 1))
    return step


def main():
    print("Starting debugging rig for @jtyped decorator...")

    step = create_step_sample()
    batch = create_batch_sample(32)
    print(spec(step, simple=True))
    print(step.keys())

    print("\n1. Testing Step processing...")
    result = process_step(Step(**step))
    print("✓ Step processing successful - types/shapes validated")

    print("1.5 Testing Step action processing...")
    result = process_step_action(step["action"])
    print("✓ Step action processing successful - types/shapes validated")

    print("\n2. Testing Batch processing...")
    batch = create_batch_sample(32)
    result = process_batch(batch)
    print("✓ Batch processing successful - types/shapes validated")

    print("\n3. Testing error case (intentional type violation)...")
    try:
        result = process_step_wrong(step["action"])
        print("✗ Error case unexpectedly passed (this shouldn't happen)")
    except Exception as e:
        print(f"✓ Expected error caught: {e}")

    print("\nDebugging rig completed.")


if __name__ == "__main__":
    main()
