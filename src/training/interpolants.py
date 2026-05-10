import jax.numpy as jnp

def linear_interpolant(x0, x1, tau):
    """I(tau, x0, x1) = (1-tau)x0 + tau*x1"""
    val = (1 - tau) * x0 + tau * x1
    dot = x1 - x0
    return val, dot

def sine_noise_schedule(tau):
    """gamma(tau) = sin(pi * tau)"""
    const  = 0.1
    val = jnp.sin(jnp.pi * tau)
    dot = jnp.pi * jnp.cos(jnp.pi * tau)
    return const*val, const*dot

# Example of a different schedule you could swap in:
def vp_noise_schedule(tau):
    """Variance Preserving-style gamma(tau) = sqrt(1 - tau^2)"""
    val = jnp.sqrt(1 - tau**2 + 1e-6)
    dot = -tau / val
    return val, dot