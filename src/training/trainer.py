import jax
import jax.numpy as jnp
import haiku as hk
import e3nn_jax as e3nn
import optax
from functools import partial

from src.training.interpolants import linear_interpolant, sine_noise_schedule
from src.modules.geometric_models import EquiJumpDeepNetwork

class EquiJumpTrainer:
    def __init__(
            self,
              latent_irreps, 
              input_irreps, 
              target_irreps,
              interpolant_fn=linear_interpolant, 
              noise_fn=sine_noise_schedule, 
                lr=1e-4
                 ):
        self.latent_irreps = e3nn.Irreps(latent_irreps)
        self.input_irreps = e3nn.Irreps(input_irreps)
        self.target_irreps = e3nn.Irreps(target_irreps)
        self.lr = lr

        # Define the network transformations
        self.interpolant_fn = interpolant_fn
        self.noise_fn = noise_fn

        # Define the network transformations
        self._cond_transform = hk.without_apply_rng(hk.transform(self._cond_fn))
        self._header_transform = hk.without_apply_rng(hk.transform(self._header_fn))


    def _cond_fn(self, residues, x_init_pos, x_init_feat):
        """fcond(R, Xt) -> Latent Tensor Cloud"""
        # residues: 21x0e, x_init_pos: 1o, x_init_feat: 13x1o
        node_input = e3nn.concatenate([residues, x_init_feat], axis=-1)
        return EquiJumpDeepNetwork(
            L=2, 
            internal_irreps="32x0e + 16x1o",
            output_irreps=self.latent_irreps,
            name="conditioner"
        )(node_input, x_init_pos)

    def _header_fn(self, x_tilde, x_tau_pos, x_tau_feat, tau):
        """Header(X_tilde, X_tau, tau) -> Drift or Noise"""
        # Concatenate latent context, current noisy features, and time
        tau_array = e3nn.IrrepsArray("1x0e", jnp.broadcast_to(tau, (x_tau_pos.shape[0], 1)))
        node_input = e3nn.concatenate([x_tilde, x_tau_feat, tau_array], axis=-1)
        
        return EquiJumpDeepNetwork(
            L=2,
            internal_irreps="32x0e + 16x1o",
            output_irreps=self.target_irreps,
            name="header"
        )(node_input, x_tau_pos)

    def init_params(self, key, batch):
        """Initialize parameters for all 5 sub-networks."""
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        R, (P, V) = batch['residues'], batch['X_init']
        tau = jnp.array([0.5])

        params = {}
        # 1. Conditioner
        params['f_cond'] = self._cond_transform.init(k1, R, P, V)
        x_tilde = self._cond_transform.apply(params['f_cond'], R, P, V)

        # 2. Headers (Drift and Noise)
        params['b_V'] = self._header_transform.init(k2, x_tilde, P, V, tau)
        params['b_P'] = self._header_transform.init(k3, x_tilde, P, V, tau)
        params['eta_V'] = self._header_transform.init(k4, x_tilde, P, V, tau)
        params['eta_P'] = self._header_transform.init(k5, x_tilde, P, V, tau)
        
        return params

    def loss_fn(self, params, key, batch):
        R = batch['residues']
        P_0, V_0 = batch['X_init'] # x0
        P_1, V_1 = batch['X_end']  # x1

        # Sample Time and Noise
        k_tau, k_z = jax.random.split(key)
        tau = jax.random.uniform(k_tau, (1,))
        
        # Standard Normal Noise Z
        z_p = jax.random.normal(k_z, P_0.shape)
        z_v = jax.random.normal(k_z, V_0.shape)

        # Compute Interpolant and Schedule
        # I_p is the value, dot_I_p is the derivative w.r.t tau
        I_p, dot_I_p = self.interpolant_fn(P_0, P_1, tau)
        I_v, dot_I_v = self.interpolant_fn(V_0, V_1, tau)
        
        gamma, dot_gamma = self.noise_fn(tau)

        # Stochastic Interpolation
        # X_tau = I(tau) + gamma(tau)Z
        P_tau = I_p + gamma * z_p
        V_tau = I_v + gamma * z_v
        
        # Target Velocity: d/dtau [ I(tau) + gamma(tau)Z ]
        target_vel_p = dot_I_p + dot_gamma * z_p
        target_vel_v = dot_I_v + dot_gamma * z_v

        # Model Predictions
        x_tilde = self._cond_transform.apply(params['f_cond'], R, P_0, V_0)
        
        # Headers receive (Conditioning, Current State, Time)
        hat_b_p = self._header_transform.apply(params['b_P'], x_tilde, P_tau, V_tau, tau)
        hat_b_v = self._header_transform.apply(params['b_V'], x_tilde, P_tau, V_tau, tau)
        hat_eta_p = self._header_transform.apply(params['eta_P'], x_tilde, P_tau, V_tau, tau)
        hat_eta_v = self._header_transform.apply(params['eta_V'], x_tilde, P_tau, V_tau, tau)

        # Flexible Tensor Cloud Loss (Eq 6 & 7)
        def cloud_dot_loss(pred, target):
            # 0.5 * ||pred||^2 - (pred · target)
            sq_norm = 0.5 * jnp.mean(e3nn.norm(pred).array**2)
            dot_prod = jnp.mean(e3nn.dot(pred, target).array)
            return sq_norm - dot_prod

        # Total Loss = Loss(Drift) + Loss(Noise)
        # Summing P and V components implements the Xi · Xj dot product defined in paper
        loss_drift = cloud_dot_loss(hat_b_p, target_vel_p) + cloud_dot_loss(hat_b_v, target_vel_v)
        loss_noise = cloud_dot_loss(hat_eta_p, z_p) + cloud_dot_loss(hat_eta_v, z_v)
        
        return loss_drift + loss_noise


    @partial(jax.jit, static_argnums=(0,))
    def step(self, params, opt_state, key, batch):
        loss, grads = jax.value_and_grad(self.loss_fn)(params, key, batch)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss