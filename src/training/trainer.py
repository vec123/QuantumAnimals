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
        target_irreps_p,
        target_irreps_v,
        interpolant_fn=linear_interpolant,
        noise_fn=sine_noise_schedule,
        num_layers = 2,
        max_degree = 1,
        lr=1e-4,
        verbose = False
    ):
        self.latent_irreps = e3nn.Irreps(latent_irreps)
        self.input_irreps = e3nn.Irreps(input_irreps)
        self.target_irreps_p = e3nn.Irreps(target_irreps_p)
        self.target_irreps_v = e3nn.Irreps(target_irreps_v)
        self.num_layers = num_layers
        self.max_degree = max_degree
        self.lr = lr


        self.interpolant_fn = interpolant_fn
        self.noise_fn = noise_fn

        # One transform to handle the shared conditioner and separate headers
        self._network = hk.without_apply_rng(hk.transform(self._model_fn))
        self.optimizer = optax.adam(learning_rate=lr, eps=1e-3)

        self.verbose = verbose

    def _model_fn(self, residues, p_tau, v_tau, tau):
        # Conditioner (Latent Context)
        node_input_cond = e3nn.concatenate([residues, v_tau], axis=-1)
        x_tilde = EquiJumpDeepNetwork(
            L=1,
            internal_irreps="32x0e + 8x1o",
            output_irreps=self.latent_irreps,
            name="conditioner",
            verbose =  self.verbose
        )(node_input_cond, p_tau)

        #  Time Embedding / Broadcast
        tau_array = e3nn.IrrepsArray("1x0e", jnp.broadcast_to(tau, (p_tau.shape[0], 1)))
        header_input = e3nn.concatenate([x_tilde, v_tau, tau_array], axis=-1)

        # Headers
        b_p = EquiJumpDeepNetwork(L=self.num_layers, internal_irreps="32x0e + 16x1o", 
                                 output_irreps=self.target_irreps_p, name="b_p", verbose=self.verbose)(header_input, p_tau)
        b_v = EquiJumpDeepNetwork(L=self.num_layers, internal_irreps="32x0e + 16x1o", 
                                 output_irreps=self.target_irreps_v, name="b_v", verbose=self.verbose)(header_input, p_tau)
        eta_p = EquiJumpDeepNetwork(L=self.num_layers, internal_irreps="32x0e + 16x1o", 
                                   output_irreps=self.target_irreps_p, name="eta_p", verbose=self.verbose)(header_input, p_tau)
        eta_v = EquiJumpDeepNetwork(L=self.num_layers, internal_irreps="32x0e + 16x1o", 
                                   output_irreps=self.target_irreps_v, name="eta_v", verbose=self.verbose)(header_input, p_tau)

        return {"b_p": b_p, "b_v": b_v, "eta_p": eta_p, "eta_v": eta_v}

    def init_params(self, key, batch):
        R = batch['residues']
        P_0, V_0 = batch['X_init']
        tau = jnp.array([0.5])
        return self._network.init(key, R, P_0, V_0, tau)

    def loss_fn(self, params, key, batch):
        R = batch['residues']
        P_0, V_0 = batch['X_init'] 
        P_1, V_1 = batch['X_end']  

        # Sample Time and Noise
        k_tau, k_z_p, k_z_v = jax.random.split(key, 3)
        tau = jax.random.uniform(k_tau, (1,))
        
        z_p = e3nn.IrrepsArray(P_0.irreps, jax.random.normal(k_z_p, P_0.shape))
        z_v = e3nn.IrrepsArray(V_0.irreps, jax.random.normal(k_z_v, V_0.shape))

        # Stochastic Interpolation
        I_p, dot_I_p = self.interpolant_fn(P_0, P_1, tau)
        I_v, dot_I_v = self.interpolant_fn(V_0, V_1, tau)
        gamma, dot_gamma = self.noise_fn(tau)

        P_tau = I_p + gamma * z_p
        V_tau = I_v + gamma * z_v

        target_vel_p = dot_I_p + dot_gamma * z_p
        target_vel_v = dot_I_v + dot_gamma * z_v

        # Forward Pass
        preds = self._network.apply(params, R, P_tau, V_tau, tau)

        # Objective: 0.5 * ||pred||^2 - (pred · target)
        def cloud_dot_loss_(pred, target):
            print("pred.irreps: ", pred.irreps)
            print("target.irreps: ", target.irreps)
            sq_norm = 0.5 * jnp.mean(e3nn.norm(pred).array**2)
            dot_prod = jnp.mean(e3nn.dot(pred, target).array)
            return sq_norm - dot_prod

        def cloud_dot_loss(pred, target):
            # Instead of norm**2, use the square of the internal array directly
            # This avoids a sqrt() followed by a **2
            sq_norm = 0.5 * jnp.mean(jnp.sum(pred.array**2, axis=-1))
            
            # e3nn.dot is fine, but ensure it's reduced correctly
            dot_prod = jnp.mean(e3nn.dot(pred, target).array)
            
            return sq_norm - dot_prod

        loss_drift = cloud_dot_loss(preds['b_p'], target_vel_p) + \
                     cloud_dot_loss(preds['b_v'], target_vel_v)
        
        loss_noise = cloud_dot_loss(preds['eta_p'], z_p) + \
                     cloud_dot_loss(preds['eta_v'], z_v)
        
        return loss_drift + loss_noise

    @partial(jax.jit, static_argnums=(0,))
    def step(self, params, opt_state, key, batch):
        loss, grads = jax.value_and_grad(self.loss_fn)(params, key, batch)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss