from geometric_models import EquiJumpDeepNetwork
import haiku as hk
import jax
import optax
import jax.numpy as jnp

class EquiJumpTrainer():

    def init_module(self,
                    input_irreps = "32x0e + 16x1o",  
                    internal_irreps = "32x0e +  124x1o + 10x2e",
                    output_irreps =  "14x1e" ):
        
        output_irreps =  output_irreps
        input_irreps = input_irreps
        internal_irreps = internal_irreps

        model_def = lambda g, p: EquiJumpDeepNetwork(L=2,
                                                    input_irreps = input_irreps,
                                                    internal_irreps = internal_irreps,
                                                    output_irreps = output_irreps)(g, p)

        model = hk.without_apply_rng(hk.transform(model_def))
    
    def init_modules(self):

        # Tensor Cloud + Residue Field -> Tensor Cloud i.e. 1e (P_i) + 13x1e (V_ij) (per node P_i)
        conditioning = self.init_module(
                    input_irreps = "32x0e + 16x1o",  
                    internal_irreps = "32x0e +  124x1o + 10x2e",
                    output_irreps =  "32x0e + 16x1o",)
        
        # tau scalar +  Tensor Cloud cond + Tensor Cloud interp -> dV, i.e. 13x1e (dV_ij) (per node P_i)
        V_drift = self.init_module(
                    input_irreps = "32x0e + 16x1o",  
                    internal_irreps = "32x0e +  124x1o + 10x2e",
                    output_irreps =  "32x0e + 16x1o",)
        
        # tau scalar + Tensor Cloud cond + Tensor Cloud interp -> dP, i.e. 1e  (dP_i) (per node  P_i)
        P_drift = self.init_module(
                    input_irreps = "32x0e + 16x1o",  
                    internal_irreps = "32x0e +  124x1o + 10x2e",
                    output_irreps =  "32x0e + 16x1o",)
        
        # tau scalar + Tensor Cloud cond + Tensor Cloud interp -> dV, i.e. 13x1e (dV_ij) (per node P_i)
        V_noise = self.init_module(
                    input_irreps = "32x0e + 16x1o",  
                    internal_irreps = "32x0e +  124x1o + 10x2e",
                    output_irreps =  "32x0e + 16x1o",)
        
        # tau scalar +  Tensor Cloud cond + Tensor Cloud interp -> dP, i.e. 1e  (dP_i) (per node  P_i)
        P_noise = self.init_module(
                    input_irreps = "32x0e + 16x1o",  
                    internal_irreps = "32x0e +  124x1o + 10x2e",
                    output_irreps =  "32x0e + 16x1o",)
        
        return conditioning, V_drift, P_drift, V_noise, P_noise
    
    def cosine_gamma_schedule(self, tau):
        # g(tau) = sin(pi * tau)
        g = jnp.sin(jnp.pi * tau)
        g_dot = jnp.pi * jnp.cos(jnp.pi * tau)
        return g, g_dot

    def linear_interpolate(self,key, X_t_init, X_t_end, gamma_fn):
        # computes the end-point fixed stochastic interpolant between two states
        # as well as the interpolant derivate at that time
        """
        X_t_init/end are tuples: (P, V) = the Tensor Cloud
        P: (num_nodes, 3) - Positions (Irreps "1o")
        V: (num_nodes, 13, 3) - Vector features (Irreps "13x1o")
        """
        
        k1, k2, k3 = jax.random.split(key, 3)
        # uniform between 0 and 1 with small epsilon offset
        tau = jax.random.uniform(k1, (1,)) * (1 - 2e-5) + 1e-5

        # sampled from isotropic gaussian with dim num_nodes,3
        P_z_tau = jax.random.normal(k2, X_t_init[0].shape)
        V_z_tau = jax.random.normal(k3, X_t_init[1].shape)


        # P_init and P end have shape (num_nodes, 3)
        # V_init and V_end have shape (num_nodes,13, 3)
        P_init, V_init = X_t_init
        P_end, V_end = X_t_end
       
        g_tau, g_dot_tau = gamma_fn(tau)
        
        # stochastic interpolant state
        P_tau = (1 - tau) * P_init + tau * P_end + g_tau * P_z_tau
        V_tau = (1 - tau) * V_init + tau * V_end + g_tau * V_z_tau

        I_vel_P = (P_end - P_init) + g_dot_tau * P_z_tau
        I_vel_V = (V_end - V_init) + g_dot_tau * V_z_tau

        return (P_tau, V_tau), (P_z_tau, V_z_tau), (I_vel_P, I_vel_V), tau
    
    def loss_fn(self, params, key, batch):
        X_init = batch['X_init'] # Xt in paper
        X_end = batch['X_end']   # Xt+1 in paper
        R = batch['residues']    # Protein sequence R
        
        # 1. Interpolate
        X_tau, z_tau, I_vel_target, tau = self.linear_interpolate(
            key, X_init, X_end, self.cosine_gamma_schedule
        )
        
        #  Conditioner: fcond(R, Xt)
        # Note: Independent of tau for efficiency
        X_tilde = self.conditioning.apply(params['conditioning'], R, X_init)

        # 3. Predict Headers: take (X_tilde, X_tau, tau)
        # Drift headers b_hat
        b_V = self.V_drift.apply(params['V_drift'], X_tilde, X_tau, tau)
        b_P = self.P_drift.apply(params['P_drift'], X_tilde, X_tau, tau)
        
        # Noise headers eta_hat
        eta_V = self.V_noise.apply(params['V_noise'], X_tilde, X_tau, tau)
        eta_P = self.P_noise.apply(params['P_noise'], X_tilde, X_tau, tau)

        # 4. Compute Loss per Algorithm 1, Line 9
        # Objective: 0.5 * ||b||^2 - b · TargetVel + 0.5 * ||eta||^2 - eta · Z
        
        # Drift Loss (Vector dot products summed over nodes)
        # TargetVel is (I_vel_P, I_vel_V)
        loss_drift_P = 0.5 * jnp.mean(jnp.sum(b_P**2, axis=-1)) - jnp.mean(jnp.sum(b_P * I_vel_target[0], axis=-1))
        loss_drift_V = 0.5 * jnp.mean(jnp.sum(b_V**2, axis=(-1, -2))) - jnp.mean(jnp.sum(b_V * I_vel_target[1], axis=(-1, -2)))
        
        # Noise Loss
        # z_tau is (P_z_tau, V_z_tau)
        loss_noise_P = 0.5 * jnp.mean(jnp.sum(eta_P**2, axis=-1)) - jnp.mean(jnp.sum(eta_P * z_tau[0], axis=-1))
        loss_noise_V = 0.5 * jnp.mean(jnp.sum(eta_V**2, axis=(-1, -2))) - jnp.mean(jnp.sum(eta_V * z_tau[1], axis=(-1, -2)))

        return loss_drift_P + loss_drift_V + loss_noise_P + loss_noise_V
    
    @jax.jit
    def gradient_step(self, params, opt_state, key, batch):
        """
        Performs one iteration of gradient descent.
        """
        # Split key for interpolation stochasticity
        key, step_key = jax.random.split(key)
        
        # Calculate loss and gradients
        loss, grads = jax.value_and_grad(self.loss_fn)(params, step_key, batch)
        
        # Update parameters using Optax
        updates, next_opt_state = self.optimizer.update(grads, opt_state, params)
        next_params = optax.apply_updates(params, updates)
        
        return next_params, next_opt_state, loss