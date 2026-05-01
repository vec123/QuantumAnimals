import flax
import jraph
import haiku as hk
import jax
import jax.numpy as jnp
import e3nn_jax as e3nn


class SelfInteraction(hk.Module):
    """ Self-Interaction"""
    def __init__(self, target_irreps, name=None):
        super().__init__(name=name)
        self.target_irreps = e3nn.Irreps(target_irreps)

    def __call__(self, graphs):
        def update_node_fn(nodes, senders, receivers, globals):
            #  V <- V \otimes V (Tensor Square)

            # computes tensor product between all nodes, 
            # combines multiplicities, sorts by L and parity
            v_sq = e3nn.tensor_product(nodes, nodes).regroup()
            
            # keep scalars for gating
            scalars = nodes.filtered(keep="0e")
            gate = hk.nets.MLP([16, v_sq.irreps.num_irreps])(scalars.array)

            v_weighted = v_sq * gate

            # learnable linear combination of input tensors with target_irreps as output
            # respects equivariance, only mixes 
            v_out = e3nn.haiku.Linear(self.target_irreps)(v_weighted)
            return v_out

        return graphs._replace(
            nodes=jax.vmap(update_node_fn)(graphs.nodes, None, None, None)
        )
    
class SpatialConvolution(hk.Module):
    """ Spatial Convolution"""
    def __init__(self, target_irreps, denominator, sh_lmax=3, name=None):
        super().__init__(name=name)
        self.target_irreps = e3nn.Irreps(target_irreps)
        self.denominator = denominator
        self.sh_lmax = sh_lmax

    def __call__(self, graphs, positions):
        def update_edge_fn(edge_features, sender_features, receiver_features, globals):
            #  kNN is handled by your jraph graph structure
            rel_pos = positions[graphs.receivers] - positions[graphs.senders] # \tilde{P} - P_i
            dist = jnp.linalg.norm(rel_pos, axis=-1, keepdims=True)
            
            # Embedding (R) and Spherical Harmonics (Y)
            # We use a simple radial basis for the MLP part of step 4
            R = e3nn.soft_one_hot_linspace(dist, start=0.0, end=2.0, number=8, basis='gaussian')
            Y = e3nn.spherical_harmonics(list(range(1, self.sh_lmax+1)), rel_pos, True)
            
            # \tilde{V} \otimes Y gated by MLP(R)
            # In e3nn, this is often a 'Linear' followed by TP, or a gated TP
            messages = e3nn.tensor_product(sender_features, Y)
            
            # Use the radial embedding R to scale the messages (MLP equivalent)
            radial_weights = hk.nets.MLP([16, messages.irreps.num_irreps])(R)
            return messages * radial_weights

        def update_node_fn(nodes, senders, receivers, globals):
            # V = Linear(V + 1/k * sum(V_tilde))
            aggregated = receivers / self.denominator
            return e3nn.haiku.Linear(self.target_irreps)(nodes + aggregated)

        return jraph.GraphNetwork(update_edge_fn, update_node_fn)(graphs)


class EquiJumpDeepNetwork(hk.Module):
    """Full Network"""
    def __init__(self, L=4, target_irreps="32x0e + 16x1o", name=None):
        super().__init__(name=name)
        self.L = L
        self.target_irreps = e3nn.Irreps(target_irreps)

    def __call__(self, graphs):
        positions = graphs.nodes[..., -3:] # P from TensorCloud
        
        #Self-Interaction(X)
        h = SelfInteraction(self.target_irreps, name="init_si")(graphs)
        history = [h.nodes]

        # for l in [0, L)
        for l in range(self.L):
            # Self-Interaction
            h = SelfInteraction(self.target_irreps, name=f"si_{l}")(h)
            
            #  Spatial Convolution
            h_next = SpatialConvolution(self.target_irreps, 1.5, name=f"conv_{l}")(h, positions)
            
            # LayerNorm (or Norm) + Residual: H_{l+1} + H_l
            # e3nn.norm is used for Equivariant LayerNorm
            h_nodes = e3nn.norm(h_next.nodes + h.nodes) 
            h = h._replace(nodes=h_nodes)
            history.append(h.nodes)

        # Hagg = Linear(Concatenate all H_l)
        h_agg_nodes = e3nn.concatenate(history)
        h_agg = h._replace(nodes=e3nn.haiku.Linear(self.target_irreps)(h_agg_nodes))

        # Hout = Self-Interaction(Hagg)
        h_out = SelfInteraction(self.target_irreps, name="final_si")(h_agg)
        
        return h_out