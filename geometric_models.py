import flax
import jraph
import haiku as hk
import jax
import jax.numpy as jnp
import e3nn_jax as e3nn


class SelfInteraction(hk.Module):
    def __init__(self, target_irreps, l_max = 3, name=None, verbose = True):
        super().__init__(name=name)
        self.target_irreps = e3nn.Irreps(target_irreps)
        self.l_max = l_max
        self.verbose = verbose
    def __call__(self, node_features: e3nn.IrrepsArray):
        # 1. Tensor Product: V_i \otimes V_i 
        v_sq = e3nn.tensor_product(node_features, node_features).regroup()
        v_sq = v_sq.filter(lmax=self.l_max)

        # skip connection
        v_intermediate = e3nn.concatenate([node_features, v_sq])

        # Gating
        scalars = v_intermediate.filtered("0e")
        vectors = v_intermediate.filtered("1o")
        v_lengths = e3nn.norm(vectors) 
        
        gate_input = e3nn.concatenate([scalars, v_lengths], axis=-1)
        
        # MLP determines the "strength" of each geometric feature
        gate = hk.nets.MLP(
            [32, v_intermediate.irreps.num_irreps], 
            name="si_gate_mlp"
        )(gate_input.array)
        
        # Residual-style update projection
        v_out = e3nn.haiku.Linear(self.target_irreps)(v_intermediate * gate)
        if self.verbose:
                print("§§§§§§§§§ SelfInteraction §§§§§§§§§")
                print("in.irreps: ", node_features.irreps)
                print("v_intermediate.irreps: ", v_intermediate.irreps)
                print("v_out.irreps: ", v_out.irreps)
        return  v_out
    
class SpatialConvolution(hk.Module):
    def __init__(self, target_irreps, sh_lmax=3, name=None, verbose = True):
        super().__init__(name=name)
        self.target_irreps = e3nn.Irreps(target_irreps)
        self.sh_lmax = sh_lmax
        self.verbose = verbose

    def __call__(self, graph: jraph.GraphsTuple, positions: jnp.ndarray):

        def update_edge_fn(edge_features, sender_features, receiver_features, globals):
            # rel_pos = P_alpha_j - P_alpha_i
            rel_pos = positions[graph.receivers] - positions[graph.senders]
            rel_pos = e3nn.IrrepsArray("1x1o", rel_pos)
            dist = e3nn.norm(rel_pos)
            
            # Spherical Harmonics Path
            Y = e3nn.spherical_harmonics(list(range(1, self.sh_lmax+1)), rel_pos, True)
            R = e3nn.soft_one_hot_linspace(dist.array, start=0.0, end=10.0, number=16, basis='gaussian', start_zero=False, end_zero = False)
            tp_message = e3nn.tensor_product(sender_features, Y).regroup()
            geo_features = e3nn.concatenate([sender_features, tp_message])

            # Gating message by distance and neighbor cloud magnitudes
            r0e, s0e, = receiver_features.filtered("0e"), receiver_features.filtered("0e")
            r0o, s0o = receiver_features.filtered("0o"),receiver_features.filtered("0o")
            R_squeezed = jnp.squeeze(R, axis=1)
            R_irreps = e3nn.IrrepsArray(f"{R_squeezed.shape[-1]}x0e", R_squeezed)    
            v_intermediate = e3nn.concatenate([ receiver_features.filtered(lmax=0),sender_features.filtered(lmax=0)])
            gate_in = jnp.concatenate([v_intermediate.array], axis=-1)

            gate = hk.nets.MLP([32, geo_features.irreps.num_irreps])(gate_in)
            
            out_put_features = geo_features * gate
            return out_put_features

        def update_node_fn(nodes, senders, receivers, globals):
            # 'nodes' contains [V_i, degree]
            # 'receivers' contains the sum of V_tilde
    
            # Extract the degree (k) we stored earlier
            # Assuming it's the last 0e channel
            k = nodes.filtered("0e").array[:, -1:] 
            
            # Perform the division (1/k * sum(V_tilde))
            receivers = receivers.filtered(lmax=self.sh_lmax)
            normalized_messages = receivers / jnp.maximum(k, 1.0)
            
            # V = Linear(V + normalized_messages)
            # filter 'nodes' to remove the extra degree scalar before the sum
            v_current = nodes.filtered(self.target_irreps) 
        
            # Check if they are identical in both symmetry and dimensions
            if v_current.irreps == normalized_messages.irreps and v_current.shape == normalized_messages.shape:
                v_residual = v_current
            else:
                v_residual =e3nn.haiku.Linear(normalized_messages.irreps, name="res_proj", force_irreps_out=True)(v_current)
          
            out = v_residual + normalized_messages
            if self.verbose:
                print("§§§§§§§§§ SpatialConvolution §§§§§§§§§")
                print("in.irreps: ",v_current.irreps)
                print("msg.irreps: ", normalized_messages.irreps)
                print("out.irreps: ", out.irreps)
            return out.filtered(lmax=self.sh_lmax)

        return jraph.GraphNetwork(update_edge_fn, update_node_fn)(graph)

class EquiJumpLayer(hk.Module):
    def __init__(self, target_irreps, name=None, verbose = True):
        super().__init__(name=name)
        self.target_irreps = e3nn.Irreps(target_irreps)
        self.verbose = verbose

    def __call__(self, graph, positions):
        # Self Interaction (Update V based on internal residue structure)
        in_irreps = graph.nodes
        h = SelfInteraction(self.target_irreps)(graph.nodes)
        graph = graph._replace(nodes=h)
        
        #  Spatial Convolution (Update V based on neighbor C_alpha)
        graph = SpatialConvolution(self.target_irreps)(graph, positions)
        
        msg = graph.nodes
        if in_irreps.irreps == msg.irreps and in_irreps.shape == msg.shape:
            skip = in_irreps
        else:
            skip =e3nn.haiku.Linear(msg.irreps, name="res_proj", force_irreps_out=True)(in_irreps)

        res = msg+skip
        #h_norm = e3nn.haiku.LayerNorm(self.target_irreps)(res)
        h_norm =res
   
        if self.verbose:
            print("§§§§§§§§§ EquiJumpLayer §§§§§§§§§")
            print("in.irreps : ", in_irreps.irreps)
            print("msg.irreps : ", msg.irreps)
            print("out.irreps: ", h_norm.irreps)

        return h_norm
    
class EquiJumpDeepNetwork(hk.Module):
    def __init__(self, L=4, input_irreps="32x0e + 16x1o", internal_irreps="32x0e + 16x1o", 
                 output_irreps="16x0e", distance_cutoff=10.0, name=None, verbose = True):
        super().__init__(name=name)
        self.L = L
        
        self.input_irreps = e3nn.Irreps(input_irreps)
        self.internal_irreps = e3nn.Irreps(internal_irreps)
        self.output_irreps = e3nn.Irreps(output_irreps)

        self.distance_cutoff = distance_cutoff

    def __call__(self, node_features: e3nn.IrrepsArray, positions: jnp.ndarray):

        # --- Internal Graph Construction ---
        num_nodes = positions.shape[0]

        senders, receivers = e3nn.radius_graph(
            pos=positions, 
            r_max=self.distance_cutoff
        )
        # Create the jraph structure
        h_jraph = jraph.GraphsTuple(
            nodes=node_features,
            edges=None, 
            senders=senders,
            receivers=receivers,
            n_node=jnp.array([num_nodes]),
            n_edge=jnp.array([len(senders)]),
            globals=None
        )
        print("made graph:")
        # --- Forward Pass ---
        # Initial Embedding
        h_nodes = e3nn.haiku.Linear(self.input_irreps, name="init_embed")(h_jraph.nodes)
        h = h_jraph._replace(nodes=h_nodes)
        
        history = [h.nodes]

        # First Self-Interaction
        h_0_nodes = SelfInteraction(target_irreps=self.internal_irreps)(h.nodes)
        history.append(h_0_nodes)
        h = h._replace(nodes=h_0_nodes)

        # Iterative Message Passing
        for l in range(self.L):
            h_l_nodes = EquiJumpLayer(
                target_irreps=self.internal_irreps, 
                name=f"equijump_layer_{l}"
            )(h, positions)
            history.append(h_l_nodes)
            h = h._replace(nodes=h_l_nodes)

        # Jumping Knowledge Aggregation
        h_agg_nodes = e3nn.concatenate(history, axis=-1)
        h_agg_nodes = e3nn.haiku.Linear(self.internal_irreps, name="agg_linear")(h_agg_nodes)
        
        h_out_nodes = SelfInteraction(target_irreps=self.output_irreps)(h_agg_nodes)

        return h_out_nodes