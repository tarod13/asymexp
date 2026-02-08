import flax.linen as nn

class CoordinateEncoder(nn.Module):
    num_eigenvector_pairs: int
    hidden_dim: int = 256
    num_hidden_layers: int = 3
    # New ablation flags
    use_residual: bool = True
    use_layernorm: bool = True

    @nn.compact
    def __call__(self, xy_coords):
        x = xy_coords

        # Shared backbone
        # Input layer
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)

        # Hidden layers
        for _ in range(self.num_hidden_layers):
            residual = x  # Save state for skip connection
            
            # Conditional Pre-LayerNorm
            if self.use_layernorm:
                x = nn.LayerNorm()(x)
                
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
            
            # Conditional Residual Connection
            if self.use_residual:
                x = x + residual

        # Helper for head creation
        def make_head_prediction(input_x, name):
            h = input_x
            
            # Conditional Normalization for heads
            if self.use_layernorm:
                h = nn.LayerNorm()(h)
                
            h = nn.Dense(self.hidden_dim, name=f'{name}_hidden')(h)
            h = nn.relu(h)
            return nn.Dense(self.num_eigenvector_pairs, name=name)(h)

        features_dict = {
            'left_real': make_head_prediction(x, 'left_real'),
            'left_imag': make_head_prediction(x, 'left_imag'),
            'right_real': make_head_prediction(x, 'right_real'),
            'right_imag': make_head_prediction(x, 'right_imag'),
        }

        return features_dict, {}