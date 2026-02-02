import flax.linen as nn

# Simple MLP network for (x,y) coordinates
class CoordinateEncoder(nn.Module):
    num_eigenvector_pairs: int
    hidden_dim: int = 256
    num_hidden_layers: int = 3

    @nn.compact
    def __call__(self, xy_coords):
        """
        Args:
            xy_coords: (batch_size, 2) array of (x,y) coordinates

        Returns:
            features_dict: Dictionary containing:
                - left_real: (batch_size, num_features) left eigenvector real components
                - left_imag: (batch_size, num_features) left eigenvector imaginary components
                - right_real: (batch_size, num_features) right eigenvector real components
                - right_imag: (batch_size, num_features) right eigenvector imaginary components
        """
        x = xy_coords

        # Shared backbone
        # Input layer
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)

        # Hidden layers
        for _ in range(self.num_hidden_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)

        # Separate heads for each component with independent transformations
        # Each head has at least one independent layer before the final prediction

        # Left real head
        left_real_hidden = nn.Dense(self.hidden_dim)(x)
        left_real_hidden = nn.relu(left_real_hidden)
        left_real = nn.Dense(self.num_eigenvector_pairs)(left_real_hidden)

        # Left imaginary head
        left_imag_hidden = nn.Dense(self.hidden_dim)(x)
        left_imag_hidden = nn.relu(left_imag_hidden)
        left_imag = nn.Dense(self.num_eigenvector_pairs)(left_imag_hidden)

        # Right real head
        right_real_hidden = nn.Dense(self.hidden_dim)(x)
        right_real_hidden = nn.relu(right_real_hidden)
        right_real = nn.Dense(self.num_eigenvector_pairs)(right_real_hidden)

        # Right imaginary head
        right_imag_hidden = nn.Dense(self.hidden_dim)(x)
        right_imag_hidden = nn.relu(right_imag_hidden)
        right_imag = nn.Dense(self.num_eigenvector_pairs)(right_imag_hidden)

        features_dict = {
            'left_real': left_real,
            'left_imag': left_imag,
            'right_real': right_real,
            'right_imag': right_imag,
        }

        return features_dict, {}