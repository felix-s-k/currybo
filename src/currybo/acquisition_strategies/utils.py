from typing import List, Dict, Tuple
import torch


def tensor_to_fixed_features_list(
        tensor: torch.Tensor,
        index_offset: int = 0
) -> List[Dict[int, float]]:
    """
    Convert  a tensor of categorical options to the `fixed_features_list` format, as required by botorch's \
    `optimize_acqf_mixed` function.

    Args:
        tensor: A `num_options x num_features` tensor of categorical options.
        index_offset: The index of the first feature in the tensor.

    Returns:
        A list of dictionaries, where each dictionary maps a feature index to a value.
    """
    fixed_feature_list = []
    for row in tensor:
        row_dict = {i + index_offset: value for i, value in enumerate(row)}
        fixed_feature_list.append(row_dict)
    return fixed_feature_list


def create_all_permutations(x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create all possible permutations of rows of two tensors.

    Args:
        x1: A `n x d1` tensor.
        x2: A `m x d2` tensor.

    Returns:
        torch.Tensor: A `n*m x (d1 + d2)` tensor of permutations
        torch.Tensor A `n*m x 2` tensor of indices of the permutations.
    """
    x1_ = x1.repeat_interleave(x2.shape[0], dim=0)
    x2_ = x2.repeat(x1.shape[0], 1)
    joint_x = torch.cat([x1_, x2_], dim=-1)

    x1_idx = torch.arange(x1.shape[0]).repeat_interleave(x2.shape[0])
    x2_idx = torch.arange(x2.shape[0]).repeat(x1.shape[0])
    joint_idx = torch.stack([x1_idx, x2_idx], dim=-1)

    return joint_x, joint_idx


if __name__ == "__main__":

    tensor_1 = torch.tensor([[1, 2], [3, 4]])
    tensor_2 = torch.tensor([[5], [6], [7]])

    print(create_all_permutations(tensor_1, tensor_2))
