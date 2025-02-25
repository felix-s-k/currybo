import torch
import itertools
import numpy as np

def find_most_general_option(x_options, w_options, proxy_model, aggregation_function):

    w_combinations = list(itertools.product(*w_options.values()))
    w_options_combinations = [torch.cat(combo) for combo in w_combinations]
    w_options_tensor = torch.stack(w_options_combinations)

    x_combinations = list(itertools.product(*x_options.values()))
    x_options_combinations = [torch.cat(combo) for combo in x_combinations]
    x_options_tensor = torch.stack(x_options_combinations)

    best_score = -torch.inf
    worst_score = torch.inf
    best_option = None

    for row in x_options_tensor:
        combined = torch.cat((row.repeat(w_options_tensor.shape[0], 1), w_options_tensor), dim=1)
        score = aggregation_function(torch.tensor(np.array([proxy_model(combined)])))
        if score > best_score:
            best_score = score
            best_option = row
        if score < worst_score:
            worst_score = score

    return best_score, worst_score, best_option


def get_generalizability_score(x_options_tensor, w_options, proxy_model, aggregation_function, min, max):

    w_combinations = list(itertools.product(*w_options.values()))
    w_options_combinations = [torch.cat(combo) for combo in w_combinations]
    w_options_tensor = torch.stack(w_options_combinations)

    combined = torch.cat((x_options_tensor.repeat(w_options_tensor.shape[0], 1), w_options_tensor), dim=1)
    score = aggregation_function(torch.tensor(np.array([proxy_model(combined)])))

    #return score

    score_scaled = (score.item() - min) / (max - min)
    return score_scaled