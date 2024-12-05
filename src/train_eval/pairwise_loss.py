import torch

def pairwise_loss(score_vuln: torch.Tensor, score_benign: torch.Tensor) -> torch.Tensor:
    """
    Compute RankNet-style BCE loss for pairwise ranking.

    Args:
        score_vuln (torch.Tensor): Scores for vulnerable code snippets.
        score_benign (torch.Tensor): Scores for benign code snippets.

    Returns:
        torch.Tensor: Loss value.
    """
    # RankNet Binary Cross Entropy (BCE) loss using logistic function: sᵢⱼ = σ(sᵢ – sⱼ)
    # Equivalent to log sigmoid function because we always want score_vuln > score_benign => yᵢⱼ = 1
    # https://towardsdatascience.com/learning-to-rank-a-complete-guide-to-ranking-using-machine-learning-4c9688d370d4
    return torch.nn.functional.logsigmoid(score_vuln - score_benign).mean()
