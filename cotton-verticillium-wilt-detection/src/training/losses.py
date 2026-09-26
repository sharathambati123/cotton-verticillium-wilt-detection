"""
Custom Loss Functions for Cotton Verticillium Wilt Detection.
Includes Multi-Class Focal Loss and Multi-Task Early-Warning Penalty Loss.
"""

from typing import Optional, List, Dict, Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class nn:
        class Module:
            pass


class MultiClassFocalLoss(nn.Module if HAS_TORCH else object):
    """
    Focal Loss for addressing class imbalance and focusing on hard early-stage wilt samples:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: Optional[List[float]] = None, gamma: float = 2.0):
        if HAS_TORCH:
            super().__init__()
            self.gamma = gamma
            if alpha is not None:
                self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
            else:
                self.alpha = None

    def forward(self, logits: Any, targets: Any) -> Any:
        if HAS_TORCH:
            ce_loss = F.cross_entropy(logits, targets, reduction="none")
            p = torch.exp(-ce_loss)
            focal_term = (1.0 - p) ** self.gamma

            if self.alpha is not None:
                alpha_t = self.alpha.gather(0, targets)
                focal_loss = alpha_t * focal_term * ce_loss
            else:
                focal_loss = focal_term * ce_loss

            return focal_loss.mean()
        return 0.0


class MultiTaskCottonWiltLoss(nn.Module if HAS_TORCH else object):
    """
    Joint loss function for:
    1. 4-class wilt diagnosis (Focal Loss)
    2. Lead-time to visual symptom regression (Smooth L1)
    3. Foliar degradation severity regression (MSE)
    """

    def __init__(
        self,
        focal_alpha: Optional[List[float]] = None,
        focal_gamma: float = 2.0,
        lead_time_weight: float = 0.5,
        severity_weight: float = 0.3
    ):
        if HAS_TORCH:
            super().__init__()
            self.classification_loss = MultiClassFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            self.lead_time_loss = nn.SmoothL1Loss()
            self.severity_loss = nn.MSELoss()
            self.w_lead = lead_time_weight
            self.w_sev = severity_weight

    def forward(self, predictions: Dict[str, Any], targets: Dict[str, Any]) -> Dict[str, Any]:
        if HAS_TORCH:
            cls_loss = self.classification_loss(predictions["logits"], targets["wilt_class"])
            lead_loss = self.lead_time_loss(predictions["days_to_visual"], targets["days_to_visual"])
            sev_loss = self.severity_loss(predictions["foliar_severity"], targets["foliar_severity"])

            total_loss = cls_loss + (self.w_lead * lead_loss) + (self.w_sev * sev_loss)

            return {
                "total_loss": total_loss,
                "classification_loss": cls_loss,
                "lead_time_loss": lead_loss,
                "severity_loss": sev_loss
            }
        return {"total_loss": 0.0}
