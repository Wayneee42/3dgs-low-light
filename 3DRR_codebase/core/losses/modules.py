from core.libs.losses import exposure_control_loss, low_light_consistency_loss, rgb_reconstruction_loss


class BaseLossModule:
    def __init__(self, name, weight=1.0, enabled=True):
        self.name = name
        self.weight = float(weight)
        self.enabled = bool(enabled)

    def compute(self, context):
        raise NotImplementedError


class RGBReconstructionLoss(BaseLossModule):
    def __init__(self, lambda_ssim):
        super().__init__(name="rgb", weight=1.0, enabled=True)
        self.lambda_ssim = float(lambda_ssim)

    def compute(self, context):
        result = rgb_reconstruction_loss(
            context["rendered"],
            context["supervision_hwc"],
            lambda_ssim=self.lambda_ssim,
        )
        return result["total"], {
            "l1": float(result["l1"].detach().item()),
            "ssim": float(result["ssim"].detach().item()),
        }


class LowLightConsistencyLoss(BaseLossModule):
    def __init__(self, weight):
        super().__init__(name="low_light", weight=weight, enabled=weight > 0.0)

    def compute(self, context):
        loss = low_light_consistency_loss(context["rendered"], context["reference_hwc"])
        return loss, {}


class ExposureControlLoss(BaseLossModule):
    def __init__(self, weight):
        super().__init__(name="exposure", weight=weight, enabled=weight > 0.0)

    def compute(self, context):
        loss = exposure_control_loss(context["rendered"], context["target_mean"])
        return loss, {}


class DepthPriorLoss(BaseLossModule):
    def __init__(self, weight):
        super().__init__(name="depth_prior", weight=weight, enabled=weight > 0.0)

    def compute(self, context):
        raise NotImplementedError("DepthPriorLoss interface is registered in stage 3, but its implementation starts in stage 4.")


class StructurePriorLoss(BaseLossModule):
    def __init__(self, weight):
        super().__init__(name="structure_prior", weight=weight, enabled=weight > 0.0)

    def compute(self, context):
        raise NotImplementedError("StructurePriorLoss interface is registered in stage 3, but its implementation starts in stage 5.")
