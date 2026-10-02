class GuidanceIntervalSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance with interval.
    """

    def _inference_model(self, model, x_t, t, cond, guidance_strength, guidance_interval,
                         guidance_ramp=None, **kwargs):
        if guidance_ramp is not None:
            # (w_at_t1, w_at_t0): linear CFG schedule over flow time — w_at_t1
            # rules the high-noise mode-selection phase, w_at_t0 the low-noise
            # detail phase. Overrides the constant guidance_strength.
            w1, w0 = guidance_ramp
            guidance_strength = w0 + (w1 - w0) * t
        if guidance_interval[0] <= t <= guidance_interval[1]:
            return super()._inference_model(model, x_t, t, cond, guidance_strength=guidance_strength, **kwargs)
        else:
            return super()._inference_model(model, x_t, t, cond, guidance_strength=1, **kwargs)
