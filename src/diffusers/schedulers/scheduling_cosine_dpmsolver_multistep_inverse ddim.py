# Copyright 2025 TSAIL Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# DISCLAIMER: This file is strongly influenced by https://github.com/LuChengTHU/dpm-solver and https://github.com/NVlabs/edm

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ..configuration_utils import ConfigMixin, register_to_config
from .scheduling_dpmsolver_sde import BrownianTreeNoiseSampler
from .scheduling_utils import SchedulerMixin, SchedulerOutput


class CosineDPMSolverMultistepInverseScheduler(SchedulerMixin, ConfigMixin):
    """
    Implements a variant of `DPMSolverMultistepScheduler` with cosine schedule, proposed by Nichol and Dhariwal (2021).
    This scheduler was used in Stable Audio Open [1].

    [1] Evans, Parker, et al. "Stable Audio Open" https://huggingface.co/papers/2407.14358

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        sigma_min (`float`, *optional*, defaults to 0.3):
            Minimum noise magnitude in the sigma schedule. This was set to 0.3 in Stable Audio Open [1].
        sigma_max (`float`, *optional*, defaults to 500):
            Maximum noise magnitude in the sigma schedule. This was set to 500 in Stable Audio Open [1].
        sigma_data (`float`, *optional*, defaults to 1.0):
            The standard deviation of the data distribution. This is set to 1.0 in Stable Audio Open [1].
        sigma_schedule (`str`, *optional*, defaults to `exponential`):
            Sigma schedule to compute the `sigmas`. By default, we the schedule introduced in the EDM paper
            (https://huggingface.co/papers/2206.00364). Other acceptable value is "exponential". The exponential
            schedule was incorporated in this model: https://huggingface.co/stabilityai/cosxl.
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        solver_order (`int`, defaults to 2):
            The DPMSolver order which can be `1` or `2`. It is recommended to use `solver_order=2`.
        prediction_type (`str`, defaults to `v_prediction`, *optional*):
            Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
            `sample` (directly predicts the noisy sample`) or `v_prediction` (see section 2.4 of [Imagen
            Video](https://imagen.research.google/video/paper.pdf) paper).
        solver_type (`str`, defaults to `midpoint`):
            Solver type for the second-order solver; can be `midpoint` or `heun`. The solver type slightly affects the
            sample quality, especially for a small number of steps. It is recommended to use `midpoint` solvers.
        lower_order_final (`bool`, defaults to `True`):
            Whether to use lower-order solvers in the final steps. Only valid for < 15 inference steps. This can
            stabilize the sampling of DPMSolver for steps < 15, especially for steps <= 10.
        euler_at_final (`bool`, defaults to `False`):
            Whether to use Euler's method in the final step. It is a trade-off between numerical stability and detail
            richness. This can stabilize the sampling of the SDE variant of DPMSolver for small number of inference
            steps, but sometimes may result in blurring.
        final_sigmas_type (`str`, defaults to `"zero"`):
            The final `sigma` value for the noise schedule during the sampling process. If `"sigma_min"`, the final
            sigma is the same as the last sigma in the training schedule. If `zero`, the final sigma is set to 0.
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        sigma_min: float = 0.3,
        sigma_max: float = 500,
        sigma_data: float = 1.0,
        sigma_schedule: str = "exponential",
        num_train_timesteps: int = 1000,
        solver_order: int = 2,
        prediction_type: str = "v_prediction",
        rho: float = 7.0,
        solver_type: str = "midpoint",
        lower_order_final: bool = True,
        euler_at_final: bool = False,
        final_sigmas_type: Optional[str] = "zero",  # "zero", "sigma_min"
    ):
        # debug：强制修改final_sigmas_type为sigma_min,solver_order=1
        # final_sigmas_type = "sigma_min"
        # solver_order = 1
        # log
        print("using cosine_inverse scheduler","final_sigmas_type:",final_sigmas_type,"\nsolver_order:",solver_order)
        
        if solver_type not in ["midpoint", "heun"]:
            if solver_type in ["logrho", "bh1", "bh2"]:
                self.register_to_config(solver_type="midpoint")
            else:
                raise NotImplementedError(f"{solver_type} is not implemented for {self.__class__}")

        ramp = torch.linspace(0, 1, num_train_timesteps)
        if sigma_schedule == "karras":
            sigmas = self._compute_karras_sigmas(ramp)
        elif sigma_schedule == "exponential":
            sigmas = self._compute_exponential_sigmas(ramp)
            
        # 反转初始的sigmas
        sigmas = sigmas.flip(0)

        # 用于 conditioning 也反转
        self.timesteps = self.precondition_noise(sigmas)
        self.timesteps = self.timesteps.flip(0)  # 注意 precondition 之后也要 flip

        # add sigma last
        if final_sigmas_type == "sigma_min":
            sigma_last = sigma_min
        elif final_sigmas_type == "zero":
            sigma_last = 0
        else:
            raise ValueError(f"`final_sigmas_type` must be one of 'zero', or 'sigma_min', but got {final_sigmas_type}")

        # self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
        # 注意此时 sigmas 是从小到大，前面插入的是最小 sigma
        self.sigmas = torch.cat([
            torch.tensor([sigma_last], dtype=torch.float32, device=sigmas.device),
            sigmas
        ])
        # print("sigmas", self.sigmas)

        # setable values
        self.num_inference_steps = None
        self.model_outputs = [None] * solver_order
        self.lower_order_nums = 0
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication

    @property
    def init_noise_sigma(self):
        # standard deviation of the initial noise distribution
        return (self.config.sigma_max**2 + 1) ** 0.5

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    # Copied from diffusers.schedulers.scheduling_edm_euler.EDMEulerScheduler.precondition_inputs
    def precondition_inputs(self, sample, sigma):
        c_in = self._get_conditioning_c_in(sigma)
        scaled_sample = sample * c_in
        return scaled_sample

    def precondition_noise(self, sigma):
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor([sigma])

        return sigma.atan() / math.pi * 2

    # Copied from diffusers.schedulers.scheduling_edm_euler.EDMEulerScheduler.precondition_outputs
    def precondition_outputs(self, sample, model_output, sigma):
        sigma_data = self.config.sigma_data
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)

        if self.config.prediction_type == "epsilon":
            c_out = sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
        elif self.config.prediction_type == "v_prediction":
            c_out = -sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
        else:
            raise ValueError(f"Prediction type {self.config.prediction_type} is not supported.")

        denoised = c_skip * sample + c_out * model_output

        return denoised

    # Copied from diffusers.schedulers.scheduling_edm_euler.EDMEulerScheduler.scale_model_input
    def scale_model_input(self, sample: torch.Tensor, timestep: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep. Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the Euler algorithm.

        Args:
            sample (`torch.Tensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.Tensor`:
                A scaled input sample.
        """
        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]
        sample = self.precondition_inputs(sample, sigma)

        self.is_scale_input_called = True
        return sample

    def set_timesteps(self, num_inference_steps: int = None, device: Union[str, torch.device] = None):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """

        self.num_inference_steps = num_inference_steps

        ramp = torch.linspace(0, 1, self.num_inference_steps)
        if self.config.sigma_schedule == "karras":
            sigmas = self._compute_karras_sigmas(ramp)
        elif self.config.sigma_schedule == "exponential":
            sigmas = self._compute_exponential_sigmas(ramp)

        sigmas = sigmas.to(dtype=torch.float32, device=device)
        self.timesteps = self.precondition_noise(sigmas)
        
        # ✅ inversion
        sigmas = sigmas.flip(0)  # ← 反向走 diffusion，即从小 sigma 到大 sigma
        self.timesteps = self.timesteps.flip(0)  # timesteps 也要反向
        
        if self.config.final_sigmas_type == "sigma_min":
            sigma_last = self.config.sigma_min
        elif self.config.final_sigmas_type == "zero":
            sigma_last = 0
        else:
            raise ValueError(
                f"`final_sigmas_type` must be one of 'zero', or 'sigma_min', but got {self.config.final_sigmas_type}"
            )

        self.sigmas = torch.cat([torch.tensor([sigma_last],dtype=torch.float32, device=device), sigmas])

        self.model_outputs = [
            None,
        ] * self.config.solver_order
        self.lower_order_nums = 0

        # add an index counter for schedulers that allow duplicated timesteps
        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication

        # if a noise sampler is used, reinitialise it
        self.noise_sampler = None

    # Copied from diffusers.schedulers.scheduling_edm_euler.EDMEulerScheduler._compute_karras_sigmas
    def _compute_karras_sigmas(self, ramp, sigma_min=None, sigma_max=None) -> torch.Tensor:
        """Constructs the noise schedule of Karras et al. (2022)."""
        sigma_min = sigma_min or self.config.sigma_min
        sigma_max = sigma_max or self.config.sigma_max

        rho = self.config.rho
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    # Copied from diffusers.schedulers.scheduling_edm_euler.EDMEulerScheduler._compute_exponential_sigmas
    def _compute_exponential_sigmas(self, ramp, sigma_min=None, sigma_max=None) -> torch.Tensor:
        """Implementation closely follows k-diffusion.

        https://github.com/crowsonkb/k-diffusion/blob/6ab5146d4a5ef63901326489f31f1d8e7dd36b48/k_diffusion/sampling.py#L26
        """
        sigma_min = sigma_min or self.config.sigma_min
        sigma_max = sigma_max or self.config.sigma_max
        sigmas = torch.linspace(math.log(sigma_min), math.log(sigma_max), len(ramp)).exp().flip(0)
        return sigmas

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._sigma_to_t
    def _sigma_to_t(self, sigma, log_sigmas):
        # get log sigma
        log_sigma = np.log(np.maximum(sigma, 1e-10))

        # get distribution
        dists = log_sigma - log_sigmas[:, np.newaxis]

        # get sigmas range
        low_idx = np.cumsum((dists >= 0), axis=0).argmax(axis=0).clip(max=log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1

        low = log_sigmas[low_idx]
        high = log_sigmas[high_idx]

        # interpolate sigmas
        w = (low - log_sigma) / (low - high)
        w = np.clip(w, 0, 1)

        # transform interpolation to time range
        t = (1 - w) * low_idx + w * high_idx
        t = t.reshape(sigma.shape)
        return t

    def _sigma_to_alpha_sigma_t(self, sigma):
        alpha_t = torch.tensor(1)  # Inputs are pre-scaled before going into unet, so alpha_t = 1
        sigma_t = sigma

        return alpha_t, sigma_t

    def convert_model_output(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Convert the model output to the corresponding type the DPMSolver/DPMSolver++ algorithm needs. DPM-Solver is
        designed to discretize an integral of the noise prediction model, and DPM-Solver++ is designed to discretize an
        integral of the data prediction model.

        <Tip>

        The algorithm and model type are decoupled. You can use either DPMSolver or DPMSolver++ for both noise
        prediction and data prediction models.

        </Tip>

        Args:
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `torch.Tensor`:
                The converted model output.
        """
        sigma = self.sigmas[self.step_index]
        x0_pred = self.precondition_outputs(sample, model_output, sigma)

        return x0_pred

    def dpm_solver_first_order_update(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        One step for the first-order DPMSolver (equivalent to DDIM).

        Args:
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `torch.Tensor`:
                The sample tensor at the previous timestep.
        """
        print("[scheduler/Inverse]step_index: ",self.step_index)
        sigma_t, sigma_s = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
        if self.step_index == 0:
            sigma_s = torch.tensor(self.sigma_min, device=model_output.device, dtype=model_output.dtype) # step_index = 0 时，会报错 or 用self.sigma_min
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s, sigma_s = self._sigma_to_alpha_sigma_t(sigma_s)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s = torch.log(alpha_s) - torch.log(sigma_s)
    
        h = lambda_t - lambda_s
        
        # debug log
        # print(f"[1阶]sigma_t: {sigma_t}, sigma_s: {sigma_s}, h: {h}") # 打印到sigma_s = 0

        # sde-dpmsolver++
        # assert noise is not None
        # x_t = (
        #     (sigma_t / sigma_s * torch.exp(-h)) * sample
        #     + (alpha_t * (1 - torch.exp(-2.0 * h))) * model_output
        #     # + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
        # )
        # dpmsolver++
        x_t = (sigma_t / sigma_s) * sample - (alpha_t * (torch.exp(-h) - 1.0)) * model_output

        return x_t

    def multistep_dpm_solver_second_order_update(
        self,
        model_output_list: List[torch.Tensor],
        sample: torch.Tensor = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        One step for the second-order multistep DPMSolver.

        Args:
            model_output_list (`List[torch.Tensor]`):
                The direct outputs from learned diffusion model at current and latter timesteps.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `torch.Tensor`:
                The sample tensor at the previous timestep.
        二阶多步 DPMSolver 更新公式（反演模式）。

        根据当前和前几个时间步的模型输出，计算上一个时间步的 latent(sample)。

        Args:
            model_output_list (`List[torch.Tensor]`):
                当前及前一时间步模型输出的列表 [m_{t}, m_{t-1}, ...]
            sample (`torch.Tensor`):
                当前 latent 张量
            noise (`torch.Tensor`, optional):
                反演时可选噪声，通常为 None

        Returns:
            `torch.Tensor`: 上一个时间步的 latent
        """
        print("[scheduler/Inverse]step_index: ",self.step_index)
        # 1️⃣ 获取当前 sigma 及前两步 sigma
        sigma_t, sigma_s0, sigma_s1 = (
            self.sigmas[self.step_index + 1],
            self.sigmas[self.step_index],
            self.sigmas[self.step_index - 1],
        )
        # 2️⃣ 将 sigma 转换为 alpha_t, sigma_t 形式
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)

        # 3️⃣ 计算 lambda = log(alpha) - log(sigma)
        #    用于 DPMSolver 公式中的积分近似
        # print("alpha_t",alpha_t)
        # print("sigma_t",sigma_t)
        # print("sigma_s0",sigma_s0)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)

        # 4️⃣ 获取模型输出 m0(当前), m1(上一时间步)
        m0, m1 = model_output_list[-1], model_output_list[-2]

        # 5️⃣ 计算时间步差
        h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
        r0 = h_0 / h # 二阶公式的比值系数
        # 6️⃣ 构造二阶微分项
        D0, D1 = m0, (1.0 / r0) * (m0 - m1)
        # print("h:", h)

        # 7️⃣ 根据 solver_type 选择二阶公式
        #    这里是反演公式，将当前 latent 更新到上一个时间步
        # sde-dpmsolver++
        # assert noise is not None
        if self.config.solver_type == "midpoint":
            x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                    - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
            )
            # x_t = (
            #     (sigma_t / sigma_s0 * torch.exp(-h)) * sample # 原 latent 缩放
            #     + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0 # 一阶项
            #     + 0.5 * (alpha_t * (1 - torch.exp(-2.0 * h))) * D1 # 二阶修正项
            #     # + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise # sde噪声项
            # )
        elif self.config.solver_type == "heun":
            x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                    + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1
                )
            # x_t = (
            #     (sigma_t / sigma_s0 * torch.exp(-h)) * sample
            #     + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
            #     + (alpha_t * ((1.0 - torch.exp(-2.0 * h)) / (-2.0 * h) + 1.0)) * D1
            #     # + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
            # )

        return x_t

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.index_for_timestep
    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        index_candidates = (schedule_timesteps == timestep).nonzero()

        if len(index_candidates) == 0:
            step_index = len(self.timesteps) - 1
        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        elif len(index_candidates) > 1:
            step_index = index_candidates[1].item()
        else:
            step_index = index_candidates[0].item()

        return step_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler._init_step_index
    def _init_step_index(self, timestep):
        """
        Initialize the step_index counter for the scheduler.
        """

        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        generator=None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the sample with
        the multistep DPMSolver.

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`int`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_utils.SchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        反演一步 DPM-Solver / DDIM 的核心计算。
        输入当前 latent(sample) 和模型输出(model_output)，计算前一个时间步的 latent(prev_sample)。

        Args:
            model_output (`torch.Tensor`): 模型输出的 noise / data 预测
            timestep (`int`): 当前时间步
            sample (`torch.Tensor`): 当前 latent
            generator (`torch.Generator`, optional): 随机生成器，用于采样噪声
            return_dict (`bool`): 是否返回 SchedulerOutput

        Returns:
            SchedulerOutput 或 tuple: 前一个时间步的 latent
        """
        # 1️⃣ 校验是否设置推理步数
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
            
        # 2️⃣ 初始化 step_index
        if self.step_index is None:
            self._init_step_index(timestep)

        # 3️⃣ 判断是否需要使用低阶公式增强数值稳定性
        # Improve numerical stability for small number of steps
        lower_order_final = (self.step_index == len(self.timesteps) - 1) and (
            self.config.euler_at_final
            or (self.config.lower_order_final and len(self.timesteps) < 15)
            or self.config.final_sigmas_type == "zero"
        )
        lower_order_second = (
            (self.step_index == len(self.timesteps) - 2) and self.config.lower_order_final and len(self.timesteps) < 15
        )
        # 4️⃣ 转换模型输出为 DPMSolver 需要的格式（噪声/数据预测）
        model_output = self.convert_model_output(model_output, sample=sample)
        
        # 5️⃣ 更新多步历史 model_outputs，用于二阶/三阶公式
        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output
        
        # if self.noise_sampler is None:
        #     seed = None
        #     if generator is not None:
        #         seed = (
        #             [g.initial_seed() for g in generator] if isinstance(generator, list) else generator.initial_seed()
        #         )
        #     self.noise_sampler = BrownianTreeNoiseSampler(
        #         model_output, sigma_min=self.config.sigma_min, sigma_max=self.config.sigma_max, seed=seed
        #     )
        # noise = self.noise_sampler(self.sigmas[self.step_index], self.sigmas[self.step_index + 1]).to(
        #     model_output.device
        # )
        noise = None
        
        # Print diagnostics before update【log】日志
        # print(f"[Step {self.step_index}] timestep: {timestep}")
        # print(f"[Step {self.step_index}] sample before update min/max: {sample.min().item()}/{sample.max().item()}")
        # print(f"[Step {self.step_index}] model_output min/max: {model_output.min().item()}/{model_output.max().item()}")

        # 8️⃣ 核心更新逻辑
        # 根据 solver_order 选择一阶或二阶公式
        if self.config.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
            prev_sample = self.dpm_solver_first_order_update(model_output, sample=sample, noise=noise)
        elif self.config.solver_order == 2 or self.lower_order_nums < 2 or lower_order_second:
            prev_sample = self.multistep_dpm_solver_second_order_update(self.model_outputs, sample=sample, noise=noise)

        # Print diagnostics after update
        # if prev_sample is not None:
        #     print(f"[Step {self.step_index}] sample after update min/max: {prev_sample.min().item()}/{prev_sample.max().item()}")

        # 1️⃣0️⃣ 更新低阶公式计数
        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1
        # 1️⃣1️⃣ 增加 step_index    
        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

    
    def step_with_index(self, model_output, timestep, sample, step_index, **kwargs):
        """
        类似 scheduler.step()，但不改变 self._step_index
        """
        # 安全索引，防止越界
        idx = min(step_index, len(self.sigmas) - 2)
        sigma_t = self.sigmas[idx]
        sigma_next = self.sigmas[idx + 1]

        # 基础 DPM-Solver 一阶更新公式（可根据你原 step 逻辑改）
        prev_sample = sample - model_output * (sigma_t - sigma_next)

        # 去噪，如果有 noise_sampler
        if hasattr(self, "noise_sampler") and self.noise_sampler is not None:
            noise = self.noise_sampler(sigma_t, sigma_next).to(sample.device)
            prev_sample = prev_sample + noise

        return prev_sample


    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler.add_noise
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps does not support float64
            schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)

        # self.begin_index is None when scheduler is used for training, or pipeline does not implement set_begin_index
        if self.begin_index is None:
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        elif self.step_index is not None:
            # add_noise is called after first denoising step (for inpainting)
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            # add noise is called before first denoising step to create initial latent(img2img)
            step_indices = [self.begin_index] * timesteps.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        noisy_samples = original_samples + noise * sigma
        return noisy_samples

    # Copied from diffusers.schedulers.scheduling_edm_euler.EDMEulerScheduler._get_conditioning_c_in
    def _get_conditioning_c_in(self, sigma):
        c_in = 1 / ((sigma**2 + self.config.sigma_data**2) ** 0.5)
        return c_in

    def __len__(self):
        return self.config.num_train_timesteps
