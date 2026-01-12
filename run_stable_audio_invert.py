import argparse
import yaml
import os
import torch
import numpy as np
import soundfile as sf
import torchaudio
from diffusers.schedulers.scheduling_cosine_dpmsolver_multistep_inverse import CosineDPMSolverMultistepInverseScheduler

class StableAudioGenerator:

    def __init__(self):
        self._load_config()
        self._parse_args()
        self._load_pipeline()
        self._load_inverse_pipeline()

    def _load_config(self, config_path='configs/layers.yaml'):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.SKIP_LAYERS = config.get('SKIP_LAYERS', [])
        self.VITAL_LAYERS = config.get('VITAL_LAYERS', [])
        self.guidance_scale = config.get('guidance_scale', 7)
        self.USE_TRATIO = config.get('USE_TRATIO', False)

    def _parse_args(self):
        parser = argparse.ArgumentParser(description="Generate audio using StableAudioPipeline")
        
        # Required parameters
        parser.add_argument("--model_path", type=str, default="xxx/Model/stabilityai/stable-audio-open-1.0",
                            help="Hugging Face model repository or local directory")
        parser.add_argument("--hf_token", type=str, default="",
                            help="Hugging Face API token (if needed for private models)")
        parser.add_argument("--prompt", type=str, nargs="+", required=True)
        parser.add_argument("--output_path", type=str, default="outputs/generated_audio.wav",
                            help="Output path for the generated audio")
        parser.add_argument("--input_audio_path", type=str, default=None,
                            help="Path to input audio file for inversion (optional)")
        parser.add_argument("--inverse_type", type=str, default="DDIM")
        
        # Audio duration parameters
        parser.add_argument("--audio_end_in_s", type=float, default=15.0,
                            help="End time of the audio clip in seconds")
        parser.add_argument("--audio_start_in_s", type=float, default=0.0,
                            help="Start time of the audio clip in seconds")
        
        # Generation parameters
        parser.add_argument("--num_inference_steps", type=int, default=100,
                            help="Number of denoising steps")
        parser.add_argument("--negative_prompt", type=str, default="",
                            help="Negative prompt to avoid certain characteristics")
        parser.add_argument("--num_waveforms_per_prompt", type=int, default=1,
                            help="Number of variations to generate per prompt")
        parser.add_argument("--seed", type=int, default=42,
                            help="Random seed for reproducibility")
        parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                            choices=["cpu", "cuda"],
                            help="Device to run the model on")
        
        # Additional options
        parser.add_argument("--verbose", action="store_true",
                            help="Show detailed progress messages")
        
        self.args = parser.parse_args()
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.args.output_path), exist_ok=True)

    def _load_pipeline(self):
        """Load the StableAudioPipeline model"""
        from diffusers import StableAudioPipeline
        print("⏳ Loading audio model...")
        self.pipe = StableAudioPipeline.from_pretrained(
            self.args.model_path,
            torch_dtype=torch.float16,
            token=self.args.hf_token
        )
        # Move to appropriate device
        self.pipe = self.pipe.to(self.args.device)
        
        # Enable memory optimizations
        if self.args.device == "cpu":
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.enable_attention_slicing()
        print("✅ Model loaded successfully!")

    def _load_inverse_pipeline(self):
        print("⏳ Loading inverse audio model...") 
        if self.args.inverse_type == "DDIM":
            from diffusers.pipelines.stable_audio.pipeline_stable_audio_inversion import StableAudioInversionPipeline
        self.inverse_pipeline = StableAudioInversionPipeline.from_pretrained(
                self.args.model_path,
                torch_dtype=torch.float16,
            ).to(self.args.device)
        self.inverse_pipeline.scheduler = CosineDPMSolverMultistepInverseScheduler.from_config(self.inverse_pipeline.scheduler.config)
        print("✅ Model loaded successfully!")

    def unload_inverse_model(self):
        if self.inverse_pipeline is not None:
            del self.inverse_pipeline
            self.inverse_pipeline = None
            torch.cuda.empty_cache()
            gc.collect()

    def invert_audio(self, audio):
        """Invert audio to latent space"""
        audio_tensor = None
        
        if isinstance(audio, str):
            if not os.path.exists(audio):
                raise FileNotFoundError(f"FileNotFoundError:{audio}")
            audio_np, sr = sf.read(audio)
            print(f"{audio}, sr: {sr}")
            if audio_np.ndim == 1:
                audio_np = np.expand_dims(audio_np, axis=0)
            audio = audio_np

        if isinstance(audio, np.ndarray):
            if audio.shape[0] > 2097152:
                audio = audio[:2097152, :]
            print("audio shape:", audio.shape) # (2097152, 2)
            # audio: (samples, channels)
            if audio.ndim == 2:
                audio = audio.T  
            audio_tensor = torch.tensor(audio).unsqueeze(0).half().to(self.args.device)  # (1, channels, samples)

        elif isinstance(audio, torch.Tensor):
            if audio.ndim == 2:  # (channels, samples)
                audio = audio.unsqueeze(0)  # (1, channels, samples)
            elif audio.ndim == 3 and audio.shape[-1] < audio.shape[-2]:
                audio = audio.permute(0, 2, 1)
            audio_tensor = audio.half().to(self.args.device)
        with torch.no_grad():
            latents = self.inverse_pipeline.vae.encode(audio_tensor).latent_dist.mean

        print("[demo]Latents shape after encode:", latents.shape)
        with torch.no_grad():
            test_audio = self.inverse_pipeline.vae.decode(latents).sample
        print("[demo]audio min/max:", test_audio.min().item(), test_audio.max().item())

        if self.args.inverse_type == "DDIM":
            with torch.no_grad():
                inversion_output = self.inverse_pipeline(
                    prompt=self.args.prompt[0],
                    negative_prompt=self.args.negative_prompt,
                    latents=latents,
                    num_inference_steps=self.args.num_inference_steps,
                    guidance_scale=1,
                    return_dict=True,
                    output_type="latent"
                )
        inverted_noise = inversion_output.audios
        return inverted_noise

    def infer_and_save(self):
        """Inference audio and save it as a WAV file"""
        # Setup generator with seed
        generator = torch.Generator(self.args.device).manual_seed(self.args.seed)
        
        if self.args.input_audio_path is None:
            latents = None
        else:
            latents = self.invert_audio(self.args.input_audio_path).tile(len(self.args.prompt), 1, 1)
        
        # Prepare arguments for the pipeline
        call_kwargs = {
            "prompt": self.args.prompt,
            # "audio_end_in_s": self.args.audio_end_in_s,
            # "audio_start_in_s": self.args.audio_start_in_s,
            "num_inference_steps": self.args.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "generator": generator,
            "num_waveforms_per_prompt": self.args.num_waveforms_per_prompt,
            "return_dict": True,
            "latents": latents,
            "skip_blocks": self.SKIP_LAYERS,
            "copy_blocks": self.VITAL_LAYERS,
            "USE_TRATIO": self.USE_TRATIO,
            "invert_audio": self.args.input_audio_path is not None,
        }
        
        if self.args.verbose:
            print(f"📊 Using parameters: steps={self.args.num_inference_steps}, guidance={self.args.guidance_scale}")
        
        # Generate audio using the pipeline
        audio = self.pipe(**call_kwargs).audios
        
        for i, seg in enumerate(audio):
            output = seg.T.float().cpu().numpy()
            filename, ext = os.path.splitext(self.args.output_path)
            save_path = f"{filename}_{i}{ext}" if len(audio) > 1 else self.args.output_path
            sf.write(save_path, output, self.pipe.vae.sampling_rate)
            print(f"🎉 Success! Audio saved to: {save_path}")

if __name__ == "__main__":
    print("🚀 Starting audio generation process...")
    generator = StableAudioGenerator()
    print(generator.args.input_audio_path)
    print("🎶 Running in audio inference mode")
    generator.infer_and_save()
    print("✅ Process completed")