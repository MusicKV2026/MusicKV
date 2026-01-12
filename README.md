# MusicKV: Training-Free Music Editing via Vital Layer Selection and Time-Varying Key-Value Injection


> Abstract: Text-to-music generation is shifting from UNet architectures to Diffusion Transformers (DiTs), but training-free editing remains challenging because homoge neous DiT blocks lack the bottlenecks that enable feature injection in UNets. In this work, we study the internal mechanisms of music DiTs under the premise that not all layers and diffusion timesteps contribute equally to structural coherence. Through layer-wise probing, we show that a sparse subset of self-attention layers acts as the primary carrier of musical form. Leveraging this insight, we propose MusicKV, a training-free editing framework for DiTs with three components: (1) vital layer selection, which identifies a compact set of structure-critical self-attention layers; (2) key–value (KV) injection, which strengthens edits while better preserving content; and (3) a time varying injection schedule, which adapts injection strength over diffusion steps to match the coarse-to-fine generative process. Experiments show that MusicKV effectively de couples content from editing attributes and significantly outperforms baselines in the trade-off between fidelity and editability. 

<p>
    <img src="docs/workflow.png" width="800px"/>  
    <br/>
     Pipeline of the proposed MusicKV framework.</p>






# Demo page

https://musickv2026.github.io/MusicKV2026.




# Installation

Install the conda virtual environment:
```bash
conda env create -f environment.yml
conda activate musickv
```




# Usage

## Generated Audios Editing

You need to provide a list of prompts, where the first prompt describes the original audio, and the second prompt describe the edited audio. For example:

```python
[
    "Piano solo",
    "Acoustic guitar, country, soft",
]
```

Then, you can generate a batch of the image and its edited versions using:
```bash
python run_stable_audio.py \
--hf_token YOUR_PERSONAL_HUGGINGFACE_TOKEN \
--prompts "Piano solo" \
"Acoustic guitar, country, soft" \
"Acoustic guitar, country, soft"
```
where `YOUR_PERSONAL_HUGGINGFACE_TOKEN` is your [personal HuggingFace user access token](https://huggingface.co/docs/hub/en/security-tokens).

Then, the results will be saved to `outputs/, or any other path you specify under `--output_path`.

When using the command with multiple prompts, the interpretation is as follows: the first prompt generates the **original audio**, the second prompt generates the **edited audio** using our method (e.g., applying edits to the original audio based on the prompt), and any subsequent prompts (third, fourth, etc.) generate new audio based solely on the given prompt without any editing applied.



## Prompt structure

Since MusicKV is based on Stable Audio Open, which does not process free-form natural language instructions optimally, you will achieve the best results by ordering your description in a logical sequence. A good prompt structure includes: **Core style and genre, key musical elements, mood and emotion, specific details, BPM, and additional instructions**. The order of these elements matters for the model's understanding.





## Real Audio Editing

If you want to edit a real audio, you still need to provide a list of prompts, where the first prompt describes the input audio and the rest of the prompts describe the edited audio. For example:

```bash
python run_stable_flow.py \
--hf_token YOUR_PERSONAL_HUGGINGFACE_TOKEN \
--input_audio_path inputs/clean-indie-guitar-sample-456143.mp3 \
--prompts "clean inide guitar" \
"classical piano solo, excited"
```
where `YOUR_PERSONAL_HUGGINGFACE_TOKEN` is your [personal HuggingFace user access token](https://huggingface.co/docs/hub/en/security-tokens).

Then, the results will be saved to `outputs/, or any other path you specify under `--output_path`.
