
import torch # optional (need torch to work; pip install torch)
import random, os
import numpy as np
import gradio as gr
from gudio.tool.models.t3 import T3 # need gudio; pip install gudio
from gudio.tool.models.s3tokenizer import S3_SR, drop_invalid_tokens
from gudio.tool.models.s3gen import S3GEN_SR, S3Gen
from gudio.tool.models.tokenizers import EnTokenizer
from gudio.tool.models.voice_encoder import VoiceEncoder
from gudio.tool.models.t3.modules.cond_enc import T3Cond
from gudio import perth

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

from dataclasses import dataclass
from pathlib import Path
import librosa
import torch.nn.functional as F
from safetensors.torch import load_file

def punc_norm(text: str) -> str:
    """
        Quick cleanup func for punctuation from LLMs or
        containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."
    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]
    # Remove multiple space chars
    text = " ".join(text.split())
    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)
    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."
    return text

from .quant3 import convert_gguf_to_safetensors
gguf_files = [file for file in os.listdir() if file.endswith('.gguf')]

if gguf_files:
    print("\nGGUF file(s) available. Select which one for MODEL (t3):")
    for index, file_name in enumerate(gguf_files, start=1):
        print(f"{index}. {file_name}")
    choice2 = input(f"Enter your choice (1 to {len(gguf_files)}): ")
    try:
        choice_index=int(choice2)-1
        selected_t3_file=gguf_files[choice_index]
        # T3
        print(f"t3 file: {selected_t3_file} is selected!")
        t3_path=selected_t3_file
        if gguf_files:
            print("\nGGUF file(s) available. Select which one for CLIP (s3):")
            for index, file_name in enumerate(gguf_files, start=1):
                print(f"{index}. {file_name}")
            choice3 = input(f"Enter your choice (1 to {len(gguf_files)}): ")
            try:
                choice_index=int(choice3)-1
                selected_s3_file=gguf_files[choice_index]
                # S3
                print(f"s3 file: {selected_s3_file} is selected!\n")
                s3_path=selected_s3_file
                # create model folder
                model_folder = './models/gudio/'
                os.makedirs(model_folder, exist_ok=True)
                if DEVICE == "cuda":
                    use_bf16 = True
                    model_path = f"{model_folder}{os.path.splitext(t3_path)[0]}-bf16.safetensors"
                    clip_path = f"{model_folder}{os.path.splitext(s3_path)[0]}-bf16.safetensors"
                else:
                    use_bf16 = False
                    model_path = f"{model_folder}{os.path.splitext(t3_path)[0]}-f32.safetensors"
                    clip_path = f"{model_folder}{os.path.splitext(s3_path)[0]}-f32.safetensors"
                # dequantization process begins
                print(f"Prepare to dequantize T3: {t3_path}")
                convert_gguf_to_safetensors(t3_path, model_path, use_bf16)
                print(f"Prepare to dequantize S3: {s3_path}")
                convert_gguf_to_safetensors(s3_path, clip_path, use_bf16)
            except (ValueError, IndexError) as e:
                print(f"Invalid choice. Please enter a valid number. ({e})")
    except (ValueError, IndexError):
        print("Invalid choice. Please enter a valid number.")
else:
    print("No GGUF files are available in the current directory.")
    input("--- Press ENTER To Exit ---")

@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        arg_dict = dict(
            t3=self.t3.__dict__,
            gen=self.gen
        )
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])

class GudioTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: EnTokenizer,
        device: str,
        conds: Conditionals = None,
    ):
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self.watermarker = perth.PerthImplicitWatermarker

    @classmethod
    def from_local(cls, device) -> 'GudioTTS':
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None

        t3 = T3()
        t3_state = load_file(model_path)
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(
            load_file(clip_path), strict=False
        )
        s3gen.to(device).eval()
        
        ve = VoiceEncoder()
        ve.load_state_dict(
            load_file(os.path.join(os.path.dirname(__file__), "../gudio/decoder/vae.safetensors"))
        )
        ve.to(device).eval()

        tokenizer = EnTokenizer(str(os.path.join(os.path.dirname(__file__), "../gudio/encoder/tokenizer.json")))
        builtin_voice = os.path.join(os.path.dirname(__file__), "../gudio/encoder/codec.pt")
        conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device) -> 'GudioTTS':
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"

        return cls.from_local(device)

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)
        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)
        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)
        # Speech cond prompt tokens
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)
        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)
        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def generate(
        self,
        text,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
    ):
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"
        # Update exaggeration if needed
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)
        # Norm and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)

        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=1000,  # TODO: use the value in config
                temperature=temperature,
                cfg_weight=cfg_weight,
            )
            # Extract only the conditional batch.
            speech_tokens = speech_tokens[0]
            # TODO: output becomes 1D
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens.to(self.device)
            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
        return torch.from_numpy(wav).unsqueeze(0)
      
def load_model():
    model = GudioTTS.from_pretrained(DEVICE)
    return model

def generate(model, text, audio_prompt_path, exaggeration, temperature, seed_num, cfgw):
    if model is None:
        model = GudioTTS.from_pretrained(DEVICE)

    if seed_num != 0:
        set_seed(int(seed_num))

    wav = model.generate(
        text,
        audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        temperature=temperature,
        cfg_weight=cfgw,
    )
    return (model.sr, wav.squeeze(0).numpy())

# gardio
block = gr.Blocks(title="gguf")
with block:
    model_state = gr.State(None)  # Loaded once per session/user

    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                value="Hey Connector, why your appearance looks so stupid? Oh, really? maybe I ate too much smart beans. Wow. Amazing. Let's go to get some more smart beans and you will become stupid as well.",
                label="gudio - text to speech synthesize",
                max_lines=5
            )
            ref_wav = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio File", value=None)
            exaggeration = gr.Slider(0.25, 2, step=.05, label="Exaggeration (Neutral = 0.5, extreme values can be unstable)", value=.5)
            cfg_weight = gr.Slider(0.0, 1, step=.05, label="CFG/Pace", value=0.5)

            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=.05, label="temperature", value=.8)

            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")

    block.load(fn=load_model, inputs=[], outputs=model_state)

    run_btn.click(
        fn=generate,
        inputs=[
            model_state,
            text,
            ref_wav,
            exaggeration,
            temp,
            seed_num,
            cfg_weight,
        ],
        outputs=audio_output,
    )

block.queue(
    max_size=50,
    default_concurrency_limit=1,
).launch()
