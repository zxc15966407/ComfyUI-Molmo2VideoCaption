import torch
import numpy as np
import os
import time
import logging
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from typing import Tuple
import folder_paths

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_caption.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

model_cache = {}
processor_cache = {}


class VideoCaptionNode:
    """ComfyUI视频反推文字描述节点 - 基于Qwen3-VL-4B"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "详细描述这个视频的内容，包括主体、动作、场景、镜头运动等所有细节，尽可能详细。",
                    "multiline": True
                }),
                "model": (["Qwen3-VL-4B-Instruct", "Qwen3-VL-8B-Instruct"], {
                    "default": "Qwen3-VL-4B-Instruct"
                }),
                "quantization": (["none", "4bit", "8bit"], {
                    "default": "none"
                }),
                "max_frames": ("INT", {
                    "default": 64,
                    "min": 1,
                    "max": 256
                }),
                "sample_fps": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10
                }),
                "max_new_tokens": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": False
                }),
                "attention": (["eager", "sdpa", "flash_attention_2"], {
                    "default": "eager"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "generate_caption"
    CATEGORY = "Video/Caption"

    def __init__(self):
        self.model = None
        self.processor = None
        self.current_model = None
        self.current_quantization = None
        self.current_attention = None
        logger.info("VideoCaptionNode initialized")

    def load_model(self, model_name: str, quantization: str, attention: str):
        """加载Qwen3-VL模型"""
        global model_cache, processor_cache
        
        cache_key = f"{model_name}_{quantization}_{attention}"
        logger.info(f"Loading model: {model_name}, quantization: {quantization}, attention: {attention}")
        
        if cache_key in model_cache and cache_key in processor_cache:
            logger.info(f"Model found in cache, skipping reload")
            return model_cache[cache_key], processor_cache[cache_key]

        start_time = time.time()
        
        model_path = self._get_model_path(model_name)
        logger.info(f"Model path: {model_path}")
        
        if not os.path.exists(model_path):
            logger.warning(f"Model path not found: {model_path}")
            logger.info(f"Trying to download model from HuggingFace: qwen/{model_name}")
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id=f"qwen/{model_name}",
                    local_dir=model_path,
                    local_dir_use_symlinks=False
                )
                logger.info(f"Model downloaded successfully to {model_path}")
            except Exception as e:
                logger.error(f"Failed to download model: {str(e)}")
                raise
        
        logger.info("Loading processor...")
        processor = AutoProcessor.from_pretrained(model_path)
        logger.info(f"Processor loaded in {time.time() - start_time:.2f}s")

        logger.info("Loading model...")
        load_start = time.time()
        
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            logger.info("Using 4-bit quantization")
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            logger.info("Using 8-bit quantization")
        else:
            quantization_config = None
            logger.info("No quantization, using full precision")

        available_attentions = []
        try:
            import flash_attn
            available_attentions.append("flash_attention_2")
            logger.info("Flash Attention 2 is available")
        except ImportError:
            logger.info("Flash Attention 2 is not available (install with: pip install flash-attn)")
        
        available_attentions.append("sdpa")
        available_attentions.append("eager")
        
        logger.info(f"Available attention implementations: {available_attentions}")
        
        if attention not in available_attentions and attention != "flash_attention_2":
            logger.warning(f"Attention '{attention}' not available, trying to find best alternative")
            if "sdpa" in available_attentions:
                attention = "sdpa"
                logger.info("Using SDPA attention as fallback")
            else:
                attention = "eager"
                logger.info("Using Eager attention as fallback")
        
        model = None
        last_error = None
        
        for attempt_attn in [attention] + [a for a in ["sdpa", "eager"] if a != attention]:
            try:
                logger.info(f"Trying to load with attention implementation: {attempt_attn}")
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
                    device_map="auto",
                    quantization_config=quantization_config,
                    attn_implementation=attempt_attn if torch.cuda.is_available() else "eager",
                    trust_remote_code=True
                )
                logger.info(f"Successfully loaded with {attempt_attn} attention")
                attention = attempt_attn
                break
            except Exception as e:
                last_error = e
                logger.warning(f"Failed to load with {attempt_attn} attention: {e}")
                continue
        
        if model is None:
            logger.error(f"All attention implementations failed. Last error: {last_error}")
            raise last_error

        logger.info(f"Model loaded in {time.time() - load_start:.2f}s")
        
        model_cache[cache_key] = model
        processor_cache[cache_key] = processor
        
        self._log_model_info(model, model_name)
        
        total_time = time.time() - start_time
        logger.info(f"Model loading completed in {total_time:.2f}s")
        
        return model, processor

    def _get_model_path(self, model_name: str) -> str:
        """获取模型路径 - 尝试多个可能的位置"""
        possible_paths = [
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "models", "LLM", "Qwen-VL", model_name
            ),
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "models", "LLM", "Qwen-VL", model_name
            ),
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "models", model_name
            ),
        ]
        
        for model_path in possible_paths:
            if os.path.exists(model_path):
                logger.info(f"Found model at: {model_path}")
                return model_path
        
        default_path = possible_paths[0]
        logger.info(f"Model not found in any path, using default: {default_path}")
        return default_path

    def _log_model_info(self, model, model_name: str):
        """记录模型信息"""
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            logger.info(f"GPU Memory Used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            logger.info(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        
        logger.info(f"Model: {model_name}")
        logger.info(f"Model dtype: {model.dtype}")
        logger.info(f"Model device: {model.device}")
        
        try:
            param_count = sum(p.numel() for p in model.parameters())
            logger.info(f"Model parameters: {param_count / 1e9:.2f}B")
        except Exception as e:
            logger.debug(f"Failed to count parameters: {e}")

    def sample_frames(self, frames, max_frames: int = 64):
        """均匀采样视频帧"""
        start_time = time.time()
        total_frames = frames.shape[0]
        num_frames = min(total_frames, max_frames)
        
        logger.info(f"Sampling {num_frames} frames from total {total_frames} frames")
        
        if total_frames <= max_frames:
            indices = np.arange(total_frames)
        else:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        sampled = []
        for i, idx in enumerate(indices):
            frame = frames[idx]
            if hasattr(frame, 'cpu'):
                frame_np = (frame * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)
            else:
                frame_np = (np.clip(frame * 255, 0, 255)).astype(np.uint8)
            
            if frame_np.shape[2] == 4:
                pil = Image.fromarray(frame_np, 'RGBA').convert('RGB')
            else:
                pil = Image.fromarray(frame_np, 'RGB')
            sampled.append(pil)

        logger.info(f"Frame sampling completed in {time.time() - start_time:.2f}s")
        return sampled

    def generate_caption(self, frames, prompt: str, model: str,
                         quantization: str, attention: str,
                         max_frames: int, sample_fps: int,
                         max_new_tokens: int, temperature: float,
                         keep_model_loaded: bool) -> Tuple[str]:
        """生成视频描述"""
        start_time = time.time()
        logger.info("=" * 50)
        logger.info(f"Starting video caption generation")
        logger.info(f"Input frames shape: {frames.shape}")
        logger.info(f"Prompt length: {len(prompt)} characters")
        logger.info(f"Parameters: model={model}, quantization={quantization}, attention={attention}")
        logger.info(f"max_frames={max_frames}, sample_fps={sample_fps}")
        logger.info(f"max_new_tokens={max_new_tokens}, temperature={temperature}")
        logger.info(f"keep_model_loaded={keep_model_loaded}")

        try:
            load_start = time.time()
            model_obj, processor = self.load_model(model, quantization, attention)
            logger.info(f"Model loaded in {time.time() - load_start:.2f}s")

            frame_start = time.time()
            pil_frames = self.sample_frames(frames, max_frames)
            logger.info(f"Frame processing completed in {time.time() - frame_start:.2f}s")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": pil_frames, "fps": float(sample_fps)},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            preprocess_start = time.time()
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt"
            )
            inputs = inputs.to(model_obj.device)
            logger.info(f"Preprocessing completed in {time.time() - preprocess_start:.2f}s")

            inference_start = time.time()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            with torch.no_grad():
                logger.info(f"Starting inference on device: {model_obj.device}")
                logger.info(f"Input tensor shape: {inputs.input_ids.shape}")
                image_count = len(image_inputs) if image_inputs is not None else 0
                video_count = len(video_inputs) if video_inputs is not None else 0
                logger.info(f"Vision inputs count: images={image_count}, videos={video_count}")
                
                start_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                
                if start_event:
                    start_event.record()
                
                generated_ids = model_obj.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    use_cache=True,
                )
                
                if end_event:
                    end_event.record()
                    torch.cuda.synchronize()
            
            if start_event and end_event:
                gpu_time_ms = start_event.elapsed_time(end_event)
                logger.info(f"GPU inference time: {gpu_time_ms:.2f} ms")
            
            inference_time = time.time() - inference_start
            logger.info(f"Inference completed in {inference_time:.2f}s")
            
            if torch.cuda.is_available():
                logger.info(f"GPU Memory After Inference: Used={torch.cuda.memory_allocated()/1e9:.2f}GB, Cached={torch.cuda.memory_reserved()/1e9:.2f}GB")

            decode_start = time.time()
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            caption = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            logger.info(f"Decoding completed in {time.time() - decode_start:.2f}s")

            if not keep_model_loaded:
                cleanup_start = time.time()
                global model_cache, processor_cache
                cache_key = f"{model}_{quantization}_{attention}"
                if cache_key in model_cache:
                    del model_cache[cache_key]
                    logger.info("Model removed from cache")
                if cache_key in processor_cache:
                    del processor_cache[cache_key]
                    logger.info("Processor removed from cache")
                
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                logger.info(f"Cleanup completed in {time.time() - cleanup_start:.2f}s")

            total_time = time.time() - start_time
            logger.info(f"Total caption generation time: {total_time:.2f}s")
            logger.info(f"Caption length: {len(caption)} characters")
            logger.info(f"Caption preview: {caption[:100]}..." if len(caption) > 100 else f"Caption: {caption}")
            logger.info("=" * 50)

            return (caption,)

        except Exception as e:
            logger.error(f"Error during caption generation: {str(e)}", exc_info=True)
            raise


NODE_CLASS_MAPPINGS = {
    "VideoCaption": VideoCaptionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoCaption": "Qwen3-VL视频反推描述"
}