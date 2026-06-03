"""
ComfyUI 视频反推文字描述节点 - GGUF版本
逐帧推理 + 文本模型整合 = 真正视频理解
基于 llama-mtmd-cli.exe (视觉) + llama-cli.exe (文本)
不与现有 VideoCaption 节点冲突
"""
import numpy as np
import os
import subprocess
import tempfile
import shutil
from PIL import Image
from typing import Tuple, List


class VideoCaptionGGUFNode:
    """ComfyUI视频反推文字描述节点 - GGUF版本 (llama.cpp Vulkan)"""

    LLAMA_MTMD = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "llama-vulkan", "llama-mtmd-cli.exe"
    )
    LLAMA_CLI = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "llama-vulkan", "llama-cli.exe"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "详细描述这个视频",
                    "multiline": True
                }),
                "model_path": ("STRING", {
                    "default": "E:\\ComfyUI-aki-v2\\ComfyUI\\models\\LLM\\Qwen3-VL-4B-Instruct-Unredacted-MAX-GGUF\\Qwen3-VL-4B-Instruct-Unredacted-MAX.Q8_0.gguf",
                    "multiline": False
                }),
                "mmproj_path": ("STRING", {
                    "default": "E:\\ComfyUI-aki-v2\\ComfyUI\\models\\LLM\\Qwen3-VL-4B-Instruct-Unredacted-MAX-GGUF\\Qwen3-VL-4B-Instruct-Unredacted-MAX.mmproj-f16.gguf",
                    "multiline": False
                }),
                "text_model_path": ("STRING", {
                    "default": "E:\\ComfyUI-aki-v2\\ComfyUI\\models\\LLM\\Qwen3.5-9B-Uncensored-HauhauCS-Aggressive\\Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf",
                    "multiline": False
                }),
                "max_frames": ("INT", {
                    "default": 6,
                    "min": 1,
                    "max": 16
                }),
                "max_new_tokens": ("INT", {
                    "default": 256,
                    "min": 64,
                    "max": 2048
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
                "n_gpu_layers": ("INT", {
                    "default": 99,
                    "min": 0,
                    "max": 99,
                }),
                "context_size": ("INT", {
                    "default": 4096,
                    "min": 512,
                    "max": 32768,
                }),
                "threads": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 32,
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "generate_caption"
    CATEGORY = "Video/Caption"

    def sample_frames(self, frames, max_frames: int = 6) -> Tuple[List[str], str]:
        """均匀采样视频帧，保存为独立临时JPG文件"""
        total_frames = frames.shape[0]
        num_frames = min(total_frames, max_frames)
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        tmp_dir = tempfile.mkdtemp(prefix="comfy_video_frames_")
        frame_paths = []

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

            fpath = os.path.join(tmp_dir, f"frame_{i:04d}.jpg")
            pil.save(fpath, "JPEG", quality=90)
            frame_paths.append(fpath)

        return frame_paths, tmp_dir

    def _run_vision_cli(self, image_path: str, prompt: str, model_path: str,
                        mmproj_path: str, max_new_tokens: int, temperature: float,
                        n_gpu_layers: int, context_size: int, threads: int) -> str:
        """调用 llama-mtmd-cli.exe 对单帧进行视觉推理"""
        cmd = [
            self.LLAMA_MTMD,
            "-m", model_path,
            "--mmproj", mmproj_path,
            "-p", prompt,
            "-n", str(max_new_tokens),
            "--temp", str(temperature),
            "-ngl", str(n_gpu_layers),
            "-c", str(context_size),
            "-t", str(threads),
            "--image-min-tokens", "1024",
            "--no-warmup",
            "--image", image_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=180, encoding='utf-8', errors='replace')

        if result.returncode != 0:
            stderr_text = result.stderr
            error_indicators = ['error:', 'Error:', 'failed', 'Failed',
                               'CUDA error', 'out of memory']
            has_real_error = any(ind in stderr_text for ind in error_indicators)
            if has_real_error:
                err_lines = stderr_text.strip().split('\n')
                return f"[ERR: {'; '.join(err_lines[-3:])[:100]}]"
            if not result.stdout.strip():
                return f"[ERR: rc={result.returncode}]"

        return self._clean_output(result.stdout)

    def _run_text_cli(self, prompt: str, model_path: str, max_new_tokens: int,
                      temperature: float, n_gpu_layers: int, context_size: int,
                      threads: int) -> str:
        """调用 llama-cli.exe 进行纯文本推理（汇总帧描述）"""
        cmd = [
            self.LLAMA_CLI,
            "-m", model_path,
            "-p", prompt,
            "-n", str(max_new_tokens),
            "--temp", str(temperature),
            "-ngl", str(n_gpu_layers),
            "-c", str(context_size),
            "-t", str(threads),
            "--no-warmup",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=180, encoding='utf-8', errors='replace')

        if result.returncode != 0:
            return ""  # 汇总失败则返回空，用原始帧描述

        return self._clean_output(result.stdout)

    def _clean_output(self, text: str) -> str:
        """清理 llama.cpp 输出：去日志行和空白"""
        lines = text.strip().split('\n')
        content_lines = [l for l in lines if l.strip()
                         and not l.startswith('llama_')
                         and not l.startswith('load_')
                         and not l.startswith('clip_')
                         and not l.startswith('print_info')
                         and not l.startswith('common_')
                         and not l.startswith('0.')]
        return '\n'.join(content_lines).strip()

    def generate_caption(self, frames, prompt: str, model_path: str,
                         mmproj_path: str, text_model_path: str,
                         max_frames: int, max_new_tokens: int,
                         temperature: float, n_gpu_layers: int,
                         context_size: int, threads: int) -> Tuple[str]:
        """逐帧视觉推理 → 文本模型整合为视频描述"""

        # 1. 采样帧
        frame_paths, tmp_dir = self.sample_frames(frames, max_frames)

        try:
            # 2. 逐帧视觉推理（每帧简述关键内容，供汇总用）
            frame_prompt = "描述画面"
            frame_captions = []

            for i, fp in enumerate(frame_paths):
                caption = self._run_vision_cli(
                    fp, frame_prompt, model_path, mmproj_path,
                    max_new_tokens, temperature, n_gpu_layers,
                    context_size, threads
                )
                frame_captions.append(f"帧{i+1}: {caption}")

            # 如果只有一帧，直接返回
            if len(frame_captions) == 1:
                return (frame_captions[0],)

            # 3. 文本模型整合为连贯视频描述（不要逐帧列举）
            combined = "\n".join(frame_captions)
            summary_prompt = (
                f"以下是一个视频不同时间点的帧描述。请将它们整合成一段完整的视频描述。"
                f"不要逐帧列出，直接输出整体视频内容，突出画面的动态变化和镜头运动。\n\n"
                f"{combined}\n\n"
                f"视频描述："
            )

            summary = self._run_text_cli(
                summary_prompt, text_model_path,
                max_new_tokens=max_new_tokens * 2,
                temperature=temperature,
                n_gpu_layers=n_gpu_layers,
                context_size=context_size,
                threads=threads
            )

            if summary:
                return (summary[:3000],)

            # 如果汇总失败，返回所有帧描述
            return (f"[视频逐帧描述 - 共{len(frame_paths)}帧]\n\n{combined}",)

        finally:
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)


# 注册节点 - 使用不同的key避免与VideoCaption冲突
NODE_CLASS_MAPPINGS = {
    "VideoCaptionGGUF": VideoCaptionGGUFNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoCaptionGGUF": "Qwen3-VL视频描述(GGUF)"
}