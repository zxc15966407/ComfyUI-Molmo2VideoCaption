# Qwen3-VL Video Caption Node for ComfyUI

![GitHub](https://img.shields.io/github/license/zouruncom/ComfyUI-Molmo2VideoCaption)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)

一个基于 Qwen3-VL 视觉语言模型的 ComfyUI 视频描述生成节点，支持高质量视频内容理解和描述生成。

---

## 🌟 功能特性

### Features
- ✅ **视频帧理解**: 基于 Qwen3-VL 模型理解视频内容
- ✅ **多模型支持**: 支持 Qwen3-VL-4B 和 Qwen3-VL-8B 模型
- ✅ **量化支持**: 支持 4bit/8bit 量化，大幅降低显存占用
- ✅ **性能优化**: 支持 SDPA 和 Flash Attention 加速
- ✅ **智能缓存**: 自动缓存模型，避免重复加载
- ✅ **详细日志**: 完整的运行日志，便于调试和性能分析

---

## 📦 安装

### Installation

1. **克隆仓库到 custom_nodes 目录**:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/zouruncom/ComfyUI-Molmo2VideoCaption.git
```

2. **安装依赖**:
```bash
cd ComfyUI-Molmo2VideoCaption
pip install -r requirements.txt
```

3. **下载模型**:
- 模型将在首次运行时自动下载
- 或手动下载到 `ComfyUI/models/LLM/Qwen-VL/` 目录

---

## ⚙️ 节点参数

### Node Parameters

| 参数 / Parameter | 类型 / Type | 默认值 / Default | 说明 / Description |
|------------------|-------------|------------------|--------------------|
| **frames** | IMAGE | - | 输入视频帧序列 / Input video frames |
| **prompt** | STRING | 详细描述视频内容... | 提示词 / Prompt |
| **model** | SELECT | Qwen3-VL-4B-Instruct | 模型选择 / Model selection |
| **quantization** | SELECT | none | 量化方式 / Quantization |
| **max_frames** | INT | 64 | 最大处理帧数 / Max frames |
| **sample_fps** | INT | 2 | 帧采样频率 / Frame sampling rate |
| **max_new_tokens** | INT | 512 | 最大生成长度 / Max output tokens |
| **temperature** | FLOAT | 0.7 | 随机性参数 / Temperature |
| **keep_model_loaded** | BOOLEAN | false | 保持模型加载 / Keep model loaded |
| **attention** | SELECT | eager | 注意力机制 / Attention mechanism |

---

## 🚀 推荐配置

### Recommended Settings

#### 速度优先 / Speed Priority
```
model: Qwen3-VL-4B-Instruct
quantization: 4bit
attention: sdpa
max_frames: 16
max_new_tokens: 256
temperature: 0.0
keep_model_loaded: true
```

#### 质量优先 / Quality Priority
```
model: Qwen3-VL-8B-Instruct
quantization: none
attention: flash_attention_2
max_frames: 64
max_new_tokens: 1024
temperature: 0.7
keep_model_loaded: true
```

---

## 📊 性能对比

### Performance Comparison

| 配置 / Configuration | 显存占用 / VRAM | 推理速度 / Speed |
|---------------------|----------------|------------------|
| Qwen3-VL-4B (无量化) | ~8GB | 基准 / Baseline |
| Qwen3-VL-4B (8bit) | ~4GB | 快 / Fast |
| Qwen3-VL-4B (4bit) | ~2.5GB | 更快 / Faster |
| Qwen3-VL-4B (4bit + SDPA) | ~2.5GB | 最快 / Fastest |

---

## 📁 项目结构

### Project Structure

```
ComfyUI-Molmo2VideoCaption/
├── molmo2_video_caption.py    # 主节点实现 / Main node implementation
├── video_caption_gguf.py      # GGUF 版本 / GGUF version
├── __init__.py                # 节点注册 / Node registration
├── requirements.txt           # 依赖列表 / Dependencies
├── video_caption.log          # 运行日志 / Runtime logs
└── README.md                  # 项目说明 / Documentation
```

---

## 🛠️ 技术栈

### Tech Stack

- **框架**: ComfyUI
- **模型**: Qwen3-VL (HuggingFace)
- **量化**: BitsAndBytes
- **加速**: PyTorch 2.0 SDPA / Flash Attention
- **语言**: Python 3.10+

---

## 🔧 故障排除

### Troubleshooting

**Q: 模型加载失败？**
A: 确保模型路径正确，或等待首次运行时自动下载

**Q: 速度很慢？**
A: 使用 4bit 量化 + SDPA 注意力机制

**Q: 显存不足？**
A: 使用 4bit 量化，并减少 max_frames

**Q: 日志在哪里？**
A: 日志保存在 `video_caption.log` 文件中

---

## 📄 许可证

### License

MIT License

---

## 🤝 贡献

### Contributing

欢迎提交 Issue 和 Pull Request！

---

## 📧 联系方式

### Contact

如有问题或建议，请提交 GitHub Issue。

---

# Qwen3-VL 视频描述节点 (中文)

一个基于 Qwen3-VL 视觉语言模型的 ComfyUI 自定义节点，用于生成视频内容的文字描述。

## 功能特点

- **视频理解**: 利用 Qwen3-VL 模型分析视频帧内容
- **多模型支持**: 支持 4B 和 8B 参数规模的模型
- **量化优化**: 支持 4bit/8bit 量化，显著降低显存需求
- **性能加速**: 支持 SDPA 和 Flash Attention 两种高效注意力机制
- **智能缓存**: 自动缓存已加载的模型，提升重复使用效率
- **日志记录**: 详细的运行日志，便于性能分析和问题排查

## 安装步骤

1. 克隆仓库到 custom_nodes 目录
2. 安装依赖包
3. 首次运行时自动下载模型

## 推荐配置

### 速度优先
- 模型: Qwen3-VL-4B-Instruct
- 量化: 4bit
- 注意力: sdpa
- 最大帧数: 16
- 保持模型加载: 开启

### 质量优先
- 模型: Qwen3-VL-8B-Instruct
- 量化: 无
- 注意力: flash_attention_2
- 最大帧数: 64
- 保持模型加载: 开启

## 性能参考

| 配置 | 显存占用 | 速度 |
|------|----------|------|
| 4B 无量化 | ~8GB | 基准 |
| 4B 8bit | ~4GB | 快 |
| 4B 4bit | ~2.5GB | 更快 |
| 4B 4bit + SDPA | ~2.5GB | 最快 |

---

*Built with ❤️ for ComfyUI*