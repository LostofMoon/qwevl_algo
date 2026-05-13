## 变更报告

### 1. Patch 切分算法修改

**文件：** `qwen3vl_improved/modeling_qwen3_vl.py`

#### 修改位置 A：`Qwen3VLVisionPatchEmbed` 类

**改动内容：**

| 修改前 | 修改后 |
|--------|--------|
| 仅有 `proj`（Conv3d）| 新增 `global_proj`（与 `proj` 结构相同的 Conv3d） |
| 每个 32×32 区域 → 4 个 token（4 个 16×16 子块） | 每个 32×32 区域 → **5 个 token**（4 个局部 + 1 个全局） |

**新增方法 `forward_global(hidden_states)`，流程如下：**
1. 将输入 reshape 为 `[N, in_channels, temporal_patch_size, 16, 16]`
2. 按 4 个一组重新分组（每组对应一个 32×32 空间区域）：图像处理器以 block-major 顺序输出 patch，组内排列为 `[左上, 右上, 左下, 右下]`
3. 用 `torch.cat` 拼接重建 32×32 区域：
   - 顶行 = cat(左上, 右上)，底行 = cat(左下, 右下)，再沿高度方向拼接 → `[G, C, T, 32, 32]`
4. **用 `F.interpolate(mode='bilinear', align_corners=False)` 下采样 32×32 → 16×16**
5. 经 `global_proj` 投影，输出 `[G, hidden_size]`（G = N/4）

#### 修改位置 B：`Qwen3VLVisionPatchMerger` 类

新增 `tokens_per_group` 参数（默认为 4，新结构传入 5）：
- `self.hidden_size = config.hidden_size * tokens_per_group`（即 `1024 * 5 = 5120`）
- `linear_fc1` / `linear_fc2` 的输入维度从 4D 扩展为 5D

#### 修改位置 C：`Qwen3VLVisionModel.__init__`

```python
_tokens_per_group = config.spatial_merge_size ** 2 + 1  # 4 + 1 = 5
self.merger = Qwen3VLVisionPatchMerger(..., tokens_per_group=_tokens_per_group)
self.deepstack_merger_list = nn.ModuleList([
    Qwen3VLVisionPatchMerger(..., tokens_per_group=_tokens_per_group)
    ...
])
```

#### 修改位置 D：`Qwen3VLVisionModel.forward()`

对 ViT forward 流程做了全面更新：

1. **生成全局 token**：调用 `patch_embed.forward_global(hidden_states)` → `[N/4, D]`
2. **交错拼接**：将局部×4 与全局×1 拼为一组 `[G, 5, D]`，展平为 `[5N/4, D]`
3. **位置编码**：对每组 4 个局部 token 的位置取均值，作为全局 token 的位置（即 2×2 块的中心）
4. **Rotary PE**：同上，取 4 个局部 rotary 频率的均值
5. **`cu_seqlens` 更新**：每帧 token 数从 `H×W` 变为 `H×W×5//4`

**输出 token 数保持不变**：输入 5N/4 个 token → merger 每组合并 5 个 → 输出 N/4 个，与原设计一致（`T×H×W / spatial_merge_size²`）

---

### 2. 权重形状不匹配的处理

#### 实际发生不匹配的层（基于 Qwen3-VL-4B-Instruct 实测）

| 层 | 旧形状 | 新形状 |
|----|--------|--------|
| `visual.patch_embed.global_proj.*` | 不存在（新增层） | `[1024, 3, 2, 16, 16]` |
| `visual.merger.linear_fc1.weight` | `[4096, 4096]` | `[5120, 5120]` |
| `visual.merger.linear_fc1.bias` | `[4096]` | `[5120]` |
| `visual.merger.linear_fc2.weight` | `[2560, 4096]` | `[2560, 5120]` |
| `visual.deepstack_merger_list.[0-2].norm.weight/bias` | `[4096]` | `[5120]` |
| `visual.deepstack_merger_list.[0-2].linear_fc1.weight/bias` | 同上 | 同上 |
| `visual.deepstack_merger_list.[0-2].linear_fc2.weight` | `[2560, 4096]` | `[2560, 5120]` |

> 注：`visual.merger.norm` 不变（`use_postshuffle_norm=False`，作用于 `hidden_size=1024`，形状不受影响）

#### 权重初始化策略

**实现位置**：`adapt_weights_for_global_token()` 函数（`modeling_qwen3_vl.py` 末尾），在 `run_example.py` 中 `from_pretrained(..., ignore_mismatched_sizes=True)` 之后调用。

---

**`global_proj`（新增 Conv3d）：**
直接复制 `proj` 的预训练权重。初始时全局 token 的嵌入方式与局部 patch 相同，fine-tuning 时模型自行学习差异。

---

**`linear_fc1.weight`：`[4096, 4096]` → `[5120, 5120]`**

```
左上 [4096, 4096]：复制预训练权重（局部→局部通路不变）
右列 [4096, 1024]：全局 token 的输入列 = 4 个局部输入块的均值
                   （初始时全局 token 的贡献 ≈ 4 个局部的平均）
下行 [1024, 4096]：全局 token 的输出行 = 复制第 4 块的行（[3072:4096, :]）
右下 [1024, 1024]：全局→全局 = 复制第 4 块的自身块（[3072:4096, 3072:4096]）
```

**`linear_fc1.bias`：`[4096]` → `[5120]`**
```
前 4096 位：复制预训练 bias
后 1024 位：复制预训练 bias 的第 4 块（[3072:4096]）
```

**`linear_fc2.weight`：`[2560, 4096]` → `[2560, 5120]`**
```
左列 [2560, 4096]：复制预训练权重
右列 [2560, 1024]：全局 token 的输入列 = 复制第 4 块的列（[:, 3072:4096]）
                   （初始时全局 token 对输出的贡献方式与第 4 个局部 token 相同）
```

**`deepstack norm.weight / norm.bias`：`[4096]` → `[5120]`**
```
前 4096 位：复制预训练值
后 1024 位：复制预训练值的第 4 块（[3072:4096]）
```

---

### 3. 修改文件汇总

| 文件 | 修改内容 |
|------|---------|
| `qwen3vl_improved/modeling_qwen3_vl.py` | PatchEmbed / PatchMerger / VisionModel 结构改动 + `adapt_weights_for_global_token()` 函数 |
| `qwen3vl_improved/__init__.py` | 导出 `adapt_weights_for_global_token` |
| `run_example.py` | 添加 `ignore_mismatched_sizes=True` 及 `adapt_weights_for_global_token` 调用 |

### 4. 运行验证

在服务器（GPU 3, 4）上执行 `run_example.py`，全程无报错，模型正常输出文本。输出示例（对图片的描述）：
> "This is a heartwarming, sun-drenched photograph capturing a tender moment between a woman and her dog on a beach at sunset..."