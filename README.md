# molbals-stats

Single ComfyUI output node for prompt diagnostics.

## Node

- `molbals-stats`

Connect the `trigger` input to the final value whose generation you want to wait for, such as the final latent or decoded image. The node displays and returns JSON plus separate numeric outputs:

- `peak_vram_mb`
- `peak_ram_mb`
- `total_seconds`

`peak_ram_mb` is the ComfyUI process RSS peak sampled during the prompt. `peak_vram_mb` is sampled from the active ComfyUI torch device; on CUDA it uses `torch.cuda.mem_get_info`, so it reflects device memory in use during the prompt, not only tensor allocations.

The node is an output node and forces fresh execution with `IS_CHANGED`, so it should not replay cached stats from an earlier prompt.

## Example JSON output

```json
{
  "complete": false,
  "device": "NVIDIA GeForce RTX 3080 Laptop GPU",
  "node_id": "284",
  "peak_ram_delta_mb": 1009.0,
  "peak_ram_mb": 15070.91,
  "peak_vram_delta_mb": 6067.86,
  "peak_vram_mb": 7425.36,
  "prompt_id": "8a2c25f1-722f-40b1-943f-c2a54c134e70",
  "sample_count": 13644,
  "sample_interval_seconds": 0.05,
  "total_seconds": 847.442
}
```