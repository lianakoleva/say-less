"""Tests for the Coral NPU TOSA tiling tooling.

These run without IREE installed: they exercise the tile planner, the MLIR
emitter, the text-level function splicing, and the driver pipeline up to the
external ``iree-opt``/``iree-compile`` calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

NPU_DIR = Path(__file__).resolve().parents[1] / "scripts" / "npu"
sys.path.insert(0, str(NPU_DIR))

from compile_litert import main as compile_main
from mlir_text import (
    bind_dynamic_dims,
    dynamic_extent_count,
    find_function,
    find_functions,
    replace_function,
    tensor_stats,
)
from tosa_mhsa import (
    AttentionShape,
    TilePlan,
    TilePlanError,
    analyze_signature,
    emit_kernel,
    emit_wrapper,
    estimate_budget,
    select_plan,
)

MODULE = """
module {
  func.func private @helper(%x: tensor<1x2xf32>) -> tensor<1x2xf32> {
    return %x : tensor<1x2xf32>
  }
  func.func @attention_0(%q: tensor<1x{seq}x2x4xf32>, %k: tensor<1x{seq}x2x4xf32>,
                         %v: tensor<1x{seq}x2x4xf32>) -> tensor<1x{seq}x2x4xf32>
                         attributes {stage = "decode"} {
    %0 = tosa.add %q, %k : (tensor<1x{seq}x2x4xf32>, tensor<1x{seq}x2x4xf32>) -> tensor<1x{seq}x2x4xf32>
    return %0 : tensor<1x{seq}x2x4xf32>
  }
}
"""


def _module(seq: int = 8) -> str:
    return MODULE.replace("{seq}", str(seq))


# ---------------------------------------------------------------------------
# Tile planner
# ---------------------------------------------------------------------------


def test_budget_rejects_oversized_tiles() -> None:
    shape = AttentionShape(batch=4, heads=8, seq=1024, kv_seq=1024, head_dim=128)
    with pytest.raises(TilePlanError, match="needs .* budget"):
        select_plan(shape, budget_bytes=8 * 1024, q_tile=128, kv_tile=256)


def test_auto_plan_fits_and_divides_padded_shape() -> None:
    shape = AttentionShape(batch=1, heads=8, seq=256, kv_seq=1024, head_dim=64)
    plan, budget = select_plan(shape, budget_bytes=256 * 1024, causal=True)
    assert budget.fits
    assert plan.padded_seq(shape) % plan.q_tile == 0
    assert plan.padded_kv(shape) % plan.kv_tile == 0
    assert plan.q_tile <= shape.seq
    assert plan.kv_tile <= shape.kv_seq


def test_tighter_budget_never_picks_a_bigger_tile() -> None:
    shape = AttentionShape(batch=1, heads=8, seq=128, kv_seq=128, head_dim=64)
    large, _ = select_plan(shape, budget_bytes=256 * 1024)
    small, _ = select_plan(shape, budget_bytes=32 * 1024)
    assert small.q_tile * small.kv_tile <= large.q_tile * large.kv_tile


def test_budget_estimate_is_monotonic_in_tile_size() -> None:
    shape = AttentionShape(batch=2, heads=4, seq=128, kv_seq=128, head_dim=64)
    small = estimate_budget(shape, TilePlan(8, 8), causal=True)
    large = estimate_budget(shape, TilePlan(32, 32), causal=True)
    assert small.peak_bytes < large.peak_bytes


# ---------------------------------------------------------------------------
# Emitter
# ---------------------------------------------------------------------------


def test_kernel_is_static_and_explicitly_partitioned() -> None:
    shape = AttentionShape(batch=1, heads=2, seq=100, kv_seq=100, head_dim=8)
    plan = TilePlan(16, 16)
    text = emit_kernel(shape, plan, causal=True)

    assert "?" not in text
    assert "tosa.matmul" in text
    assert "tosa.slice" in text
    assert "tosa.reduce_max" in text
    assert "tosa.exp" in text
    # Padding is explicit and the result is sliced back to the real length.
    assert "tosa.pad" in text
    assert "dense<[0, 0, 0]>" in text and "dense<[2, 100, 8]>" in text
    # 7 query tiles x 7 key/value tiles, each with two matmuls.
    assert text.count("tosa.matmul") == 2 * 7 * 7
    assert "tile_q = 16 : i64" in text
    assert "tile_kv = 16 : i64" in text
    # No op may be wider than the padded tile.
    assert "tensor<2x32x32xf32>" not in text


def test_kernel_shape_annotation_and_unrolled_steps() -> None:
    shape = AttentionShape(batch=1, heads=4, seq=64, kv_seq=64, head_dim=16)
    plan = TilePlan(16, 32)
    text = emit_kernel(shape, plan, budget_bytes=65536)
    assert "unrolled_steps = 8 : i64" in text
    assert "required_bytes" in text


def test_causal_mask_marks_future_tiles() -> None:
    shape = AttentionShape(batch=1, heads=1, seq=8, kv_seq=8, head_dim=4)
    text = emit_kernel(shape, TilePlan(4, 4), causal=True)
    # Second query tile / first key tile: rows 4..7 see keys 0..3 only.
    assert "dense<[[[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]" in text
    assert "-1.000000e+30" in text


def test_wrapper_preserves_signature_and_layout() -> None:
    signature = find_function(_module(8), "attention_0")
    assert signature is not None
    spec = analyze_signature(signature.args, signature.result_type)
    assert (
        spec.shape.batch,
        spec.shape.seq,
        spec.shape.heads,
        spec.shape.head_dim,
    ) == (
        1,
        8,
        2,
        4,
    )
    text = emit_wrapper(signature, spec, TilePlan(4, 4), "kernel_test", causal=True)
    assert "func.func @attention_0(%q: tensor<1x8x2x4xf32>" in text
    assert "func.func @kernel_test(%q: tensor<2x8x4xf32>" in text
    assert "tosa.transpose" in text
    assert "func.call @kernel_test" in text
    assert 'attributes {stage = "decode"}' in text


def test_analyze_signature_rejects_mismatched_heads() -> None:
    text = _module(8).replace("%k: tensor<1x8x2x4xf32>", "%k: tensor<1x8x3x4xf32>")
    signature = find_function(text, "attention_0")
    assert signature is not None
    with pytest.raises(ValueError, match="layout mismatch"):
        analyze_signature(signature.args, signature.result_type)


def test_packed_layout() -> None:
    text = _module(8).replace("1x8x2x4xf32", "2x8x4xf32")
    signature = find_function(text, "attention_0")
    assert signature is not None
    spec = analyze_signature(signature.args, signature.result_type)
    assert spec.layout == "packed"
    assert spec.shape.bh == 2


# ---------------------------------------------------------------------------
# MLIR text handling
# ---------------------------------------------------------------------------


def test_bind_dynamic_dims() -> None:
    text = _module(8).replace("1x8x2x4", "1x?x2x4")
    assert dynamic_extent_count(text) == 8
    bound, replaced = bind_dynamic_dims(text, 8)
    assert replaced == 8
    assert dynamic_extent_count(bound) == 0
    assert "1x8x2x4" in bound


def test_find_and_replace_function() -> None:
    text = _module(8)
    functions = find_functions(text)
    assert [f.name for f in functions] == ["helper", "attention_0"]
    attention = find_function(text, "attention_0")
    assert attention is not None
    assert attention.header_attrs == 'attributes {stage = "decode"}'
    replacement = "func.func @attention_0(%q: tensor<1x8x2x4xf32>) -> tensor<1x8x2x4xf32> {\n  return %q : tensor<1x8x2x4xf32>\n}\n"
    spliced = replace_function(text, attention, replacement)
    assert "tosa.add" not in spliced
    assert "func.func private @helper" in spliced
    assert find_function(spliced, "attention_0") is not None


def test_tensor_stats_are_sorted() -> None:
    stats = tensor_stats(_module(8))
    assert stats[0].bytes_per_instance >= stats[-1].bytes_per_instance
    assert stats[0].type == "tensor<1x8x2x4xf32>"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def test_driver_attention_stage_end_to_end(tmp_path: Path) -> None:
    model = tmp_path / "model.mlir"
    model.write_text(_module(8))
    out = tmp_path / "out"

    code = compile_main(
        [
            str(model),
            "--seq-len",
            "8",
            "--stop-after",
            "attention",
            "--out-dir",
            str(out),
        ]
    )

    assert code == 0
    tiled = (out / "03_tiled.mlir").read_text()
    assert "func.func @attention_0" in tiled
    assert "func.func @mhsa_tiled_attention_0" in tiled
    assert "tosa.matmul" in tiled
    assert "?" not in tiled
    assert (out / "tile_plan.json").exists()
    assert (out / "mhsa_tiled.mlir").exists()


def test_driver_binds_dynamic_extent(tmp_path: Path) -> None:
    model = tmp_path / "model.mlir"
    model.write_text(_module(8).replace("1x8x2x4", "1x?x2x4"))
    out = tmp_path / "out"

    assert (
        compile_main(
            [
                str(model),
                "--seq-len",
                "8",
                "--stop-after",
                "bind",
                "--out-dir",
                str(out),
            ]
        )
        == 0
    )
    bound = (out / "02_static.mlir").read_text()
    assert "?" not in bound
    assert "tensor<1x8x2x4xf32>" in bound


def test_driver_requires_seq_len_for_dynamic_graph(tmp_path: Path) -> None:
    model = tmp_path / "model.mlir"
    model.write_text(_module(8).replace("1x8x2x4", "1x?x2x4"))
    with pytest.raises(SystemExit, match="--seq-len"):
        compile_main(
            [str(model), "--stop-after", "bind", "--out-dir", str(tmp_path / "o")]
        )


def test_driver_requires_attention_function(tmp_path: Path) -> None:
    model = tmp_path / "model.mlir"
    model.write_text(_module(8).replace("@attention_0", "@projection"))
    with pytest.raises(SystemExit, match="no multi-head attention function"):
        compile_main(
            [str(model), "--stop-after", "attention", "--out-dir", str(tmp_path / "o")]
        )
