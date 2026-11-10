"""Export Kyutai Pocket TTS (MIT code, `pocket_tts`) to weight-free ONNX graphs for peresmeshnik/local/pocket.py.

Run by us, not by users: on a machine with `pocket_tts` (torch), `onnx` and `onnxruntime` (made with pocket_tts
4.0.1, torch 2.15 CPU, onnx 1.23.2, onnxruntime 1.31.1; the legacy TorchScript exporter, opset 17; deterministic):

    python packaging/pocket/export.py --out peresmeshnik/local/pocket_onnx [--config english] [--check WEIGHTS]

The graphs, all streaming state as explicit inputs and outputs:
- flow_lm.onnx: the FlowLM transformer over input embeddings `x` with its KV cache, then the EOS head and one LSD
  step of the flow head for the given `noise` (latent = noise + flow(h, 0, 1, noise));
- mimi_decoder.onnx: Mimi latents to 24 kHz audio, its conv and attention (context 250) states carried along;
- mimi_encoder.onnx: a voice sample (24 kHz, a multiple of 1920 samples) to the FlowLM voice conditioning;
- flow_lm_int8.onnx: flow_lm.onnx with its transformer's Linear layers in int8 (int8(): MatMulInteger).
Every weight is an initializer with external data and its safetensors name (`.T`: transposed, `.int8` /
`.int8_scale`: quantized, both made by the app): the app reads the official `model.safetensors` with numpy and hands
the arrays to onnxruntime (`add_external_initializers`), so no weight is in the graphs. graphs.json lists the weights
(name: shape) per graph and the decoder's initial state shapes.
The text lookup table, the latent input projection and the BOS vectors stay outside (numpy in the app).
--check compares the graphs, fed through the app's loader, with torch on a weights file (or random weights).
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pocket_tts.models.tts_model import TTSModel, stamp_state_names
from pocket_tts.modules import conv as pconv
from pocket_tts.modules.attention import StreamingMultiheadAttention
from pocket_tts.modules.rope import apply_rope
from pocket_tts.utils.config import CONFIGS_DIR, load_config

OPSET = 17
LOCATION = 'model.safetensors'  # placeholder: the arrays come from the app at session creation
IN: dict[str, torch.Tensor] = {}  # streaming state of the traced call, by module name ('offset': RoPE position)
OUT: dict[str, torch.Tensor] = {}


def conv_forward(self, x, model_state):
    """StreamingConv1d with its `previous` samples as a graph input and output (no in-place state)."""
    if model_state is None:
        return ORIG_CONV(self, x, model_state)
    tp = self._effective_kernel_size - self._stride
    if not tp:
        return self.conv(x)
    x = torch.cat([IN[self._module_absolute_name], x], dim=-1)
    OUT[self._module_absolute_name] = x[..., -tp:]
    return self.conv(x)


def convtr_forward(self, x, model_state):
    y = self.convtr(x)
    pt = self._kernel_size - self._stride
    y = torch.cat([y[..., :pt] + IN[self._module_absolute_name], y[..., pt:]], dim=-1)
    tail = y[..., -pt:]
    if self.convtr.bias is not None:
        tail = tail - self.convtr.bias[:, None]
    OUT[self._module_absolute_name] = tail
    return y[..., :-pt]


def attn_forward(self, query, model_state, attn_mask=None):
    """Attention with the KV cache [2, H, P, D] as input and output (P + T, or the last `context` positions)."""
    b, t, _ = query.shape
    d = self.dim_per_head
    proj = self.in_proj(query)
    q = proj[..., :self.attn_dim].reshape(b, t, self.num_heads, d)
    kv = proj[..., self.attn_dim:].reshape(b, t, 2, self.num_kv_heads, d)
    k, v = kv[:, :, 0], kv[:, :, 1]
    offset = torch.zeros((), dtype=torch.int64) if model_state is None else IN['offset']
    q, k = apply_rope(q, k, offset, self.rope.max_period)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    past_len = 0
    if model_state is not None:
        past = IN[self._module_absolute_name]
        past_len = past.shape[2]
        k = torch.cat([past[0][None], k], dim=2)
        v = torch.cat([past[1][None], v], dim=2)
        present = torch.cat([k, v], dim=0)
        OUT[self._module_absolute_name] = present[:, :, -self.context:] if self.context else present
    delta = (torch.arange(t) + past_len)[:, None] - torch.arange(k.shape[2])[None, :]
    mask = delta >= 0
    if self.context:
        mask = mask & (delta < self.context)
    if self.num_kv_heads != self.num_heads:
        k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
    x = F.scaled_dot_product_attention(q, k, v, mask[None, None])
    return self.out_proj(x.transpose(1, 2).reshape(b, t, self.num_heads * d))


ORIG_CONV = pconv.StreamingConv1d.forward


def patch():
    pconv.StreamingConv1d.forward = conv_forward
    pconv.StreamingConvTranspose1d.forward = convtr_forward
    StreamingMultiheadAttention.forward = attn_forward


def attn_names(module: torch.nn.Module) -> list[str]:
    return [m._module_absolute_name for _, m in module.named_modules() if isinstance(m, StreamingMultiheadAttention)]


class FlowStep(torch.nn.Module):
    def __init__(self, tts: TTSModel):
        super().__init__()
        self.flow_lm = tts.flow_lm
        self.names = attn_names(tts.flow_lm.transformer)

    def forward(self, x, cache, offset, noise):
        IN.clear(), OUT.clear()
        IN.update({n: cache[i] for i, n in enumerate(self.names)}, offset=offset)
        fl = self.flow_lm
        h = fl.out_norm(fl.transformer(x, {}))[:, -1].float()
        if fl.z_proj is not None:
            h = fl.z_proj(h)
        eos = fl.out_eos(h)
        one = torch.ones_like(noise[..., :1])
        latent = noise + fl.flow_net(h, one * 0, one, noise)
        return latent, eos, torch.stack([OUT[n] for n in self.names])


def decoder_states(mimi) -> list[tuple[str, list[int]]]:
    """(module name, initial shape) of every streaming state on the decoder side, in graph order."""
    out = []
    for name, m in mimi.named_modules():
        if name.startswith(('encoder', 'downsample')):
            continue
        if isinstance(m, pconv.StreamingConv1d) and (tp := m._effective_kernel_size - m._stride):
            out.append((m._module_absolute_name, [1, m.conv.in_channels, tp]))
        elif isinstance(m, pconv.StreamingConvTranspose1d):
            out.append((m._module_absolute_name, [1, m.convtr.out_channels, m._kernel_size - m._stride]))
        elif isinstance(m, StreamingMultiheadAttention):
            out.append((m._module_absolute_name, [2, m.num_kv_heads, 0, m.dim_per_head]))
    return out


class Decoder(torch.nn.Module):
    def __init__(self, tts: TTSModel):
        super().__init__()
        self.mimi = tts.mimi
        self.names = [n for n, _ in decoder_states(tts.mimi)]

    def forward(self, latent, offset, *states):
        IN.clear(), OUT.clear()
        IN.update(zip(self.names, states), offset=offset)
        audio = self.mimi.decode_from_latent(latent, {})
        return (audio, *[OUT[n] for n in self.names])


class Encoder(torch.nn.Module):
    def __init__(self, tts: TTSModel):
        super().__init__()
        self.mimi = tts.mimi
        self.flow_lm = tts.flow_lm

    def forward(self, audio):
        """MimiModel.encode_to_latent without its padding to whole frames (traced, it would crop every input to the
        example's length): the app pads."""
        (emb,) = self.mimi.encoder_transformer(self.mimi.encoder(audio, model_state=None), model_state=None)
        latent = self.mimi._to_framerate(emb).transpose(-1, -2)
        return F.linear(latent.float(), self.flow_lm.speaker_proj_weight)


def build(config: str) -> TTSModel:
    cfg = load_config(CONFIGS_DIR / f'{config}.yaml' if not config.endswith(('.yaml', '.yml')) else config)
    tts = TTSModel._from_pydantic_config(cfg, 0.3, 1, None, -4.0, origin=None)
    from pocket_tts.models.mimi import build_mimi
    tts.mimi = build_mimi(cfg.mimi)
    dim = cfg.flow_lm.transformer.d_model
    inner = cfg.mimi.inner_dim or cfg.mimi.seanet.dimension
    tts.flow_lm.speaker_proj_weight = torch.nn.Parameter(torch.randn(dim, inner) * 0.02)
    tts.eval()
    with torch.no_grad():  # no two weights alike: the exporter merges equal initializers (both alphas start at 1)
        for t in tts.state_dict().values():
            if t.is_floating_point():
                t.add_(torch.randn_like(t) * 1e-3)
    stamp_state_names(tts)
    return tts


def externalize(path: Path, names: set[str]) -> dict[str, list[int]]:
    """Strip the weights out of an exported graph: initializers named as weights become external placeholders."""
    import onnx
    model = onnx.load(str(path))
    # Cast float -> float of a weight (RMSNorm's alpha.to(var)): onnxruntime's constant folding loses the folded value
    # of an injected initializer there ("not a graph input, initializer, or output"); the cast does nothing anyway
    init = {t.name: t for t in model.graph.initializer}
    noop = {n.output[0]: n.input[0] for n in model.graph.node if n.op_type == 'Cast' and n.input[0] in init
            and init[n.input[0]].data_type == onnx.TensorProto.FLOAT
            and onnx.helper.get_attribute_value(n.attribute[0]) == onnx.TensorProto.FLOAT}
    # Linear on 3-D input: MatMul(x, Transpose(W)). onnxruntime folds the transpose into a copy of W while the app's W
    # stays alive (~330 MB twice): the graph takes `<name>.T` instead, transposed once by the app
    for n in model.graph.node:
        if n.op_type == 'Transpose' and n.input[0] in names and n.input[0] in init and len(init[n.input[0]].dims) == 2:
            w = onnx.numpy_helper.to_array(init[n.input[0]])
            model.graph.initializer.append(onnx.numpy_helper.from_array(np.ascontiguousarray(w.T), n.input[0] + '.T'))
            noop[n.output[0]] = n.input[0] + '.T'
    keep = [n for n in model.graph.node if n.output[0] not in noop]
    del model.graph.node[:]
    model.graph.node.extend(keep)
    for n in model.graph.node:
        for i, x in enumerate(n.input):
            n.input[i] = noop.get(x, x)
    used = {x for n in model.graph.node for x in n.input} | {o.name for o in model.graph.output}
    keep = [t for t in model.graph.initializer if t.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    names = names | {n + '.T' for n in names}
    kept, inline = {}, 0
    for t in model.graph.initializer:
        if t.name in names:
            t.ClearField('raw_data')
            for f in ('float_data', 'int64_data', 'int32_data', 'double_data'):
                t.ClearField(f)
            t.data_location = onnx.TensorProto.EXTERNAL
            del t.external_data[:]
            for k, v in (('location', LOCATION), ('offset', '0'), ('length', '0')):
                e = t.external_data.add()
                e.key, e.value = k, v
            kept[t.name] = list(t.dims)
        else:
            inline += onnx.numpy_helper.to_array(t).nbytes
    unknown = [t.name for t in model.graph.initializer
               if t.name not in names and onnx.numpy_helper.to_array(t).size > 64]
    if unknown:
        raise SystemExit(f'{path.name}: initializers that are not weights: {unknown[:10]}')
    dangling = {x for n in model.graph.node for x in n.input} & names - set(kept)
    if dangling:
        raise SystemExit(f'{path.name}: weights used but not initializers (merged?): {sorted(dangling)[:10]}')
    onnx.save(model, str(path))
    print(f'{path.name}: {len(kept)} weights external, {inline} bytes inline, {path.stat().st_size / 1e3:.0f} kB')
    return kept


def int8(src: Path, dst: Path) -> dict[str, list[int]]:
    """The FlowLM graph with its transformer's Linear layers (attention and feed-forward, as pocket_tts' quantize=True)
    in int8: activations quantized per call (DynamicQuantizeLinear), weights per output channel, symmetric, by the app
    from the float weights (`<name>.int8` [in, out], `<name>.int8_scale` [out]; pocket.quantize)."""
    import onnx
    from onnx import TensorProto, helper
    model = onnx.load(str(src), load_external_data=False)
    g, init = model.graph, {t.name: t for t in model.graph.initializer}
    nodes, done = [], {}
    for n in g.node:
        w = n.input[1] if n.op_type == 'MatMul' and len(n.input) > 1 else ''
        if not (w.endswith('.T') and '.transformer.layers.' in w):
            nodes.append(n)
            continue
        base, p = w[:-2], n.output[0]
        k, m = init[w].dims
        done[base] = (k, m)
        nodes += [helper.make_node('DynamicQuantizeLinear', [n.input[0]], [p + '_xq', p + '_xs', p + '_xz']),
                  helper.make_node('MatMulInteger', [p + '_xq', base + '.int8', p + '_xz'], [p + '_yi']),
                  helper.make_node('Cast', [p + '_yi'], [p + '_yf'], to=TensorProto.FLOAT),
                  helper.make_node('Mul', [p + '_xs', base + '.int8_scale'], [p + '_s']),
                  helper.make_node('Mul', [p + '_yf', p + '_s'], [p])]
    del g.node[:]
    g.node.extend(nodes)
    keep = [t for t in g.initializer if not (t.name.endswith('.T') and t.name[:-2] in done)]
    del g.initializer[:]
    g.initializer.extend(keep)
    shapes = {}
    for base, (k, m) in done.items():
        for name, dt, dims in ((base + '.int8', TensorProto.INT8, [k, m]),
                               (base + '.int8_scale', TensorProto.FLOAT, [m])):
            t = TensorProto(name=name, data_type=dt, dims=dims)
            t.data_location = TensorProto.EXTERNAL
            for key, v in (('location', LOCATION), ('offset', '0'), ('length', '0')):
                e = t.external_data.add()
                e.key, e.value = key, v
            g.initializer.append(t)
    for t in g.initializer:
        shapes[t.name] = list(t.dims)
    onnx.save(model, str(dst))
    print(f'{dst.name}: {len(done)} MatMul in int8, {dst.stat().st_size / 1e3:.0f} kB')
    return shapes


def export(tts: TTSModel, out: Path) -> dict:
    patch()
    out.mkdir(parents=True, exist_ok=True)
    weights = set(tts.state_dict())
    fl, layers = tts.flow_lm, len(tts.flow_lm.transformer.layers)
    a0 = fl.transformer.layers[0].self_attn
    kw = dict(opset_version=OPSET, dynamo=False, do_constant_folding=False)
    p = 3
    cache = torch.randn(layers, 2, a0.num_kv_heads, p, a0.dim_per_head)
    torch.onnx.export(FlowStep(tts), (torch.randn(1, 4, fl.dim), cache, torch.tensor(p), torch.randn(1, fl.ldim)),
                      str(out / 'flow_lm.onnx'), input_names=['x', 'cache', 'offset', 'noise'],
                      output_names=['latent', 'eos', 'cache_out'],
                      dynamic_axes={'x': {1: 't'}, 'cache': {3: 'p'}, 'cache_out': {3: 'p_out'}}, **kw)
    states = decoder_states(tts.mimi)
    ex = [torch.randn(*[5 if d == 0 else d for d in s]) for _, s in states]
    names = [f'state{i}' for i in range(len(states))]
    torch.onnx.export(Decoder(tts), (torch.randn(1, 2, fl.ldim), torch.tensor(80), *ex), str(out / 'mimi_decoder.onnx'),
                      input_names=['latent', 'offset', *names], output_names=['audio', *[n + '_out' for n in names]],
                      dynamic_axes={'latent': {1: 'n'}, 'audio': {2: 'samples'},
                                    **{n: {2: 'p'} for n, (_, s) in zip(names, states) if s[2] == 0},
                                    **{n + '_out': {2: 'p_out'} for n, (_, s) in zip(names, states) if s[2] == 0}},
                      **kw)
    torch.onnx.export(Encoder(tts), (torch.randn(1, 1, 1920 * 4),), str(out / 'mimi_encoder.onnx'),
                      input_names=['audio'], output_names=['cond'],
                      dynamic_axes={'audio': {2: 'samples'}, 'cond': {1: 'frames'}}, **kw)
    meta = {'graphs': {g: externalize(out / f'{g}.onnx', weights) for g in ('flow_lm', 'mimi_decoder', 'mimi_encoder')},
            'layers': layers, 'heads': a0.num_kv_heads, 'head_dim': a0.dim_per_head, 'dim': fl.dim, 'ldim': fl.ldim,
            'decoder_states': [s for _, s in states], 'frame': int(tts.mimi.frame_size),
            'decoder_steps_per_latent': int(tts.mimi.encoder_frame_rate / tts.mimi.frame_rate)}
    meta['graphs']['flow_lm_int8'] = int8(out / 'flow_lm.onnx', out / 'flow_lm_int8.onnx')
    text = json.dumps(meta, indent=1)
    text = re.sub(r'\[\s+([\d,\s]+?)\s+\]', lambda m: '[' + ', '.join(m[1].split()).replace(',,', ',') + ']', text)
    (out / 'graphs.json').write_text(text + '\n')
    return meta


def check(tts: TTSModel, out: Path, weights: Path | None) -> None:
    """The graphs through the app's loader against the torch modules (patched, same math as pocket_tts)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from safetensors.torch import load_file, save_file

    from peresmeshnik.local import pocket
    if weights is None:  # the model's own random weights, outside the graphs' folder
        import tempfile
        weights = Path(tempfile.mkdtemp()) / 'random.safetensors'
        save_file({k: v.contiguous() for k, v in tts.state_dict().items()}, str(weights))
    else:
        tts.load_state_dict(load_file(str(weights)), strict=True)
    eng = pocket.Graphs(pocket.read_safetensors(weights), threads=2, graphs=out)
    torch.manual_seed(0)
    with torch.no_grad():
        audio = torch.randn(1, 1, 1920 * 30) * 0.1
        ref = Encoder(tts)(audio).numpy()
        got = eng.encode(audio.numpy())
        print('encoder max abs diff', np.abs(ref - got).max(), 'scale', np.abs(ref).max())
        fl = tts.flow_lm
        a0 = fl.transformer.layers[0].self_attn
        cache = torch.zeros(len(fl.transformer.layers), 2, a0.num_kv_heads, 0, a0.dim_per_head)
        step, worst = FlowStep(tts), 0.0
        state = eng.empty_cache()
        for t in (31, 7, 1, 1, 1):
            x, noise = torch.randn(1, t, fl.dim), torch.randn(1, fl.ldim)
            lat, eos, cache = step(x, cache, torch.tensor(cache.shape[3]), noise)
            glat, geos, state = eng.flow(x.numpy(), state, noise.numpy())
            worst = max(worst, float(np.abs(lat.numpy() - glat).max()), float(np.abs(eos.numpy() - geos).max()))
        print('flow max abs diff', worst, 'cache', float(np.abs(cache.numpy() - state.numpy()).max()))
        dec, st = Decoder(tts), [torch.zeros(*s) for _, s in decoder_states(tts.mimi)]
        gst, offset, worst = eng.decoder_state(), 0, 0.0
        for n in (1, 3, 1, 20, 1):  # past 250 positions of the decoder transformer
            lat = torch.randn(1, n, fl.ldim)
            ref_audio, *st = dec(lat, torch.tensor(offset), *st)
            got_audio, gst = eng.decode(lat.numpy(), gst)
            offset += n * 16
            worst = max(worst, float(np.abs(ref_audio.numpy() - got_audio).max()))
        print('decoder max abs diff', worst, 'positions', offset)
        # the stock streaming path (pocket_tts' own cache) gives the same audio as the patched one
        mimi_state = __import__('pocket_tts.modules.stateful_module', fromlist=['x']).init_states(tts.mimi, 1, 1000)
        patch_off()
        lat = torch.randn(1, 24, fl.ldim)  # 384 positions: past the decoder transformer's context of 250
        a = tts.mimi.decode_from_latent(lat, mimi_state)
        stock = F.linear(tts.mimi.encode_to_latent(audio).float(), fl.speaker_proj_weight)
        patch()
        b = dec(lat, torch.tensor(0), *[torch.zeros(*s) for _, s in decoder_states(tts.mimi)])[0]
        print('patched vs stock: decoder', float((a - b).abs().max()),
              'encoder', float((stock - Encoder(tts)(audio)).abs().max()))
        eng8 = pocket.Graphs(pocket.read_safetensors(weights), threads=2, graphs=out, int8=True)
        s32, s8, err = eng.empty_cache(), eng8.empty_cache(), []
        for t in (31, 7, 1, 1, 1):
            x, noise = np.random.randn(1, t, fl.dim).astype(np.float32), np.random.randn(1, fl.ldim).astype(np.float32)
            a32, _, s32 = eng.flow(x, s32, noise)
            a8, _, s8 = eng8.flow(x, s8, noise)
            err.append(float(np.abs(a32 - a8).max() / np.abs(a32).max()))
        print('int8 vs fp32 flow: max relative diff of the latent', max(err))
        assert math.isfinite(worst)


def patch_off():
    pconv.StreamingConv1d.forward = ORIG_CONV
    pconv.StreamingConvTranspose1d.forward = ORIG_CONVTR
    StreamingMultiheadAttention.forward = ORIG_ATTN


ORIG_CONVTR = pconv.StreamingConvTranspose1d.forward
ORIG_ATTN = StreamingMultiheadAttention.forward


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', default='english', help='a pocket_tts language or a config .yaml (shapes only)')
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[2] / 'peresmeshnik/local/pocket_onnx')
    ap.add_argument('--check', nargs='?', const='', help='compare with torch: on this weights file, or random weights')
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    tts = build(a.config)
    print(json.dumps({k: v for k, v in export(tts, a.out).items() if k != 'graphs'}))
    if a.check is not None:
        check(tts, a.out, Path(a.check) if a.check else None)


if __name__ == '__main__':
    main()
