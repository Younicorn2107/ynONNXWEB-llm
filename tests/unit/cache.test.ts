import { describe, expect, it } from "vitest";

import {
  CACHE_FORMAT,
  buildCacheExport,
  dequantizeInt8,
  deserializeCache,
  identityMismatch,
  quantizeInt8,
  readCacheExport,
  serializeCache,
  type ModelIdentity,
} from "../../src/cache.js";

const identity: ModelIdentity = { model: "qwen3-0.6b", repo: "onnx-community/Qwen3-0.6B-ONNX", variant: "q4f16", layers: 2, kvHeads: 2, headDim: 4, fingerprint: "model_q4f16.onnx:570" };

/** A [1, 2, 3, 4] tensor whose keys have one loud channel and values one loud token. */
function snapshot(dtype: "float32" | "float16" = "float32") {
  const dims = [1, 2, 3, 4];
  const tensors = new Map<string, { dims: number[]; type: string; data: Float32Array | Uint16Array }>();
  for (let layer = 0; layer < 2; layer += 1) {
    for (const kind of ["key", "value"] as const) {
      const data = new Float32Array(24).map((_, i) => {
        const d = i % 4;
        const t = Math.floor(i / 4) % 3;
        const base = Math.sin(i + layer) * 0.5;
        return kind === "key" && d === 2 ? base * 40 : kind === "value" && t === 1 ? base * 40 : base;
      });
      tensors.set(`past_key_values.${layer}.${kind}`, { dims, type: dtype, data: dtype === "float16" ? toHalf(data) : data });
    }
  }
  return { tokens: [5, 6, 7], tensors };
}

function toHalf(values: Float32Array): Uint16Array {
  const f = new Float32Array(1);
  const u = new Uint32Array(f.buffer);
  return Uint16Array.from(values, (v) => {
    f[0] = v;
    const x = u[0]!;
    const sign = (x >>> 16) & 0x8000;
    const exponent = ((x >>> 23) & 0xff) - 112;
    if (exponent <= 0) return sign;
    return sign | (exponent << 10) | ((x >>> 13) & 0x3ff);
  });
}

describe("KV cache export", () => {
  it("round-trips a float32 cache exactly through one buffer", () => {
    const snap = snapshot();
    const bytes = serializeCache(buildCacheExport(identity, snap));
    const back = deserializeCache(bytes);
    expect(back.header.format).toBe(CACHE_FORMAT);
    expect(back.header.identity).toEqual(identity);
    expect(back.header.tokens).toEqual([5, 6, 7]);
    expect(back.header.tensors.map((t) => `${t.layer}.${t.kind}`)).toEqual(["0.key", "0.value", "1.key", "1.value"]);
    const tensors = readCacheExport(back);
    for (const [name, value] of snap.tensors) expect(Array.from(tensors.get(name)!.data)).toEqual(Array.from(value.data));
  });

  it("keeps float16 caches in float16", () => {
    const snap = snapshot("float16");
    const tensors = readCacheExport(deserializeCache(serializeCache(buildCacheExport(identity, snap))));
    const key = tensors.get("past_key_values.0.key")!;
    expect(key.type).toBe("float16");
    expect(key.data).toBeInstanceOf(Uint16Array);
    expect(Array.from(key.data)).toEqual(Array.from(snap.tensors.get("past_key_values.0.key")!.data));
  });

  it("quantizes keys per channel and values per token, each apart", () => {
    const snap = snapshot();
    const exported = buildCacheExport(identity, snap, { key: "int8", value: "none" });
    const records = Object.fromEntries(exported.header.tensors.map((t) => [t.name, t]));
    expect(records["past_key_values.0.key"]!.quantization).toBe("int8");
    expect(records["past_key_values.0.key"]!.scaleAxis).toBe("channel");
    expect(records["past_key_values.0.value"]!.quantization).toBe("none");
    const both = buildCacheExport(identity, snap, { key: "int8", value: "int8" });
    expect(both.body.byteLength).toBeLessThan(buildCacheExport(identity, snap).body.byteLength);
    const restored = readCacheExport(deserializeCache(serializeCache(both)));
    for (const [name, value] of snap.tensors) {
      const original = value.data as Float32Array;
      const back = restored.get(name)!.data as Float32Array;
      const max = Math.max(...Array.from(original, Math.abs));
      original.forEach((v, i) => expect(Math.abs(v - back[i]!)).toBeLessThanOrEqual(max / 127 + 1e-6));
    }
  });

  it("per-channel scales keep a loud channel from crushing the quiet ones", () => {
    const dims = [1, 1, 3, 4];
    const values = Float32Array.from([0.01, 0.02, 50, -0.03, 0.02, -0.01, -40, 0.01, 0.03, 0.02, 45, -0.02]);
    const channel = dequantizeInt8(quantizeInt8(values, dims, "channel").q, quantizeInt8(values, dims, "channel").scales, dims, "channel");
    const token = quantizeInt8(values, dims, "token");
    const perToken = dequantizeInt8(token.q, token.scales, dims, "token");
    const error = (out: Float32Array) => Math.max(...[0, 1, 3, 4, 5, 7, 8, 9, 11].map((i) => Math.abs(out[i]! - values[i]!)));
    expect(error(channel)).toBeLessThan(error(perToken));
  });

  it("names what differs between a cache and a model", () => {
    expect(identityMismatch(identity, identity)).toEqual([]);
    expect(identityMismatch(identity, { ...identity, variant: "q4", fingerprint: "model_q4.onnx:919" })).toEqual(["variant", "fingerprint"]);
  });

  it("refuses what is not a cache", () => {
    const header = new TextEncoder().encode(JSON.stringify({ format: "other", version: 1 }));
    const bytes = new Uint8Array(8 + header.byteLength);
    new DataView(bytes.buffer).setUint32(0, header.byteLength, true);
    bytes.set(header, 4);
    expect(() => deserializeCache(bytes)).toThrow(/not a KV cache export/);
  });
});
