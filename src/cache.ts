/**
 * The KV cache and the context, exported and imported — to keep a conversation across page loads,
 * or to hand it to a server.
 */

import { ConfigurationError, YnONNXError, registerErrors } from "@youni/onnxweb-core/core";

import type { KvData } from "./session.js";

export class CacheMismatchError extends YnONNXError {
  static override readonly code: string = "CacheMismatchError";
}
registerErrors(CacheMismatchError);

export const CACHE_FORMAT = "ynonnxweb.llm.kv-cache";
export const CONTEXT_FORMAT = "ynonnxweb.llm.context";
export const CACHE_VERSION = 1;

/** What a cache belongs to. Two caches are interchangeable only if every field matches. */
export interface ModelIdentity {
  /** Catalogue key, or the repository for a model outside the catalogue. */
  model: string;
  repo: string | null;
  variant: string;
  layers: number;
  kvHeads: number;
  headDim: number;
  /** The graph files and their sizes: a re-export changes it. */
  fingerprint: string;
}

export type KvQuantization = "none" | "int8";

export interface TensorRecord {
  /**
   * `past_key_values.N.key` / `.value`; `past_conv.N` / `past_recurrent.N` for a hybrid model's
   * linear-attention states.
   */
  name: string;
  /** "key" or "value"; "conv" or "recurrent" for the states (kept as they are, never quantized). */
  kind: "key" | "value" | "conv" | "recurrent";
  layer: number;
  dims: number[];
  /** The model's dtype (float16 or float32). */
  dtype: string;
  quantization: KvQuantization;
  /** int8 only: one scale per channel (keys) or per token (values), as float32. */
  scaleAxis: "channel" | "token" | null;
  byteOffset: number;
  byteLength: number;
  scaleOffset: number | null;
  scaleLength: number | null;
}

export interface CacheHeader {
  format: typeof CACHE_FORMAT;
  version: number;
  identity: ModelIdentity;
  tokens: number[];
  tensors: TensorRecord[];
  createdAt: string;
}

export interface ExportedCache {
  header: CacheHeader;
  /** The tensors' bytes, as the header's offsets describe them. */
  body: Uint8Array;
}

export interface ContextExport {
  format: typeof CONTEXT_FORMAT;
  version: number;
  identity: ModelIdentity;
  /** The conversation, as given to `chat()` (the last turn's answer included). */
  messages: { role: string; content: string }[];
  /** The tokens the cache held when exported. */
  tokens: number[];
  /** The rendered prompt those tokens came from. */
  text: string;
}

const PAST = /^past_key_values\.(\d+)\.(key|value)$/;
const STATE = /^past_(conv|recurrent)\.(\d+)$/;

/** Layer and kind of a cache input: KV, or a linear-attention state. */
function partOf(name: string): { layer: number; kind: TensorRecord["kind"] } | null {
  const kv = PAST.exec(name);
  if (kv) return { layer: Number(kv[1]), kind: kv[2] as "key" | "value" };
  const state = STATE.exec(name);
  if (state) return { layer: Number(state[2]), kind: state[1] as "conv" | "recurrent" };
  return null;
}

function float16ToFloat32(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 31) return fraction ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function float32ToFloat16(value: number): number {
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  f32[0] = value;
  const x = u32[0]!;
  const sign = (x >>> 16) & 0x8000;
  let exponent = ((x >>> 23) & 0xff) - 127 + 15;
  let mantissa = x & 0x7fffff;
  if (exponent >= 31) return sign | 0x7c00 | (Number.isNaN(value) ? 0x200 : 0);
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    mantissa = (mantissa | 0x800000) >> (1 - exponent);
    return sign | ((mantissa + 0x1000) >> 13);
  }
  const rounded = mantissa + 0x1000;
  if (rounded & 0x800000) {
    exponent += 1;
    if (exponent >= 31) return sign | 0x7c00;
    return sign | (exponent << 10);
  }
  return sign | (exponent << 10) | (rounded >> 13);
}

/** A tensor's values as float32, whatever its storage. */
function toFloat32(data: KvData | ArrayLike<number>, dtype: string): Float32Array {
  if (dtype !== "float16" || !(data instanceof Uint16Array)) return Float32Array.from(data as ArrayLike<number>);
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i += 1) out[i] = float16ToFloat32(data[i]!);
  return out;
}

/**
 * int8 with a scale per channel (the last axis) or per token (the third): `q = round(x / scale)`,
 * `scale = max|x| / 127` over the other axes.
 */
export function quantizeInt8(values: Float32Array, dims: readonly number[], axis: "channel" | "token"): { q: Int8Array; scales: Float32Array } {
  const [batch, heads, steps, dim] = dims as [number, number, number, number];
  const groups = axis === "channel" ? batch * heads * dim : batch * heads * steps;
  const scales = new Float32Array(groups);
  const group = (b: number, h: number, t: number, d: number) => (axis === "channel" ? (b * heads + h) * dim + d : (b * heads + h) * steps + t);
  const at = (b: number, h: number, t: number, d: number) => ((b * heads + h) * steps + t) * dim + d;
  for (let b = 0; b < batch; b += 1)
    for (let h = 0; h < heads; h += 1)
      for (let t = 0; t < steps; t += 1)
        for (let d = 0; d < dim; d += 1) {
          const g = group(b, h, t, d);
          scales[g] = Math.max(scales[g]!, Math.abs(values[at(b, h, t, d)]!));
        }
  for (let g = 0; g < groups; g += 1) scales[g] = scales[g]! / 127 || 1;
  const q = new Int8Array(values.length);
  for (let b = 0; b < batch; b += 1)
    for (let h = 0; h < heads; h += 1)
      for (let t = 0; t < steps; t += 1)
        for (let d = 0; d < dim; d += 1) {
          const i = at(b, h, t, d);
          q[i] = Math.max(-127, Math.min(127, Math.round(values[i]! / scales[group(b, h, t, d)]!)));
        }
  return { q, scales };
}

export function dequantizeInt8(q: Int8Array, scales: Float32Array, dims: readonly number[], axis: "channel" | "token"): Float32Array {
  const [batch, heads, steps, dim] = dims as [number, number, number, number];
  const out = new Float32Array(q.length);
  for (let b = 0; b < batch; b += 1)
    for (let h = 0; h < heads; h += 1)
      for (let t = 0; t < steps; t += 1)
        for (let d = 0; d < dim; d += 1) {
          const i = ((b * heads + h) * steps + t) * dim + d;
          const g = axis === "channel" ? (b * heads + h) * dim + d : (b * heads + h) * steps + t;
          out[i] = q[i]! * scales[g]!;
        }
  return out;
}

const align8 = (n: number) => (n + 7) & ~7;

/** Build an export from a session snapshot. */
export function buildCacheExport(
  identity: ModelIdentity,
  snapshot: { tokens: number[]; tensors: ReadonlyMap<string, { dims: number[]; type: string; data: KvData }> },
  quantization: { key?: KvQuantization; value?: KvQuantization } = {},
): ExportedCache {
  const parts: Uint8Array[] = [];
  const tensors: TensorRecord[] = [];
  let offset = 0;
  const push = (bytes: Uint8Array) => {
    const at = offset;
    parts.push(bytes);
    const padded = align8(bytes.byteLength);
    if (padded > bytes.byteLength) parts.push(new Uint8Array(padded - bytes.byteLength));
    offset += padded;
    return at;
  };
  const names = [...snapshot.tensors.keys()].sort((a, b) => {
    const pa = partOf(a);
    const pb = partOf(b);
    return (pa?.layer ?? 0) - (pb?.layer ?? 0) || String(pa?.kind).localeCompare(String(pb?.kind));
  });
  for (const name of names) {
    const tensor = snapshot.tensors.get(name)!;
    const part = partOf(name);
    if (!part) throw new ConfigurationError(`'${name}' is not a cache input (past_key_values.N.key/value, past_conv.N, past_recurrent.N)`);
    const kind = part.kind;
    const match = [name, String(part.layer)];
    const mode = kind === "key" ? (quantization.key ?? "none") : kind === "value" ? (quantization.value ?? "none") : "none";
    if (mode === "int8") {
      const axis = kind === "key" ? "channel" : "token";

      const { q, scales } = quantizeInt8(toFloat32(tensor.data, tensor.type), tensor.dims, axis);
      const byteOffset = push(new Uint8Array(q.buffer, q.byteOffset, q.byteLength));
      const scaleOffset = push(new Uint8Array(scales.buffer));
      tensors.push({ name, kind, layer: Number(match[1]), dims: tensor.dims, dtype: tensor.type, quantization: "int8", scaleAxis: axis, byteOffset, byteLength: q.byteLength, scaleOffset, scaleLength: scales.byteLength });
    } else {
      const data = tensor.data as ArrayBufferView;
      const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
      const byteOffset = push(bytes);
      tensors.push({ name, kind, layer: Number(match[1]), dims: tensor.dims, dtype: tensor.type, quantization: "none", scaleAxis: null, byteOffset, byteLength: bytes.byteLength, scaleOffset: null, scaleLength: null });
    }
  }
  const body = new Uint8Array(offset);
  let cursor = 0;
  for (const part of parts) {
    body.set(part, cursor);
    cursor += part.byteLength;
  }
  return { header: { format: CACHE_FORMAT, version: CACHE_VERSION, identity, tokens: [...snapshot.tokens], tensors, createdAt: new Date().toISOString() }, body };
}

/** The tensors of an export, back in their model dtype. */
export function readCacheExport(exported: ExportedCache): Map<string, { dims: number[]; type: string; data: KvData }> {
  const out = new Map<string, { dims: number[]; type: string; data: KvData }>();
  const { body } = exported;
  const slice = (offset: number, length: number) => body.buffer.slice(body.byteOffset + offset, body.byteOffset + offset + length);
  for (const record of exported.header.tensors) {
    let values: KvData;
    if (record.quantization === "int8") {
      const q = new Int8Array(slice(record.byteOffset, record.byteLength));
      const scales = new Float32Array(slice(record.scaleOffset!, record.scaleLength!));
      const restored = dequantizeInt8(q, scales, record.dims, record.scaleAxis!);
      values = record.dtype === "float16" ? Uint16Array.from(restored, float32ToFloat16) : restored;
    } else {
      values = record.dtype === "float16" ? new Uint16Array(slice(record.byteOffset, record.byteLength)) : new Float32Array(slice(record.byteOffset, record.byteLength));
    }
    out.set(record.name, { dims: record.dims, type: record.dtype, data: values });
  }
  return out;
}

/** One buffer: u32 header length, the JSON header, then the body (8-byte aligned). */
export function serializeCache(exported: ExportedCache): Uint8Array {
  const header = new TextEncoder().encode(JSON.stringify(exported.header));
  const start = align8(4 + header.byteLength);
  const out = new Uint8Array(start + exported.body.byteLength);
  new DataView(out.buffer).setUint32(0, header.byteLength, true);
  out.set(header, 4);
  out.set(exported.body, start);
  return out;
}

export function deserializeCache(bytes: Uint8Array | ArrayBuffer): ExportedCache {
  const view = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const length = new DataView(view.buffer, view.byteOffset, view.byteLength).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(view.subarray(4, 4 + length))) as CacheHeader;
  if (header.format !== CACHE_FORMAT) throw new ConfigurationError(`not a KV cache export (format '${String(header.format)}')`);
  if (header.version !== CACHE_VERSION) throw new ConfigurationError(`KV cache export version ${header.version}; this library reads ${CACHE_VERSION}`);
  return { header, body: view.slice(align8(4 + length)) };
}

/** The fields on which two identities differ (empty when the cache fits). */
export function identityMismatch(cache: ModelIdentity, model: ModelIdentity): string[] {
  return (Object.keys(cache) as (keyof ModelIdentity)[]).filter((k) => cache[k] !== model[k]);
}
