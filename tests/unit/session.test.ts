import { describe, expect, it } from "vitest";

import { Sampler, argmax } from "../../src/sampling.js";
import { DecoderSession } from "../../src/session.js";

const VOCAB = 8;
const HEADS = 2;
const DIM = 3;

class FakeTensor {
  disposed = false;
  constructor(
    readonly type: string,
    readonly data: Float32Array | BigInt64Array | Uint16Array,
    readonly dims: number[],
  ) {}
  dispose(): void {
    this.disposed = true;
  }
}

/**
 * A two-layer "model" whose cache stores, at each position, the token id itself, and whose logits
 * favour (last token + 1) mod VOCAB — so a wrong cache or a wrong position shows in the tensors.
 */
function fakeBundle() {
  const runs: { ids: number[]; past: number; mask: number; positions: number[] }[] = [];
  const pastNames = ["past_key_values.0.key", "past_key_values.0.value", "past_key_values.1.key", "past_key_values.1.value"];
  const session = {
    filename: "model.onnx",
    backend: "wasm",
    inputs: [{ name: "input_ids" }, { name: "attention_mask" }, { name: "position_ids" }, ...pastNames.map((name) => ({ name, type: "float32", dims: ["batch", HEADS, "past", DIM] }))],
    outputs: [{ name: "logits" }, ...pastNames.map((n) => ({ name: n.replace("past_key_values.", "present.") }))],
    async run(feeds: Record<string, FakeTensor>) {
      const ids = Array.from(feeds["input_ids"]!.data as BigInt64Array, Number);
      const past = feeds["past_key_values.0.key"]!;
      const pastLength = past.dims[2]!;
      runs.push({ ids, past: pastLength, mask: feeds["attention_mask"]!.dims[1]!, positions: Array.from(feeds["position_ids"]!.data as BigInt64Array, Number) });
      const out: Record<string, FakeTensor> = {};
      for (const name of pastNames) {
        const before = feeds[name]!;
        const length = pastLength + ids.length;
        const data = new Float32Array(HEADS * length * DIM);
        for (let h = 0; h < HEADS; h += 1) {
          data.set((before.data as Float32Array).subarray(h * pastLength * DIM, (h + 1) * pastLength * DIM), h * length * DIM);
          ids.forEach((id, i) => data.fill(id, (h * length + pastLength + i) * DIM, (h * length + pastLength + i + 1) * DIM));
        }
        out[name.replace("past_key_values.", "present.")] = new FakeTensor("float32", data, [1, HEADS, length, DIM]);
      }
      const logits = new Float32Array(ids.length * VOCAB);
      ids.forEach((id, i) => (logits[i * VOCAB + ((id + 1) % VOCAB)] = 5));
      out["logits"] = new FakeTensor("float32", logits, [1, ids.length, VOCAB]);
      return out;
    },
  };
  const bundle = { only: session, close: async () => {} };
  return { bundle, runs };
}

const make = (chunk = 512) => {
  const { bundle, runs } = fakeBundle();
  return { session: new DecoderSession(bundle as never, FakeTensor as never, chunk), runs };
};

describe("DecoderSession", () => {
  it("feeds a prompt in chunks with the right positions and masks", async () => {
    const { session, runs } = make(3);
    const logits = await session.feed([1, 2, 3, 4, 5, 6, 7]);
    expect(runs.map((r) => r.ids)).toEqual([[1, 2, 3], [4, 5, 6], [7]]);
    expect(runs.map((r) => r.past)).toEqual([0, 3, 6]);
    expect(runs.map((r) => r.mask)).toEqual([3, 6, 7]);
    expect(runs[1]!.positions).toEqual([3, 4, 5]);
    expect(argmax(logits)).toBe(0);
    expect(session.tokens).toEqual([1, 2, 3, 4, 5, 6, 7]);
  });

  it("reuses the common prefix of the next prompt", async () => {
    const { session, runs } = make();
    await session.prime([1, 2, 3, 4]);
    runs.length = 0;
    const { reused } = await session.prime([1, 2, 3, 4, 5, 6]);
    expect(reused).toBe(4);
    expect(runs.map((r) => r.ids)).toEqual([[5, 6]]);
    expect(runs[0]!.positions).toEqual([4, 5]);
  });

  it("cuts the cache back when the prompt diverges, and reruns the last token of an identical prompt", async () => {
    const { session, runs } = make();
    await session.prime([1, 2, 3, 4, 5]);
    runs.length = 0;
    const { reused } = await session.prime([1, 2, 7, 7]);
    expect(reused).toBe(2);
    expect(runs).toEqual([{ ids: [7, 7], past: 2, mask: 4, positions: [2, 3] }]);
    expect(session.tokens).toEqual([1, 2, 7, 7]);
    runs.length = 0;
    const again = await session.prime([1, 2, 7, 7]);
    expect(again.reused).toBe(3);
    expect(runs.map((r) => r.ids)).toEqual([[7]]);
  });

  it("keeps the cached values of the kept positions when truncating", async () => {
    const { session } = make();
    await session.feed([3, 1, 4, 1, 5]);
    await session.truncate(2);

    const { bundle } = { bundle: session.bundle as unknown as { only: { run: (f: Record<string, FakeTensor>) => Promise<unknown> } } };
    const original = bundle.only.run.bind(bundle.only);
    let seen: number[] = [];
    bundle.only.run = async (feeds) => {
      const past = feeds["past_key_values.1.value"]!;
      seen = Array.from(past.data as Float32Array).filter((_, i) => i % DIM === 0).slice(0, past.dims[2]!);
      return original(feeds);
    };
    await session.feed([9 % VOCAB]);
    expect(seen).toEqual([3, 1]);
  });
});

describe("snapshot and restore", () => {
  it("moves a cache into another session and continues from it", async () => {
    const first = make();
    await first.session.feed([3, 1, 4]);
    const snap = await first.session.snapshot();
    const second = make();
    second.session.restore(snap.tokens, snap.tensors);
    expect(second.session.tokens).toEqual([3, 1, 4]);
    const { reused } = await second.session.prime([3, 1, 4, 1]);
    expect(reused).toBe(3);
    expect(second.runs).toEqual([{ ids: [1], past: 3, mask: 4, positions: [3] }]);
  });

  it("refuses a snapshot whose lengths disagree with its tokens", async () => {
    const first = make();
    await first.session.feed([3, 1, 4]);
    const snap = await first.session.snapshot();
    expect(() => make().session.restore([3, 1], snap.tensors)).toThrow(/positions/);
  });
});

describe("Sampler", () => {
  it("is greedy at temperature 0 and reproducible with a seed", () => {
    const logits = () => Float32Array.from([0.1, 2, 0.5, 1.9]);
    expect(new Sampler().next(logits(), [])).toBe(1);
    const draws = (seed: number) => {
      const sampler = new Sampler({ temperature: 1.5, seed });
      return Array.from({ length: 20 }, () => sampler.next(logits(), []));
    };
    expect(draws(7)).toEqual(draws(7));
    expect(new Set(draws(7)).size).toBeGreaterThan(1);
  });

  it("penalizes repetition and keeps top-k", () => {
    const sampler = new Sampler({ repetitionPenalty: 10 });
    expect(sampler.next(Float32Array.from([0.1, 2, 0.5, 1.9]), [1])).toBe(3);
    const topK = new Sampler({ temperature: 1, topK: 1, seed: 1 });
    for (let i = 0; i < 10; i += 1) expect(topK.next(Float32Array.from([0.1, 2, 0.5, 1.9]), [])).toBe(1);
  });
});
