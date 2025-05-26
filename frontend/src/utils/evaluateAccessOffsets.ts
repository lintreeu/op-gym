// src/utils/evaluateAccessOffsets.ts

export type Access = {
  kind: "load" | "store";
  base: string;
  offset: string;
  eltype: string;
};

export type Dim3 = { x: number; y: number; z: number };

export function evaluateAccessOffsets(
  accesses: Access[],
  blockDim: Dim3,
  blockIdx: Dim3,
  params: Record<string, number>
): Record<string, Set<number>> {
  const result: Record<string, Set<number>> = {};

  for (const acc of accesses) {
    const base = acc.base;
    if (!result[base]) result[base] = new Set();

    for (let tx = 0; tx < blockDim.x; tx++) {
      const ctx = {
        ...params,
        "threadIdx.x": tx,
        "blockIdx.x": blockIdx.x,
        "blockDim.x": blockDim.x,
        "threadIdx.y": 0,
        "blockIdx.y": 0,
        "blockDim.y": 1,
        "threadIdx.z": 0,
        "blockIdx.z": 0,
        "blockDim.z": 1,
      };

      try {
        const fn = new Function(...Object.keys(ctx), `return ${acc.offset};`);
        const offset = fn(...Object.values(ctx));
        result[base].add(Math.floor(offset)); // offset treated as unit=1
      } catch (e) {
        console.warn("eval error:", acc.offset, e);
      }
    }
  }

  return result;
}
