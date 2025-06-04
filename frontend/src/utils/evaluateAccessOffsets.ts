// src/utils/evaluateAccessOffsets.ts

export type Access = {
  kind: "load" | "store";
  base: string;
  offset: string;
  eltype: string;
  param: string[];         
};


export type Dim3 = { x: number; y: number; z: number };

/**
 * 將 CUDA 風格的變數名稱替換成合法 JS 名稱
 */
function sanitizeExpr(expr: string): string {
  return expr
    .replace(/\bthreadIdx\.x\b/g, 'threadIdx_x')
    .replace(/\bthreadIdx\.y\b/g, 'threadIdx_y')
    .replace(/\bthreadIdx\.z\b/g, 'threadIdx_z')
    .replace(/\bblockIdx\.x\b/g, 'blockIdx_x')
    .replace(/\bblockIdx\.y\b/g, 'blockIdx_y')
    .replace(/\bblockIdx\.z\b/g, 'blockIdx_z')
    .replace(/\bblockDim\.x\b/g, 'blockDim_x')
    .replace(/\bblockDim\.y\b/g, 'blockDim_y')
    .replace(/\bblockDim\.z\b/g, 'blockDim_z')
    .replace(/\barg(\d+)\b/g, 'arg$1');
}

/**
 * Evaluate memory access expressions across all threads in a block
 */
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
        threadIdx_x: tx,
        threadIdx_y: 0,
        threadIdx_z: 0,
        blockIdx_x: blockIdx.x,
        blockIdx_y: blockIdx.y ?? 0,
        blockIdx_z: blockIdx.z ?? 0,
        blockDim_x: blockDim.x,
        blockDim_y: blockDim.y ?? 1,
        blockDim_z: blockDim.z ?? 1,
        [`element_size_${acc.eltype}`]: 1  // <--- 強制該 element_size 為 1
      };

      const expr = sanitizeExpr(acc.offset);

      try {
        const fn = new Function(...Object.keys(ctx), `return ${expr};`);
        const offset = fn(...Object.values(ctx));
        result[base].add(Math.floor(offset));
      } catch (e) {
        console.warn("eval error:", acc.offset, e);
      }
    }
  }

  return result;
}