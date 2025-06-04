export const GITHUB_RAW_BASE =
  'https://raw.githubusercontent.com/lintreeu/op-gym-challenges/main';

export const METADATA_URL = `${GITHUB_RAW_BASE}/metadata.json`;

export function getKernelUrl(id: string) {
  return `${GITHUB_RAW_BASE}/${id}/kernel.cu`;
}

export function getMainUrl(id: string) {
  return `${GITHUB_RAW_BASE}/${id}/main.cu`;
}
