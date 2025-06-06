import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';

import CudaBlock3DViewer from './CudaBlock3DViewer';
import type { Access, Dim3 } from '../utils/evaluateAccessOffsets';
import SafeNumberInput from '../utils/SafeNumberInput';

interface LayoutInfo {
    layout: '1d' | 'row-major' | 'col-major' | '3d-row-major' | '3d-col-major';
    dims: { rows?: number; cols?: number; depth?: number };
}

interface Props {
    accesses: Access[];
    blockDim: Dim3;
    blockIdx: Dim3;
    params: Record<string, number>;
    activeKind?: 'load' | 'store';
    layoutMap?: Record<string, LayoutInfo>;
    colors: Record<string, string>;
    elementSizeTable: Record<string, number>;
    onLayoutChange?: (base: string, layout: LayoutInfo['layout']) => void;
    onDimsChange?: (base: string, dims: { rows?: number; cols?: number; depth?: number }) => void;
    onParamsChange?: (param: string, value: number) => void;
}


export default function CudaBlockCanvasGrid(props: Props) {
    const bases = Object.entries(props.colors)
        .map(([k]) => k)
        .sort((a, b) => {
            const aNum = parseInt(a.replace(/\D/g, ''), 10);
            const bNum = parseInt(b.replace(/\D/g, ''), 10);
            return aNum - bNum;
        });

    const getParamSize = (filtered: Access[]) => {
        const base = filtered[0]?.base;
        const v = base ? props.params[base] : undefined;
        return v && v > 0 ? v : 10;
    };

    const computeDims = (filtered: Access[], info: LayoutInfo) => {
        const size = getParamSize(filtered);
        const { layout, dims } = info;
        const rows = dims.rows ?? 1;
        const cols = dims.cols ?? 1;

        if (layout === '1d') return { rows: size };
        if (layout === 'row-major') return { rows, cols: Math.ceil(size / rows) };
        if (layout === 'col-major') return { rows: Math.ceil(size / cols), cols };
        if (layout.startsWith('3d-'))
            return { rows, cols, depth: Math.ceil(size / (rows * cols || 1)) };
        return dims;
    };

    const getBaseSize = (filtered: Access[], info: LayoutInfo) => {
        const { layout } = info;
        const d = computeDims(filtered, info);
        if (layout === '1d') return getParamSize(filtered);
        if (layout === 'row-major' || layout === 'col-major')
            return d.rows! * d.cols!;
        return d.rows! * d.cols! * d.depth!;
    };

    return (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5, width: '100%', height: '100%' }}>
            {bases.map((base) => {
                const filtered = props.accesses.filter(a => a.base === base);
                if (filtered.length === 0) return null;  // 防止錯誤，跳過無 access 的 base

                const info = props.layoutMap?.[base] ?? { layout: '1d', dims: {} };
                const dims = computeDims(filtered, info);
                const baseSize = getBaseSize(filtered, info);
                const size = getParamSize(filtered);

                const updateRows = (rows: number) => {
                    const newDims = { ...dims, rows };
                    if (info.layout === 'row-major') newDims.cols = Math.ceil(size / rows);
                    if (info.layout.startsWith('3d'))
                        newDims.depth = Math.ceil(size / (rows * (newDims.cols ?? 1)));
                    props.onDimsChange?.(base, newDims);
                };

                const updateCols = (cols: number) => {
                    const newDims = { ...dims, cols };
                    if (info.layout === 'col-major')
                        newDims.rows = Math.ceil(size / cols);
                    if (info.layout.startsWith('3d'))
                        newDims.depth = Math.ceil(size / ((newDims.rows ?? 1) * cols));
                    props.onDimsChange?.(base, newDims);
                };

                return (
                    <div
                        key={base}
                        style={{
                            flex: '1 1 calc(33.33% - 16px)',
                            minWidth: 240,
                            maxWidth: 400,
                            display: 'flex',
                            flexDirection: 'column',
                            background: '#fff',
                            borderRadius: 8,
                            boxShadow: '0 1px 1px rgba(0,0,0,0.1)',
                            overflow: 'hidden',
                        }}
                    >
                        {/* ---------- 3D Canvas ---------- */}
                        <div style={{ flex: '1 1 0', position: 'relative' }}>
                            <Canvas
                                style={{ position: 'absolute', inset: 0 }}
                                camera={{ position: [5, 5, 5], fov: 50 }}
                            >
                                <ambientLight />
                                <pointLight position={[10, 10, 10]} />
                                <CudaBlock3DViewer
                                    accesses={filtered}
                                    blockDim={props.blockDim}
                                    blockIdx={props.blockIdx}
                                    params={props.params}
                                    baseSize={baseSize}
                                    colors={props.colors}
                                    activeKind={props.activeKind}
                                    layoutInfo={{ ...info, dims }}
                                />
                                <OrbitControls />
                            </Canvas>
                        </div>

                        {/* ---------- 控制面板 ---------- */}
                        <div style={{ flex: '1 1 0', padding: 8, fontSize: 13 }}>
                            <div style={{ fontWeight: 600 }}>{base}</div>
                            <div>
                                <b>Operation:</b> {filtered[0].kind}
                            </div>
                            <div>
                                <b>Offset:</b>
                                <pre
                                    style={{
                                        backgroundColor: '#eee',
                                        padding: '8px 12px',
                                        borderRadius: '4px',
                                        fontFamily: 'Consolas, monospace',
                                        whiteSpace: 'pre-wrap',
                                        marginTop: '4px',
                                    }}
                                >
                                    {filtered[0].offset}
                                </pre>
                            </div>

                            <div style={{ marginTop: 4 }}>
                                <b>Element (Cube):</b> {filtered[0].eltype} (
                                {props.elementSizeTable?.[filtered[0].eltype] ?? '?'} bytes)
                            </div>

                            {/* Params */}
                            <div style={{ marginTop: 4 }}>
                                <b>Params:</b>
                                <div
                                    style={{
                                        display: 'flex',
                                        flexWrap: 'wrap',
                                        gap: 4,
                                        marginTop: 2,
                                    }}
                                >
                                    {filtered[0].param.map((p) => (
                                        <div key={p}>
                                            {p}:&nbsp;
                                            <SafeNumberInput
                                                value={props.params[p]}
                                                onChange={(v) => props.onParamsChange?.(p, v)}
                                                style={{ width: 60 }}
                                            />
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {/* Layout selector */}
                            <div style={{ marginTop: 6 }}>
                                <label>
                                    <b>Layout:</b>
                                </label>
                                <select
                                    value={info.layout}
                                    onChange={(e) =>
                                        props.onLayoutChange?.(
                                            base,
                                            e.target.value as LayoutInfo['layout']
                                        )
                                    }
                                    style={{ marginLeft: 8 }}
                                >
                                    <option value="1d">1D Row</option>
                                    <option value="row-major">2D Row Major</option>
                                    <option value="col-major">2D Col Major</option>
                                    <option value="3d-row-major">3D Row Major</option>
                                    <option value="3d-col-major">3D Col Major</option>
                                </select>
                            </div>

                            {/* Dimensions */}
                            {info.layout === '1d' ? (
                                <div style={{ marginTop: 4 }}>
                                    <label>Length:</label>
                                    <input
                                        type="number"
                                        readOnly
                                        value={dims.rows}
                                        style={{ width: 80, background: '#eee', marginLeft: 6 }}
                                    />
                                </div>
                            ) : (
                                <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
                                    <div>
                                        <label>Rows:</label>
                                        <SafeNumberInput
                                            value={dims.rows}
                                            disabled={info.layout === 'col-major'}
                                            onChange={updateRows}
                                            style={{ width: 60 }}
                                        />
                                    </div>

                                    {(info.layout === 'row-major' ||
                                        info.layout.includes('col') ||
                                        info.layout.includes('3d')) && (
                                            <div>
                                                <label>Cols:</label>
                                                <SafeNumberInput
                                                    value={dims.cols}
                                                    disabled={info.layout === 'row-major'}
                                                    onChange={updateCols}
                                                    style={{ width: 60 }}
                                                />
                                            </div>
                                        )}

                                    {info.layout.startsWith('3d') && (
                                        <div>
                                            <label>Depth:</label>
                                            <input
                                                type="number"
                                                readOnly
                                                value={dims.depth}
                                                style={{ width: 60, background: '#eee' }}
                                            />
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    </div>
                );
            })}
        </div>
    );
}