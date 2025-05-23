import React, { useEffect, useRef } from "react";
import { select } from "d3-selection";
import { sankey, sankeyLinkHorizontal } from "d3-sankey";
import type { SankeyGraph, SankeyLink, SankeyNode } from "d3-sankey";
import {
  sankeyLeft,
  sankeyRight,
  sankeyCenter,
  sankeyJustify
} from "d3-sankey";
import "./MemoryFlowSankey.css";

/* ---------- 配色 ---------- */
const NODE_COLORS: Record<string, string> = {
  "X[0~31]": "#66c2ff",
  "Y[0~31]": "#ffd966",
  "Block 0": "#b06ab3",
  "Warp 0": "#999999",
  "Z[0~31]": "#4c8aff"
};

/* ---------- 型別 ---------- */
type NodeSpec = string | { id: string; w?: number; alpha?: number };
type LayerDef = { tag: string; nodes: NodeSpec[] };

type LinkSpec =
  | [string, string]
  | [string, string, number]
  | { from: string; to: string; v: number; style?: 'solid' | 'hairline' };

interface NodeExtra { name: string; layer: string; }
type SNode = SankeyNode<NodeExtra, {}>;
type SLink = SankeyLink<NodeExtra, {}> & { style?: 'solid' | 'hairline' };

/* ---------- 節點定義 ---------- */
const layers: LayerDef[] = [
  { tag: "global_memory_0", nodes: ["X[0~31]", "Y[0~31]", "H[0~31]"] },
  { tag: "share_memory", nodes: [
      { id: "Block 0", w: 1, alpha: 1.0 },
      { id: "Block 1", w: 1, alpha: 1.0 },
      { id: "Block 2", w: 1, alpha: 1.0 },
  ]},
  { tag: "register", nodes: Array.from({ length: 1 }, (_, i) => `Warp ${i}`) },
   { tag: "global_memory_1", nodes: ["Z[0~31]"] },
];


/* ---------- 連線定義 ---------- */
const links: LinkSpec[] = [
  { from: "X[0~31]", to: "Block 0", v: 2, style: "solid" },
  { from: "Y[0~31]", to: "Block 1", v: 1, style: "solid" },
  { from: "Y[0~31]", to: "Block 2", v: 1, style: "solid" },
  { from: "Block 0", to: "Warp 0", v: 2, style: "hairline" },
  { from: "Block 1", to: "Warp 0", v: 1, style: "hairline" },
  { from: "Block 2", to: "Warp 0", v: 1, style: "hairline" },
  { from: "H[0~31]", to: "Warp 0", v: 2, style: "hairline" },
   { from: "Warp 0", to: "Z[0~31]", v: 2, style: "hairline" }
];


/* ---------- Slider ---------- */
function Slider(props: {
  label: string;
  unit: string;
  min: number;
  max: number;
  defaultValue: number;
}) {
  const { label, unit, min, max, defaultValue } = props;
  const [val, setVal] = React.useState(defaultValue);
  return (
    <div style={{ marginBottom: 28 }}>
      <div style={{ fontSize: 14, marginBottom: 6 }}>
        {label} <b>{val}{unit}</b>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        value={val}
        onChange={e => setVal(+e.target.value)}
        style={{ width: "100%", accentColor: "#0e4a62" }}
      />
    </div>
  );
}

export default function MemoryFlowSankey() {
  const svgRef = useRef<SVGSVGElement>(null);
  const tipRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const widthFactor = new Map<string, number>();
    const nodeAlpha = new Map<string, number>();
    const idList: string[] = [];

    layers.forEach(l => {
      l.nodes.forEach(n => {
        const id = typeof n === "string" ? n : n.id;
        idList.push(id);
        widthFactor.set(id, typeof n === "string" ? 1 : n.w ?? 1);
        nodeAlpha.set(id, typeof n === "string" ? 1 : n.alpha ?? 1);
      });
    });

    const nodes: SNode[] = idList.map(id => ({
      name: id,
      layer: layers.find(L =>
        L.nodes.some(n => typeof n === "string" ? n === id : n.id === id)
      )!.tag,
    }) as unknown as SNode);

    const linkObjs: SLink[] = links.map(l => {
      if (Array.isArray(l)) {
        const [s, t, val] = l;
        return { source: s, target: t, value: val ?? 1, style: "solid" } as SLink;
      }
      return { source: l.from, target: l.to, value: l.v, style: l.style ?? "solid" } as SLink;
    });

    const graph: SankeyGraph<NodeExtra, {}> = { nodes, links: linkObjs };

    const baseW = 16, width = 640, height = 380;
    const sk = sankey<NodeExtra, {}>()
      .nodeWidth(baseW)
      .nodePadding(22)
      .size([width, height])
      .nodeId((d: any) => d.name)
      .nodeAlign(n => layers.findIndex(l => l.tag === n.layer) as any);

    const { nodes: ln, links: ll } = sk(graph);

    ln.forEach(nd => {
      const f = widthFactor.get(nd.name) ?? 1;
      nd.x1 = nd.x0! + baseW * f;
    });

    const svg = select(svgRef.current!)
      .attr("viewBox", `0 0 ${width} ${height}`)
      .style("font-family", "Inter, sans-serif");
    svg.selectAll("*").remove();

    /* ----- 髮絲線 pattern 定義 ----- */
    const defs = svg.append("defs");
    defs.append("pattern")
      .attr("id", "hairlinePattern")
      .attr("patternUnits", "userSpaceOnUse")
      .attr("width", 640)
      .attr("height", 6)
      .append("rect")
      .attr("x", 0).attr("y", 0)
      .attr("width", 640)
      .attr("height", 2)
      .attr("fill", "#4c8aff");

    const tip = select(tipRef.current!)
      .style("position", "fixed").style("pointer-events", "none")
      .style("padding", "6px 10px").style("font-size", "12px")
      .style("background", "#fff").style("border", "1px solid #ccc")
      .style("border-radius", "4px")
      .style("box-shadow", "0 2px 6px rgba(0,0,0,.15)")
      .style("opacity", 0);

    const linkSel = svg.append("g").attr("fill", "none")
      .selectAll("path").data(ll).join("path")
      .attr("d", sankeyLinkHorizontal())
      .attr("stroke", d => d.style === "hairline"
        ? "url(#hairlinePattern)"
        : NODE_COLORS[(d.source as any).name] ?? "#ccc"
      )
      .attr("stroke-width", d => Math.max(1, d.width!))
      .attr("stroke-opacity", d => {
        const a1 = nodeAlpha.get((d.source as any).name) ?? 1;
        const a2 = nodeAlpha.get((d.target as any).name) ?? 1;
        return 0.75 * Math.min(a1, a2);
      })
      .style("cursor", "pointer")
      .on("click", function (evt, d: any) {
        clearSel();
        select(this)
          .attr("stroke", "#000")
          .attr("stroke-width", Math.max(1, d.width!) + 2);
        showTip(evt, `<strong>${d.source.name} → ${d.target.name}</strong><br/>value: ${d.value}`);
        evt.stopPropagation();
      });

    const nodeG = svg.append("g").selectAll("g").data(ln).join("g")
      .style("cursor", "pointer")
      .on("click", function (evt, d: any) {
        clearSel();
        select(this).select("rect").attr("stroke", "black").attr("stroke-width", 3);
        showTip(evt, `<strong>${d.name}</strong><br/>layer: ${d.layer}`);
        evt.stopPropagation();
      });

    nodeG.append("rect")
      .attr("x", d => d.x0!).attr("y", d => d.y0!)
      .attr("width", d => d.x1! - d.x0!).attr("height", d => d.y1! - d.y0!)
      .attr("fill", d => NODE_COLORS[d.name] ?? "#d3d3d3")
      .attr("fill-opacity", d => nodeAlpha.get(d.name) ?? 1)
      .attr("stroke", "#333").attr("stroke-width", 1);

    nodeG.append("text")
      .attr("x", d => (d.x0! < width / 2 ? d.x1! + 6 : d.x0! - 6))
      .attr("y", d => (d.y0! + d.y1!) / 2).attr("dy", ".35em")
      .attr("text-anchor", d => (d.x0! < width / 2 ? "start" : "end"))
      .attr("fill", "#ffffff")
      .style("font-size", "12px")
      .text(d => d.name);

    function clearSel() {
      svg.selectAll("rect").attr("stroke", "#333").attr("stroke-width", 1);
      linkSel
        .attr("stroke", d => d.style === "hairline"
          ? "url(#hairlinePattern)"
          : NODE_COLORS[(d.source as any).name] ?? "#ccc"
        )
        .attr("stroke-width", d => Math.max(1, d.width!));
      tip.style("opacity", 0);
    }

    function showTip(evt: any, html: string) {
      tip.style("opacity", 1).html(html)
        .style("left", `${evt.clientX + 14}px`).style("top", `${evt.clientY + 14}px`);
    }

    svg.on("click", () => clearSel());

  }, []);

  return (
    <>
      <header className="control-panel">
        <div className="controls">
          <button id="play">▶</button>
          Epoch <input id="epoch" defaultValue="000000" size={6} />
          <select><option>grid size</option></select>
          <select><option>block size</option></select>
          <select><option>warp / CTA</option></select>
        </div>
      </header>

      <div style={{ display: "flex", padding: 10 }}>
        <div style={{ width: 170, marginRight: 24 }}>
          <Slider label="Ratio of training to test data:" unit="%" min={0} max={100} defaultValue={10} />
          <Slider label="Noise:" unit="" min={0} max={10} defaultValue={0} />
          <Slider label="Batch size:" unit="" min={1} max={128} defaultValue={10} />
        </div>
        <div style={{ flex: 1, position: "relative" }}>
          <svg ref={svgRef} width="100%" height="380" />
        </div>
      </div>

      <div ref={tipRef} />
    </>
  );
}
