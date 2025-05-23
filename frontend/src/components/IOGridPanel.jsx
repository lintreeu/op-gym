import { useEffect, useRef } from 'react';
import * as d3 from 'd3';

/** 4×4 input / output grid（含上標題） */
export default function IOGridPanel() {
  const inRef  = useRef(null);
  const outRef = useRef(null);

  useEffect(() => {
    const ROWS = 4, GAP = 2, CELL = 36;

    const build = (div, highlight) => {
      const sel = d3.select(div);
      sel.selectAll('*').remove();
      sel.style('display', 'grid')
         .style('grid-template-columns', `repeat(${ROWS},${CELL}px)`)
         .style('grid-template-rows',    `repeat(${ROWS},${CELL}px)`)
         .style('gap', `${GAP}px`);

      for (let r = 0; r < ROWS; r++)
        for (let c = 0; c < ROWS; c++) {
          const box = sel.append('div')
            .style('width', `${CELL}px`)
            .style('height', `${CELL}px`)
            .style('background', '#333')
            .style('border', '1px solid #444')
            .style('display', 'flex')
            .style('align-items', 'center')
            .style('justify-content', 'center')
            .style('color', '#aaa')
            .style('font-size', '12px')
            .text(`${r}.${c}`);

          if (highlight) {
            if ((r + c) % 3 === 0) box.style('background', '#2e4a78');
            if ((r + c) % 3 === 1) box.style('background', '#6e3a3a');
          }
        }
    };

    build(inRef.current,  true);
    build(outRef.current, false);
  }, []);

  return (
    <div style={{ display: 'flex', gap: '24px' }}>
      <div>
        <h4 style={{ margin: '0 0 4px' }}>input grid layout</h4>
        <div ref={inRef} />
      </div>
      <div>
        <h4 style={{ margin: '0 0 4px' }}>output grid layout</h4>
        <div ref={outRef} />
      </div>
    </div>
  );
}
