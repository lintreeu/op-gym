import { useEffect, useRef } from 'react';
import * as d3 from 'd3';

/** 10×10 pattern + intensity bar */
export default function PatternPanel() {
  const svgRef = useRef(null);

  useEffect(() => {
    const SIZE = 180;
    const CELL = SIZE / 10;

    const svg = d3.select(svgRef.current)
      .attr('width', SIZE)
      .attr('height', SIZE);

    svg.selectAll('*').remove();

    for (let y = 0; y < 10; y++)
      for (let x = 0; x < 10; x++)
        svg.append('rect')
          .attr('x', x * CELL)
          .attr('y', y * CELL)
          .attr('width', CELL - 1)
          .attr('height', CELL - 1)
          .attr('fill', (x === 4 && y === 5) ? '#ff9900' : '#444')
          .attr('stroke', '#666');
  }, []);

  return (
    <div style={{ marginTop: '1rem' }}>
      <h4 style={{ margin: '0 0 4px' }}>each block pattern (enlarge)</h4>
      <svg ref={svgRef} />
      
    </div>
  );
}
