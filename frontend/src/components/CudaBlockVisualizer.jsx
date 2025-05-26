import { useEffect } from "react";

export default function CudaBlockVisualizer() {
  useEffect(() => {
    const matrixSize = 512;
    const colChunk = 8;
    const displayCols = matrixSize / colChunk;
    const matrixEl = document.getElementById("matrix");
    const gridSlider = document.getElementById("gridIndex");
    const gridValText = document.getElementById("gridValue");
    const blockSizeSelect = document.getElementById("blockSize");

    let cells = [];

    function createMatrix() {
      matrixEl.innerHTML = "";
      cells = [];
      for (let row = 0; row < matrixSize; row++) {
        let rowCells = [];
        for (let col = 0; col < displayCols; col++) {
          const div = document.createElement("div");
          div.classList.add("cell");
          matrixEl.appendChild(div);
          rowCells.push(div);
        }
        cells.push(rowCells);
      }
    }

    function updateHighlight() {
      const blockSize = parseInt(blockSizeSelect.value);
      const gridIdx = parseInt(gridSlider.value);
      gridValText.textContent = gridIdx;

      for (let row = 0; row < matrixSize; row++) {
        for (let col = 0; col < displayCols; col++) {
          cells[row][col].classList.remove("highlight-row", "highlight-block");
        }
      }

      for (let col = 0; col < displayCols; col++) {
        cells[gridIdx][col].classList.add("highlight-row");
      }

      for (let tid = 0; tid < blockSize; tid++) {
        const featureIdx = tid;
        const visualCol = Math.floor(featureIdx / colChunk);
        if (visualCol < displayCols) {
          cells[gridIdx][visualCol].classList.add("highlight-block");
        }
      }
    }

    gridSlider.addEventListener("input", updateHighlight);
    blockSizeSelect.addEventListener("change", updateHighlight);

    createMatrix();
    updateHighlight();
  }, []);

  return (
    <div style={{ fontFamily: "sans-serif", margin: "20px" }}>
      <h2>CUDA Kernel - Block 可視化</h2>

      <div className="controls" style={{ marginBottom: "20px" }}>
        <label htmlFor="blockSize">Block Size:</label>
        <select id="blockSize" defaultValue="32">
          <option value="16">16</option>
          <option value="32">32</option>
          <option value="64">64</option>
        </select>

        <label htmlFor="gridIndex" style={{ marginLeft: "20px" }}>
          Grid Index (Row):
        </label>
        <input type="range" id="gridIndex" min="0" max="511" defaultValue="0" />
        <span id="gridValue">0</span>
      </div>

      <div
        id="matrix"
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(64, 8px)",
          gridTemplateRows: "repeat(512, 8px)",
          gap: "1px",
        }}
      ></div>

      <style>{`
        .cell {
          width: 8px;
          height: 8px;
          background-color: #eee;
        }
        .highlight-row {
          background-color: #ffe08a;
        }
        .highlight-block {
          background-color: #66c2ff;
        }
      `}</style>
    </div>
  );
}