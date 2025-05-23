import IOGridPanel   from './IOGridPanel';
import PatternPanel  from './PatternPanel';

/** 右欄底部小工具（已套用深灰卡片背景） */
export default function RightBottomWidgets() {
  return (
    <div className="card">
      <IOGridPanel />
      <PatternPanel />
    </div>
  );
}
