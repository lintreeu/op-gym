import { useEffect, useState } from 'react';

interface SafeNumberInputProps {
  value: number | undefined;
  disabled?: boolean;
  readOnly?: boolean;
  onChange: (v: number) => void;
  style?: React.CSSProperties;
}

export default function SafeNumberInput({
  value,
  disabled = false,
  readOnly = false,
  onChange,
  style,
}: SafeNumberInputProps) {
  const [temp, setTemp] = useState<string>(value?.toString() ?? '');

  // 外部 value 更新時同步 temp（除非使用者正在輸入）
  useEffect(() => {
    if (
      document.activeElement &&
      (document.activeElement as HTMLElement).tagName !== 'INPUT'
    ) {
      setTemp(value?.toString() ?? '');
    }
  }, [value]);

  const handleBlur = () => {
    const parsed = +temp;
    if (temp.trim() === '' || isNaN(parsed) || parsed <= 0) {
      // 還原原始值
      setTemp(value?.toString() ?? '');
    } else {
      onChange(parsed);
    }
  };

  return (
    <input
      type="number"
      min={1}  // 禁止 0 與負數
      value={temp}
      disabled={disabled}
      readOnly={readOnly}
      onChange={(e) => setTemp(e.target.value)}
      onBlur={handleBlur}
      style={style}
    />
  );
}
