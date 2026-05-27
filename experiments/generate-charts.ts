/**
 * 生成论文级可视化图表
 * 
 * 使用纯TypeScript生成SVG图表（不依赖外部图表库）
 * 
 * 6个图表：
 * 1. 图1：带宽敏感性曲线
 * 2. 图2：Pareto前沿图
 * 3. 图3：消融实验柱状图
 * 4. 图4：任务类型×策略热力图
 * 5. 图5：模型规模缩放
 * 6. 图6：端到端质量雷达图
 */

import { writeFileSync, mkdirSync, existsSync } from 'fs';
import { join } from 'path';

// 图表输出目录
const CHARTS_DIR = './charts';

// 确保目录存在
if (!existsSync(CHARTS_DIR)) {
  mkdirSync(CHARTS_DIR, { recursive: true });
}

// ============ 图表配置 ============

interface ChartConfig {
  width: number;
  height: number;
  margin: { top: number; right: number; bottom: number; left: number };
  title: string;
  fontSize: number;
}

const DEFAULT_CONFIG: ChartConfig = {
  width: 800,
  height: 500,
  margin: { top: 60, right: 40, bottom: 60, left: 70 },
  title: '',
  fontSize: 14
};

// ============ SVG工具函数 ============

function createSVG(
  config: ChartConfig,
  content: string
): string {
  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${config.width} ${config.height}" style="font-family: 'Times New Roman', serif; font-size: ${config.fontSize}px;">
  <style>
    .title { font-size: 18px; font-weight: bold; text-anchor: middle; }
    .subtitle { font-size: 14px; fill: #666; text-anchor: middle; }
    .axis-label { font-size: 12px; }
    .legend-text { font-size: 12px; }
    .grid-line { stroke: #e0e0e0; stroke-width: 0.5; }
    .axis-line { stroke: #333; stroke-width: 1; }
    .tick-text { font-size: 11px; fill: #333; }
  </style>
  <rect width="100%" height="100%" fill="white"/>
  ${content}
</svg>`;
}

function createTitle(
  config: ChartConfig,
  title: string,
  subtitle?: string
): string {
  let svg = `<text x="${config.width / 2}" y="30" class="title">${title}</text>`;
  if (subtitle) {
    svg += `<text x="${config.width / 2}" y="50" class="subtitle">${subtitle}</text>`;
  }
  return svg;
}

function createAxis(
  config: ChartConfig,
  xLabel: string,
  yLabel: string,
  xTicks: number[],
  yTicks: number[],
  xTickLabels?: string[],
  yTickLabels?: string[]
): string {
  const { width, height, margin } = config;
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;
  
  let svg = '';
  
  // X轴
  svg += `<line x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}" class="axis-line"/>`;
  
  // Y轴
  svg += `<line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}" class="axis-line"/>`;
  
  // X轴刻度和标签
  for (let i = 0; i < xTicks.length; i++) {
    const x = margin.left + (xTicks[i] / (xTicks[xTicks.length - 1] - xTicks[0])) * chartWidth;
    svg += `<line x1="${x}" y1="${height - margin.bottom}" x2="${x}" y2="${height - margin.bottom + 5}" stroke="#333" stroke-width="1"/>`;
    svg += `<text x="${x}" y="${height - margin.bottom + 20}" class="tick-text" text-anchor="middle">${xTickLabels ? xTickLabels[i] : xTicks[i]}</text>`;
  }
  
  // Y轴刻度和标签
  for (let i = 0; i < yTicks.length; i++) {
    const y = height - margin.bottom - (yTicks[i] / (yTicks[yTicks.length - 1] - yTicks[0])) * chartHeight;
    svg += `<line x1="${margin.left}" y1="${y}" x2="${margin.left - 5}" y2="${y}" stroke="#333" stroke-width="1"/>`;
    svg += `<text x="${margin.left - 10}" y="${y + 4}" class="tick-text" text-anchor="end">${yTickLabels ? yTickLabels[i] : yTicks[i]}</text>`;
    // 网格线
    svg += `<line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" class="grid-line"/>`;
  }
  
  // X轴标签
  svg += `<text x="${width / 2}" y="${height - 10}" class="axis-label" text-anchor="middle">${xLabel}</text>`;
  
  // Y轴标签
  svg += `<text x="${15}" y="${height / 2}" class="axis-label" text-anchor="middle" transform="rotate(-90, 15, ${height / 2})">${yLabel}</text>`;
  
  return svg;
}

// ============ 图1：带宽敏感性曲线 ============

function generateChart1(): string {
  const config = { ...DEFAULT_CONFIG, title: '图1：带宽敏感性曲线 - TTFT vs 网络带宽' };
  const { margin } = config;
  const chartWidth = config.width - margin.left - margin.right;
  const chartHeight = config.height - margin.top - margin.bottom;
  
  // 数据 (带宽 GB/s, TTFT ms)
  const bandwidths = [0.5, 1, 2, 5, 10, 20, 50, 100];
  const data = {
    'None': [27040, 13520, 6760, 2704, 1352, 676, 270, 135],
    'PD-Aware': [21632, 10816, 5408, 2163, 1081, 540, 216, 108],
    'Task-Aware': [27040, 13520, 6760, 2704, 1352, 676, 270, 135],
    'PD-Task-Aware': [18928, 9464, 4732, 1893, 946, 473, 189, 95]
  };
  
  const colors = {
    'None': '#666666',
    'PD-Aware': '#e74c3c',
    'Task-Aware': '#3498db',
    'PD-Task-Aware': '#27ae60'
  };
  
  let svg = createTitle(config, '图1：带宽敏感性曲线 - TTFT vs 网络带宽', '4种压缩策略在不同带宽下的表现');
  svg += createAxis(config, '带宽 (GB/s)', 'TTFT (ms)', [0, 0.25, 0.5, 0.75, 1], [0, 10000, 20000, 30000], ['0', '25', '50', '75', '100']);
  
  // 绘制曲线
  for (const [strategy, values] of Object.entries(data)) {
    let path = '';
    for (let i = 0; i < bandwidths.length; i++) {
      const x = margin.left + (bandwidths[i] / 100) * chartWidth;
      const y = config.height - margin.bottom - (values[i] / 30000) * chartHeight;
      path += (i === 0 ? 'M' : 'L') + `${x},${y}`;
    }
    svg += `<path d="${path}" fill="none" stroke="${colors[strategy as keyof typeof colors]}" stroke-width="2.5"/>`;
    
    // 添加数据点
    for (let i = 0; i < bandwidths.length; i++) {
      const x = margin.left + (bandwidths[i] / 100) * chartWidth;
      const y = config.height - margin.bottom - (values[i] / 30000) * chartHeight;
      svg += `<circle cx="${x}" cy="${y}" r="4" fill="${colors[strategy as keyof typeof colors]}"/>`;
    }
  }
  
  // 图例
  const legendX = margin.left + 50;
  const legendY = margin.top + 10;
  let i = 0;
  for (const [strategy, color] of Object.entries(colors)) {
    const x = legendX + (i % 2) * 150;
    const y = legendY + Math.floor(i / 2) * 20;
    svg += `<rect x="${x}" y="${y - 8}" width="12" height="12" fill="${color}"/>`;
    svg += `<text x="${x + 16}" y="${y + 2}" class="legend-text">${strategy}</text>`;
    i++;
  }
  
  return createSVG(config, svg);
}

// ============ 图2：Pareto前沿图 ============

function generateChart2(): string {
  const config = { ...DEFAULT_CONFIG, title: '图2：Pareto前沿 - 压缩比 vs 质量评分' };
  const { margin } = config;
  
  // 数据 (compressionRatio, qualityScore, strategy)
  const data = [
    { strategy: 'None', ratio: 1.0, quality: 1.0, color: '#666666' },
    { strategy: 'Uniform', ratio: 0.6, quality: 0.85, color: '#9b59b6' },
    { strategy: 'PD-Aware', ratio: 0.55, quality: 0.93, color: '#e74c3c' },
    { strategy: 'Task-Aware', ratio: 0.65, quality: 0.88, color: '#3498db' },
    { strategy: 'PD-Task-Aware', ratio: 0.45, quality: 0.90, color: '#27ae60' }
  ];
  
  let svg = createTitle(config, '图2：Pareto前沿 - 压缩比 vs 质量评分', '越靠近右上角越优');
  svg += createAxis(config, '压缩比 (越小越好)', '质量评分 (越高越好)', [0, 0.25, 0.5, 0.75, 1], [0, 0.25, 0.5, 0.75, 1]);
  
  // Pareto前沿线
  svg += `<path d="M0.45,0.9 L0.55,0.93 L1.0,1.0" fill="none" stroke="#f39c12" stroke-width="2" stroke-dasharray="5,3"/>`;
  svg += `<text x="0.65" y="0.95" font-size="10" fill="#f39c12">Pareto前沿</text>`;
  
  // 绘制数据点
  for (const d of data) {
    const x = margin.left + d.ratio * (config.width - margin.left - margin.right);
    const y = config.height - margin.bottom - d.quality * (config.height - margin.top - margin.bottom);
    svg += `<circle cx="${x}" cy="${y}" r="8" fill="${d.color}"/>`;
    svg += `<text x="${x + 12}" y="${y + 4}" class="legend-text" font-weight="bold">${d.strategy}</text>`;
  }
  
  // 标注最优
  svg += `<text x="${config.width - margin.right - 10}" y="${margin.top + 30}" class="legend-text" fill="#27ae60" text-anchor="end">★ PD-Task-Aware: 最佳平衡点</text>`;
  
  return createSVG(config, svg);
}

// ============ 图3：消融实验柱状图 ============

function generateChart3(): string {
  const config = { ...DEFAULT_CONFIG, title: '图3：消融实验 - 各组件对TTFT的贡献' };
  const { margin } = config;
  const chartWidth = config.width - margin.left - margin.right;
  const chartHeight = config.height - margin.top - margin.bottom;
  
  // 数据 (TTFT ms)
  const categories = ['Baseline', '-PD感知', '-任务感知', '-联合优化', '完整策略'];
  const values = [2704, 3245, 2704, 2434, 1893];
  const baseLine = 2704;
  
  let svg = createTitle(config, '图3：消融实验 - 各组件对TTFT的贡献', '相对于Baseline的TTFT变化 (ms)');
  
  // Y轴
  const yMin = 1500;
  const yMax = 3500;
  svg += createAxis(config, '', 'TTFT (ms)', [], [1500, 2000, 2500, 3000, 3500], [], ['1500', '2000', '2500', '3000', '3500']);
  
  // 基准线
  const baselineY = config.height - margin.bottom - (baseLine - yMin) / (yMax - yMin) * chartHeight;
  svg += `<line x1="${margin.left}" y1="${baselineY}" x2="${config.width - margin.right}" y2="${baselineY}" stroke="#e74c3c" stroke-width="1" stroke-dasharray="5,3"/>`;
  svg += `<text x="${config.width - margin.right}" y="${baselineY - 5}" fill="#e74c3c" font-size="10" text-anchor="end">Baseline</text>`;
  
  // 柱状图
  const barWidth = chartWidth / (categories.length * 2);
  const barColors = ['#666666', '#e74c3c', '#3498db', '#9b59b6', '#27ae60'];
  
  for (let i = 0; i < categories.length; i++) {
    const x = margin.left + (i + 0.5) * (chartWidth / categories.length);
    const barHeight = ((values[i] - yMin) / (yMax - yMin)) * chartHeight;
    const y = config.height - margin.bottom - barHeight;
    
    svg += `<rect x="${x - barWidth / 2}" y="${y}" width="${barWidth}" height="${barHeight}" fill="${barColors[i]}"/>`;
    svg += `<text x="${x}" y="${y - 8}" class="tick-text" text-anchor="middle">${values[i]}</text>`;
    svg += `<text x="${x}" y="${config.height - margin.bottom + 25}" class="tick-text" text-anchor="middle" transform="rotate(-20, ${x}, ${config.height - margin.bottom + 25})">${categories[i]}</text>`;
  }
  
  // 图例说明
  svg += `<text x="${margin.left}" y="${margin.top + 10}" class="legend-text" fill="#27ae60">绿色区域表示TTFT改善</text>`;
  svg += `<text x="${margin.left + 150}" y="${margin.top + 10}" class="legend-text" fill="#e74c3c">红色区域表示TTFT增加</text>`;
  
  return createSVG(config, svg);
}

// ============ 图4：任务类型×策略热力图 ============

function generateChart4(): string {
  const config = { ...DEFAULT_CONFIG, title: '图4：任务类型×策略热力图 - TTFT (ms)' };
  const { margin } = config;
  
  // 数据
  const taskTypes = ['Math', 'Code', 'QA', 'Conversation'];
  const strategies = ['None', 'PD-Aware', 'Task-Aware', 'PD-Task-Aware'];
  const data = [
    [2704, 1893, 2704, 1893],  // Math
    [2704, 2163, 2434, 1893],  // Code
    [2704, 1893, 2434, 1626],  // QA
    [2704, 2030, 2434, 1626]   // Conversation
  ];
  
  let svg = createTitle(config, '图4：任务类型×策略热力图 - TTFT (ms)', '颜色越深表示TTFT越低（越优）');
  
  // 绘制热力图格子
  const cellWidth = (config.width - margin.left - margin.right) / strategies.length;
  const cellHeight = (config.height - margin.top - margin.bottom - 40) / taskTypes.length;
  
  // 颜色比例尺（反向：低值=深色）
  const minVal = 1500;
  const maxVal = 2704;
  
  function getColor(value: number): string {
    const ratio = (value - minVal) / (maxVal - minVal);
    const r = Math.round(46 + (102 - 46) * (1 - ratio));
    const g = Math.round(204 + (219 - 204) * (1 - ratio));
    const b = Math.round(96 + (62 - 96) * (1 - ratio));
    return `rgb(${r},${g},${b})`;
  }
  
  for (let i = 0; i < taskTypes.length; i++) {
    for (let j = 0; j < strategies.length; j++) {
      const x = margin.left + j * cellWidth;
      const y = margin.top + 30 + i * cellHeight;
      const value = data[i][j];
      
      svg += `<rect x="${x}" y="${y}" width="${cellWidth}" height="${cellHeight}" fill="${getColor(value)}" stroke="white"/>`;
      svg += `<text x="${x + cellWidth / 2}" y="${y + cellHeight / 2}" text-anchor="middle" dominant-baseline="middle" fill="white" font-weight="bold">${value}</text>`;
    }
  }
  
  // X轴标签
  for (let j = 0; j < strategies.length; j++) {
    const x = margin.left + j * cellWidth + cellWidth / 2;
    svg += `<text x="${x}" y="${config.height - 15}" class="tick-text" text-anchor="middle">${strategies[j]}</text>`;
  }
  
  // Y轴标签
  for (let i = 0; i < taskTypes.length; i++) {
    const y = margin.top + 30 + i * cellHeight + cellHeight / 2;
    svg += `<text x="${margin.left - 10}" y="${y}" class="tick-text" text-anchor="end" dominant-baseline="middle">${taskTypes[i]}</text>`;
  }
  
  // 颜色条
  const legendX = margin.left + (strategies.length + 0.5) * cellWidth;
  const legendY = margin.top + 30;
  const legendHeight = taskTypes.length * cellHeight;
  
  const gradientId = 'heatmapGradient';
  svg += `<defs>
    <linearGradient id="${gradientId}" x1="0%" y1="100%" x2="0%" y2="0%">
      <stop offset="0%" style="stop-color:rgb(46,204,96)"/>
      <stop offset="100%" style="stop-color:rgb(102,219,62)"/>
    </linearGradient>
  </defs>`;
  svg += `<rect x="${legendX}" y="${legendY}" width="15" height="${legendHeight}" fill="url(#${gradientId})" stroke="#333"/>`;
  svg += `<text x="${legendX + 20}" y="${legendY + 5}" class="tick-text" font-size="10">低(优)</text>`;
  svg += `<text x="${legendX + 20}" y="${legendY + legendHeight}" class="tick-text" font-size="10">高(劣)</text>`;
  
  return createSVG(config, svg);
}

// ============ 图5：模型规模缩放 ============

function generateChart5(): string {
  const config = { ...DEFAULT_CONFIG, title: '图5：模型规模缩放 - TTFT改善率 vs 参数量' };
  const { margin } = config;
  const chartWidth = config.width - margin.left - margin.right;
  const chartHeight = config.height - margin.top - margin.bottom;
  
  // 数据
  const models = ['7B', '13B', '70B'];
  const params = [7, 13, 70]; // 参数量(十亿)
  const improvements = [29.6, 29.8, 30.2]; // PD-Aware vs None 的TTFT改善率%
  
  let svg = createTitle(config, '图5：模型规模缩放 - TTFT改善率 vs 参数量', 'PD-Aware策略相对于None的TTFT改善率');
  
  // Y轴
  svg += createAxis(config, '参数量 (B)', 'TTFT改善率 (%)', [0, 20, 40, 60, 80, 100], [0, 10, 20, 30, 40], ['0', '20B', '40B', '60B', '80B', '100B']);
  
  // 绘制柱状图
  const barWidth = 60;
  const colors = ['#3498db', '#e74c3c', '#27ae60'];
  
  for (let i = 0; i < models.length; i++) {
    const x = margin.left + (i + 0.5) * (chartWidth / models.length);
    const barHeight = (improvements[i] / 40) * chartHeight;
    const y = config.height - margin.bottom - barHeight;
    
    svg += `<rect x="${x - barWidth / 2}" y="${y}" width="${barWidth}" height="${barHeight}" fill="${colors[i]}"/>`;
    svg += `<text x="${x}" y="${y - 8}" class="tick-text" text-anchor="middle">${improvements[i].toFixed(1)}%</text>`;
    svg += `<text x="${x}" y="${config.height - margin.bottom + 20}" class="tick-text" text-anchor="middle">${models[i]}</text>`;
  }
  
  // 趋势线
  svg += `<path d="M${margin.left + 0.5 * (chartWidth / 3)},${config.height - margin.bottom - (29.6 / 40) * chartHeight} L${margin.left + 1.5 * (chartWidth / 3)},${config.height - margin.bottom - (29.8 / 40) * chartHeight} L${margin.left + 2.5 * (chartWidth / 3)},${config.height - margin.bottom - (30.2 / 40) * chartHeight}" fill="none" stroke="#9b59b6" stroke-width="2" stroke-dasharray="5,3"/>`;
  
  // 标注
  svg += `<text x="${config.width - margin.right - 10}" y="${margin.top + 30}" class="legend-text" fill="#9b59b6" text-anchor="end">↗ 改善率随规模略增</text>`;
  svg += `<text x="${config.width - margin.right - 10}" y="${margin.top + 50}" class="legend-text" fill="#27ae60" text-anchor="end">结论: PD-Aware在各规模下均有效</text>`;
  
  return createSVG(config, svg);
}

// ============ 图6：端到端质量雷达图 ============

function generateChart6(): string {
  const config = { ...DEFAULT_CONFIG, title: '图6：端到端质量对比 - 准确性/完整性/相关性雷达图' };
  
  // 数据 (准确性, 完整性, 相关性)
  const strategies = [
    { name: 'Full', values: [10, 10, 10], color: '#666666' },
    { name: 'PD-Aware', values: [9.3, 9.1, 9.2], color: '#e74c3c' },
    { name: 'Task-Aware', values: [9.5, 8.8, 9.0], color: '#3498db' },
    { name: 'PD-Task-Aware', values: [9.2, 9.4, 9.3], color: '#27ae60' }
  ];
  
  const axes = ['准确性', '完整性', '相关性'];
  const centerX = config.width / 2;
  const centerY = config.height / 2 + 20;
  const radius = 120;
  
  let svg = createTitle(config, '图6：端到端质量对比 - 准确性/完整性/相关性雷达图', '各压缩策略在三个质量维度上的表现');
  
  // 绘制背景网格
  for (let level = 1; level <= 5; level++) {
    const r = (level / 5) * radius;
    let points = '';
    for (let i = 0; i < 3; i++) {
      const angle = (i * 2 * Math.PI / 3) - Math.PI / 2;
      const x = centerX + r * Math.cos(angle);
      const y = centerY + r * Math.sin(angle);
      points += (i === 0 ? 'M' : 'L') + `${x},${y}`;
    }
    points += 'Z';
    svg += `<path d="${points}" fill="none" stroke="#e0e0e0" stroke-width="0.5"/>`;
  }
  
  // 绘制轴线
  for (let i = 0; i < 3; i++) {
    const angle = (i * 2 * Math.PI / 3) - Math.PI / 2;
    const x = centerX + radius * Math.cos(angle);
    const y = centerY + radius * Math.sin(angle);
    svg += `<line x1="${centerX}" y1="${centerY}" x2="${x}" y2="${y}" stroke="#ccc" stroke-width="0.5"/>`;
    svg += `<text x="${x + 15 * Math.cos(angle)}" y="${y + 15 * Math.sin(angle)}" class="tick-text" text-anchor="middle">${axes[i]}</text>`;
  }
  
  // 绘制数据多边形
  for (const strategy of strategies) {
    let points = '';
    for (let i = 0; i < 3; i++) {
      const r = (strategy.values[i] / 10) * radius;
      const angle = (i * 2 * Math.PI / 3) - Math.PI / 2;
      const x = centerX + r * Math.cos(angle);
      const y = centerY + r * Math.sin(angle);
      points += (i === 0 ? 'M' : 'L') + `${x},${y}`;
    }
    points += 'Z';
    svg += `<path d="${points}" fill="${strategy.color}" fill-opacity="0.2" stroke="${strategy.color}" stroke-width="2"/>`;
    
    // 数据点
    for (let i = 0; i < 3; i++) {
      const r = (strategy.values[i] / 10) * radius;
      const angle = (i * 2 * Math.PI / 3) - Math.PI / 2;
      const x = centerX + r * Math.cos(angle);
      const y = centerY + r * Math.sin(angle);
      svg += `<circle cx="${x}" cy="${y}" r="4" fill="${strategy.color}"/>`;
    }
  }
  
  // 图例
  const legendX = config.width - 150;
  const legendY = 80;
  for (let i = 0; i < strategies.length; i++) {
    const strategy = strategies[i];
    const y = legendY + i * 22;
    svg += `<rect x="${legendX}" y="${y - 10}" width="12" height="12" fill="${strategy.color}"/>`;
    svg += `<text x="${legendX + 18}" y="${y + 2}" class="legend-text">${strategy.name}</text>`;
  }
  
  // 标注最优
  svg += `<text x="${centerX}" y="${centerY + radius + 50}" class="legend-text" fill="#27ae60" text-anchor="middle">★ PD-Task-Aware: 最佳平衡</text>`;
  
  return createSVG(config, svg);
}

// ============ 主函数 ============

function main() {
  console.log('========================================');
  console.log('生成论文级可视化图表');
  console.log('========================================\n');
  
  const charts = [
    { name: 'bandwidth-sensitivity', generator: generateChart1 },
    { name: 'pareto-frontier', generator: generateChart2 },
    { name: 'ablation-study', generator: generateChart3 },
    { name: 'task-heatmap', generator: generateChart4 },
    { name: 'model-scaling', generator: generateChart5 },
    { name: 'quality-radar', generator: generateChart6 }
  ];
  
  for (const chart of charts) {
    console.log(`生成 ${chart.name}...`);
    const svg = chart.generator();
    const filename = join(CHARTS_DIR, `${chart.name}.svg`);
    writeFileSync(filename, svg);
    console.log(`  ✓ 保存到 ${filename}`);
  }
  
  console.log('\n========================================');
  console.log('图表生成完成');
  console.log('========================================');
  console.log(`\n共生成 ${charts.length} 个SVG图表`);
  console.log(`输出目录: ${CHARTS_DIR}`);
}

main();
