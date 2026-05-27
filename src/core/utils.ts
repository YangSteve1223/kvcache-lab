// 数学工具函数
export class MathUtils {
  /**
   * 四舍五入到指定小数位
   */
  static round(val: number, decimals: number = 4): number {
    const factor = Math.pow(10, decimals);
    return Math.round(val * factor) / factor;
  }

  /**
   * 计算百分位数
   */
  static percentile(arr: number[], p: number): number {
    if (arr.length === 0) return 0;
    const sorted = [...arr].sort((a, b) => a - b);
    const index = (p / 100) * (sorted.length - 1);
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    if (lower === upper) return sorted[lower];
    return sorted[lower] * (upper - index) + sorted[upper] * (index - lower);
  }

  /**
   * 计算平均值
   */
  static average(arr: number[]): number {
    if (arr.length === 0) return 0;
    return arr.reduce((a, b) => a + b, 0) / arr.length;
  }

  /**
   * 计算总和
   */
  static sum(arr: number[]): number {
    return arr.reduce((a, b) => a + b, 0);
  }
}

/**
 * 确定性随机数生成器 (Mulberry32 PRNG)
 * 使用种子42确保结果可复现
 */
export class DeterministicRandom {
  private state: number;

  constructor(seed: number = 42) {
    this.state = seed;
  }

  /**
   * 生成0-1之间的随机数
   */
  next(): number {
    let t = (this.state += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /**
   * 生成min-max之间的随机整数
   */
  nextInt(min: number, max: number): number {
    return Math.floor(this.next() * (max - min + 1)) + min;
  }

  /**
   * 生成min-max之间的随机浮点数
   */
  nextFloat(min: number, max: number): number {
    return this.next() * (max - min) + min;
  }

  /**
   * 以一定概率返回true
   */
  chance(probability: number): boolean {
    return this.next() < probability;
  }

  /**
   * 从数组中随机选择一个元素
   */
  pick<T>(arr: T[]): T {
    return arr[Math.floor(this.next() * arr.length)];
  }

  /**
   * 打乱数组顺序
   */
  shuffle<T>(arr: T[]): T[] {
    const result = [...arr];
    for (let i = result.length - 1; i > 0; i--) {
      const j = Math.floor(this.next() * (i + 1));
      [result[i], result[j]] = [result[j], result[i]];
    }
    return result;
  }

  /**
   * 重置随机种子
   */
  reset(seed: number = 42): void {
    this.state = seed;
  }
}
