// KV缓存管理器
// 管理KV缓存的生命周期：allocate → compress → transfer → decompress → use → evict

import { ServingRequest } from './types.js';
import { MathUtils } from './utils.js';

/**
 * 缓存条目接口
 */
interface CacheEntry {
  prefixHash: string;
  tokenCount: number;
  sizeBytes: number;
  lastAccessTime: number;
  accessCount: number;
  createdAt: number;
}

/**
 * 缓存统计信息
 */
export interface CacheStats {
  hits: number;
  misses: number;
  evictions: number;
  totalEvictedBytes: number;
  currentSizeBytes: number;
  maxSizeBytes: number;
  hitRate: number;
}

/**
 * KV缓存管理器
 * 使用简化的RadixTree模拟（用Map实现）
 * 支持LRU和LFU驱逐策略
 */
export class KVCacheManager {
  private cache: Map<string, CacheEntry>;
  private maxSizeBytes: number;
  private currentSizeBytes: number;
  private stats: CacheStats;
  private kvBytesPerToken: number;
  private totalLayers: number;
  private timeCounter: number = 0;  // 用于确保时间戳唯一性

  constructor(
    maxSizeBytes: number = 1024 * 1024 * 1024,  // 默认1GB
    kvBytesPerToken: number = 1024,               // 每token约1KB
    totalLayers: number = 32
  ) {
    this.cache = new Map();
    this.maxSizeBytes = maxSizeBytes;
    this.currentSizeBytes = 0;
    this.kvBytesPerToken = kvBytesPerToken;
    this.totalLayers = totalLayers;
    this.stats = {
      hits: 0,
      misses: 0,
      evictions: 0,
      totalEvictedBytes: 0,
      currentSizeBytes: 0,
      maxSizeBytes,
      hitRate: 0
    };
  }

  /**
   * 计算缓存条目大小
   */
  private computeEntrySize(tokenCount: number): number {
    return tokenCount * this.kvBytesPerToken * this.totalLayers;
  }

  /**
   * 获取高精度时间戳（确保唯一性）
   * 使用递增计数器确保在同一毫秒内的操作也能区分
   */
  private getHighResTime(): number {
    this.timeCounter++;
    // 使用微秒 + 计数器确保唯一
    return Math.floor(performance.now() * 1000) + this.timeCounter;
  }

  /**
   * 查找缓存命中
   * @param prefixHash 前缀哈希
   * @returns 命中结果：hit为true时savedTokens表示节省的token数
   */
  lookup(prefixHash: string): { hit: boolean; savedTokens: number } {
    const entry = this.cache.get(prefixHash);
    
    if (entry) {
      // 缓存命中，更新访问信息
      entry.lastAccessTime = this.getHighResTime();
      entry.accessCount++;
      
      this.stats.hits++;
      this.stats.hitRate = this.stats.hits / (this.stats.hits + this.stats.misses);
      
      return {
        hit: true,
        savedTokens: entry.tokenCount
      };
    }
    
    // 缓存未命中
    this.stats.misses++;
    this.stats.hitRate = this.stats.hits / (this.stats.hits + this.stats.misses);
    
    return {
      hit: false,
      savedTokens: 0
    };
  }

  /**
   * 存储缓存条目
   * @param prefixHash 前缀哈希
   * @param tokenCount token数量
   */
  store(prefixHash: string, tokenCount: number): void {
    const requiredSize = this.computeEntrySize(tokenCount);
    
    // 如果新条目太大，直接丢弃旧条目直到有足够空间
    if (requiredSize > this.maxSizeBytes) {
      console.warn(`Cache entry too large: ${requiredSize} bytes > ${this.maxSizeBytes} bytes`);
      return;
    }
    
    // 确保有足够空间：需要释放的字节数
    const spaceNeeded = Math.max(0, this.currentSizeBytes + requiredSize - this.maxSizeBytes);
    if (spaceNeeded > 0) {
      this.evict('lru', spaceNeeded);
    }
    
    // 检查是否已存在（更新）
    const existingEntry = this.cache.get(prefixHash);
    if (existingEntry) {
      this.currentSizeBytes -= existingEntry.sizeBytes;
    }
    
    const entry: CacheEntry = {
      prefixHash,
      tokenCount,
      sizeBytes: requiredSize,
      lastAccessTime: this.getHighResTime(),
      accessCount: 1,
      createdAt: Date.now()
    };
    
    this.cache.set(prefixHash, entry);
    this.currentSizeBytes += requiredSize;
    this.stats.currentSizeBytes = this.currentSizeBytes;
  }

  /**
   * 驱逐缓存条目
   * @param policy 驱逐策略 ('lru' 或 'lfu')
   * @param requiredBytes 需要释放的字节数
   * @returns 实际释放的字节数
   */
  evict(policy: 'lru' | 'lfu', requiredBytes: number): number {
    let freedBytes = 0;
    const entries = Array.from(this.cache.values());
    
    if (policy === 'lru') {
      // LRU: 按最后访问时间排序，最老的先驱逐
      entries.sort((a, b) => a.lastAccessTime - b.lastAccessTime);
    } else {
      // LFU: 按访问次数排序，最少访问的先驱逐
      entries.sort((a, b) => a.accessCount - b.accessCount);
    }
    
    // 驱逐直到释放足够的空间
    for (const entry of entries) {
      if (freedBytes >= requiredBytes) {
        break;
      }
      
      this.cache.delete(entry.prefixHash);
      freedBytes += entry.sizeBytes;
      this.stats.evictions++;
      this.stats.totalEvictedBytes += entry.sizeBytes;
    }
    
    this.currentSizeBytes -= freedBytes;
    this.stats.currentSizeBytes = this.currentSizeBytes;
    
    return freedBytes;
  }

  /**
   * 获取缓存统计信息
   */
  getStats(): CacheStats {
    return { ...this.stats };
  }

  /**
   * 重置统计信息
   */
  resetStats(): void {
    this.stats.hits = 0;
    this.stats.misses = 0;
    this.stats.evictions = 0;
    this.stats.totalEvictedBytes = 0;
    this.stats.hitRate = 0;
  }

  /**
   * 清空所有缓存
   */
  clear(): void {
    this.cache.clear();
    this.currentSizeBytes = 0;
    this.stats.currentSizeBytes = 0;
  }

  /**
   * 获取当前缓存大小（字节）
   */
  getCurrentSize(): number {
    return this.currentSizeBytes;
  }

  /**
   * 获取缓存条目数量
   */
  getEntryCount(): number {
    return this.cache.size;
  }

  /**
   * 检查前缀是否存在（不更新访问信息）
   */
  has(prefixHash: string): boolean {
    return this.cache.has(prefixHash);
  }

  /**
   * 删除指定缓存条目
   */
  remove(prefixHash: string): boolean {
    const entry = this.cache.get(prefixHash);
    if (entry) {
      this.cache.delete(prefixHash);
      this.currentSizeBytes -= entry.sizeBytes;
      this.stats.currentSizeBytes = this.currentSizeBytes;
      return true;
    }
    return false;
  }

  /**
   * 获取缓存利用率
   */
  getUtilization(): number {
    return MathUtils.round(this.currentSizeBytes / this.maxSizeBytes, 4);
  }
}
