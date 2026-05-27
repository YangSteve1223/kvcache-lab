/**
 * Communication Agent 通信索引
 * 
 * 提供Transmission-Aware Attention相关工具
 */

// 重新导出CommunicationAgent相关
export {
  CommunicationAgent,
  communicationAgent,
  computeTransmissionAwareScores,
  computeCongestionAwareLatency,
  getBetaCoefficient,
} from './CommunicationAgent.js';

export type {
  CommunicationAgentInput,
  CommunicationState,
  TokenLocation,
  CongestionLevel,
} from './CommunicationAgent.js';

// Transmission-Aware Attention核心函数
export {
  computeTransmissionAwareScores,
  computeCongestionAwareLatency,
  getBetaCoefficient,
} from './CommunicationAgent.js';
