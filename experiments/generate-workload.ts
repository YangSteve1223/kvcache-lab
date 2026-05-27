/**
 * 生成模拟的真实场景请求trace
 * 当API不可用时，使用本地生成模拟数据
 */

import { writeFileSync } from 'fs';

// 输出文件
const OUTPUT_FILE = './data/real-workload.json';

// 任务类型
type TaskType = 'math' | 'code' | 'qa' | 'conversation';

interface WorkloadRequest {
  id: string;
  taskType: TaskType;
  inputTokens: number;
  outputTokens: number;
  content: string;
}

// 数学推理上下文模板
const MATH_TEMPLATES = [
  `设函数 f(x) = x^3 - 3x^2 + 4x - 12，求其导数并分析单调性。

已知条件：
1. f'(x) = 3x^2 - 6x + 4
2. 判别式 Δ = (-6)^2 - 4*3*4 = 36 - 48 = -12 < 0

证明：由于导数恒正（因为二次项系数3>0且判别式<0），函数f(x)在R上单调递增。

进一步分析极值点：令f'(x) = 0，解得...`,

  `证明：若数列{a_n}满足 a_1 = 1, a_{n+1} = 2a_n + 1，则通项公式为 a_n = 2^n - 1。

数学归纳法证明：
基例：当n=1时，a_1 = 1 = 2^1 - 1，成立。
归纳假设：假设当n=k时，a_k = 2^k - 1成立。
归纳步骤：当n=k+1时，
  a_{k+1} = 2a_k + 1 = 2(2^k - 1) + 1 = 2^{k+1} - 2 + 1 = 2^{k+1} - 1
故结论成立。`,

  `设矩阵 A = [[2,1],[1,2]]，求其特征值和特征向量。

解：特征方程 |A - λE| = 0
即 (2-λ)(2-λ) - 1 = λ^2 - 4λ + 3 = 0
解得 λ_1 = 1, λ_2 = 3

对应特征向量：
当λ_1=1时，(A-E)v=0，解得v_1=[1,-1]^T
当λ_2=3时，(A-3E)v=0，解得v_2=[1,1]^T`
];

// 代码生成上下文模板
const CODE_TEMPLATES = [
  `# 用户认证服务 API 文档

## 接口概述
本服务提供用户注册、登录、认证等功能。

## API 接口

### POST /api/auth/register
注册新用户

**请求参数：**
- username: string (必填, 3-20字符)
- email: string (必填, 有效邮箱格式)
- password: string (必填, 8位以上)
- age?: number (可选)

**响应示例：**
\`\`\`json
{
  "code": 200,
  "data": {
    "userId": "u123456",
    "token": "eyJhbGciOiJIUzI1NiIs..."
  }
}
\`\`\`

**Python实现示例：**
\`\`\`python
from pydantic import BaseModel, EmailStr
from typing import Optional

class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str
    age: Optional[int] = None
    
    def validate(self):
        if len(self.username) < 3 or len(self.username) > 20:
            raise ValueError("用户名长度必须在3-20字符之间")
        if len(self.password) < 8:
            raise ValueError("密码至少8位")
\`\`\``,

  `TypeScript 类型定义 - 数据模型

\`\`\`typescript
// 用户模型
interface User {
  id: string;
  username: string;
  email: string;
  profile: UserProfile;
  settings: UserSettings;
  createdAt: Date;
}

// 用户资料
interface UserProfile {
  avatar?: string;
  bio?: string;
  age?: number;
  interests: string[];
}

// 用户设置
interface UserSettings {
  theme: 'light' | 'dark';
  language: string;
  notifications: NotificationSettings;
}

// 通知设置
interface NotificationSettings {
  email: boolean;
  push: boolean;
  sms: boolean;
}

// 工厂函数
function createUser(data: Partial<User>): User {
  return {
    id: data.id || generateId(),
    username: data.username || '',
    email: data.email || '',
    profile: { interests: [], ...data.profile },
    settings: { theme: 'light', language: 'zh-CN', notifications: { email: true, push: true, sms: false }, ...data.settings },
    createdAt: new Date()
  };
}
\`\`\``,

  `分布式缓存系统设计

## 核心组件
1. CacheNode: 单个缓存节点
2. ConsistentHash: 一致性哈希环
3. ReplicationManager: 副本管理器

\`\`\`python
import hashlib
from typing import Dict, List, Optional
import threading

class ConsistentHash:
    def __init__(self, nodes: List[str], virtual_nodes: int = 150):
        self.ring: Dict[int, str] = {}
        self.sorted_keys: List[int] = []
        self._build_ring(nodes, virtual_nodes)
    
    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def _build_ring(self, nodes: List[str], vnodes: int):
        for node in nodes:
            for i in range(vnodes):
                hash_key = self._hash(f"{node}#vn{i}")
                self.ring[hash_key] = node
        self.sorted_keys = sorted(self.ring.keys())
\`\`\``
];

// QA问答上下文模板
const QA_TEMPLATES = [
  `## Transformer架构详解

Transformer是一种基于自注意力机制的神经网络架构，由Google在2017年提出。它彻底改变了自然语言处理领域的研究范式。

### 核心组件

**1. Self-Attention机制**
自注意力机制的核心公式：
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

其中：
- Q (Query): 查询向量，表示当前位置想要关注的内容
- K (Key): 键向量，用于匹配查询
- V (Value): 值向量，包含实际的信息内容
- d_k: 键向量的维度，用于缩放

**2. 多头注意力 (Multi-Head Attention)**
将注意力分成多个头，允许模型在不同的表示子空间学习信息。

**3. 位置编码 (Positional Encoding)**
由于Transformer没有循环结构，需要添加位置编码来注入序列位置信息。`,

  `## Kubernetes容器编排系统

Kubernetes（简称K8s）是一个开源的容器编排平台，用于自动化容器化应用的部署、扩缩容和管理。

### 核心概念

**Pod**: Kubernetes的基本调度单元，一个Pod可以包含一个或多个容器。

**Service**: 抽象的服务发现机制，为一组Pod提供稳定的访问入口。

**Deployment**: 声明式的Pod管理，用于定义应用的部署状态。

**ConfigMap和Secret**: 管理配置数据和敏感信息。

### 自动扩缩容机制
HPA (Horizontal Pod Autoscaler) 根据CPU利用率或自定义指标自动调整Pod副本数。
VPA (Vertical Pod Autoscaler) 自动调整Pod的资源请求。
Cluster Autoscaler 在节点级别进行扩缩容。`,

  `## 数据库索引原理与优化

### B+树索引结构
B+树是一种自平衡的多路搜索树，所有记录都存储在叶子节点，叶子节点之间用链表连接。

**优势：**
1. 高度平衡，查询效率稳定 O(log n)
2. 叶子节点链表便于范围查询
3. 磁盘IO次数少（高度通常为3-4层）

### 索引优化策略
1. 选择性高的列优先建立索引
2. 联合索引遵循最左前缀原则
3. 避免在索引列上使用函数
4. 覆盖索引减少回表查询
5. 考虑使用部分索引`
];

// 对话上下文模板
const CONVERSATION_TEMPLATES = [
  `用户：我正在设计一个高并发的Web系统，预计日活100万，需要支持每秒10万请求。想请教一下架构设计的问题。

顾问：好的，这是一个典型的大流量系统设计问题。首先我需要了解几个关键信息：
1. 业务类型是什么？是读多写少还是读写均衡？
2. 数据的一致性要求有多高？
3. 预算是多少？

用户：主要是社交类应用，读多写少，数据一致性要求中等。

顾问：明白。基于你的场景，我建议采用以下架构：
1. 前端使用CDN加速静态资源
2. LVS+Nginx做负载均衡
3. 应用服务器用Kubernetes容器化部署，支持自动扩缩容
4. 缓存层：Redis集群做热点数据缓存
5. 数据库：主从复制+读写分离...

用户：关于Redis缓存，有什么最佳实践吗？

顾问：几个关键点：
1. key设计要简洁，带业务前缀便于管理
2. value避免过大，单个key不超过10MB
3. 设置合理的过期时间
4. 热点数据预加载到缓存
5. 缓存穿透、击穿、雪崩的防护措施`,

  `用户：我们公司的微服务架构遇到了性能瓶颈，API响应时间从100ms增加到了2秒。请帮我分析一下可能的原因。

顾问：这是一个很常见的问题。让我从几个维度来分析：

1. **网络层面**
   - 检查服务间调用的网络延迟
   - 是否有跨机房调用
   - 网络带宽是否成为瓶颈

2. **数据库层面**
   - 慢查询分析
   - 索引是否失效
   - 连接池是否耗尽

3. **应用层面**
   - GC频繁导致停顿
   - 线程池配置不合理
   - 同步调用阻塞

4. **缓存层面**
   - 缓存命中率下降
   - 缓存穿透
   - 大key问题

用户：排查后发现是某个微服务频繁Full GC导致的，怎么优化？

顾问：可以从以下几个方面优化JVM：
1. 升级到G1或ZGC垃圾收集器
2. 调整堆大小和年轻代比例
   - 建议堆大小为容器内存的50-75%
   - 年轻代占比可以提高到60%
3. 优化代码减少对象创建
4. 使用对象池复用大对象`
];

// 生成随机请求
function generateRequest(
  taskType: TaskType,
  index: number
): WorkloadRequest {
  const id = `${taskType}_${String(index).padStart(3, '0')}`;
  
  let content: string;
  let inputTokens: number;
  let outputTokens: number;
  
  switch (taskType) {
    case 'math':
      content = MATH_TEMPLATES[index % MATH_TEMPLATES.length];
      inputTokens = 800 + Math.floor(Math.random() * 400);
      outputTokens = 200 + Math.floor(Math.random() * 300);
      break;
    case 'code':
      content = CODE_TEMPLATES[index % CODE_TEMPLATES.length];
      inputTokens = 600 + Math.floor(Math.random() * 500);
      outputTokens = 150 + Math.floor(Math.random() * 250);
      break;
    case 'qa':
      content = QA_TEMPLATES[index % QA_TEMPLATES.length];
      inputTokens = 700 + Math.floor(Math.random() * 300);
      outputTokens = 180 + Math.floor(Math.random() * 220);
      break;
    case 'conversation':
      content = CONVERSATION_TEMPLATES[index % CONVERSATION_TEMPLATES.length];
      inputTokens = 500 + Math.floor(Math.random() * 400);
      outputTokens = 100 + Math.floor(Math.random() * 200);
      break;
  }
  
  return { id, taskType, inputTokens, outputTokens, content };
}

// 主函数
function main() {
  console.log('========================================');
  console.log('生成模拟真实Workload数据');
  console.log('========================================\n');
  
  const requests: WorkloadRequest[] = [];
  
  // 生成4种场景的请求
  const scenarios: Array<{ type: TaskType; count: number }> = [
    { type: 'math', count: 50 },
    { type: 'code', count: 50 },
    { type: 'qa', count: 50 },
    { type: 'conversation', count: 50 }
  ];
  
  for (const scenario of scenarios) {
    console.log(`生成 ${scenario.type} 场景 (${scenario.count}个)...`);
    for (let i = 1; i <= scenario.count; i++) {
      requests.push(generateRequest(scenario.type, i));
    }
  }
  
  // 统计
  const stats = {
    total: requests.length,
    byType: {
      math: requests.filter(r => r.taskType === 'math').length,
      code: requests.filter(r => r.taskType === 'code').length,
      qa: requests.filter(r => r.taskType === 'qa').length,
      conversation: requests.filter(r => r.taskType === 'conversation').length
    },
    avgInputTokens: Math.round(
      requests.reduce((sum, r) => sum + r.inputTokens, 0) / requests.length
    ),
    avgOutputTokens: Math.round(
      requests.reduce((sum, r) => sum + r.outputTokens, 0) / requests.length
    )
  };
  
  // 保存结果
  writeFileSync(OUTPUT_FILE, JSON.stringify(requests, null, 2));
  
  console.log('\n========================================');
  console.log('生成完成');
  console.log('========================================');
  console.log(`\n✓ 总请求数: ${stats.total}`);
  console.log(`  - math: ${stats.byType.math}`);
  console.log(`  - code: ${stats.byType.code}`);
  console.log(`  - qa: ${stats.byType.qa}`);
  console.log(`  - conversation: ${stats.byType.conversation}`);
  console.log(`\n✓ 平均输入tokens: ${stats.avgInputTokens}`);
  console.log(`✓ 平均输出tokens: ${stats.avgOutputTokens}`);
  console.log(`\n✓ 已保存到: ${OUTPUT_FILE}`);
}

main();
