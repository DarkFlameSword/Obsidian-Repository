---
date: 2026-04-01
author:
  - Siyuan Liu
tags:
  - summary
---
## 背景

OpenAI 发布了一个插件，可以直接在 Anthropic 的 Claude Code 中嵌入 Codex，让开发者无需离开现有工作流就能运行代码审查和任务委托。这是首个专为竞争对手编码环境设计的官方 OpenAI 集成。

---

## 前置要求

需要 ChatGPT 订阅（含免费版）或 OpenAI API Key，以及 Node.js 18.18 或更高版本。

---

## 安装步骤

**第一步：添加到 marketplace**

```
/plugin marketplace add openai/codex-plugin-cc
```

执行后会询问你希望插件在哪个作用域生效（用户级或项目级），选择你需要的那个。

**第二步：安装插件**

```
/plugin install codex@openai-codex
```

**第三步：初始化设置**

```
/codex:setup
```

这个命令会检测 Codex 是否已安装就绪。如果缺少 Codex 且 npm 可用，它会提示你安装。

> 如果命令不被识别，先执行 `/reload-plugins` 刷新插件列表再重试。

**第四步：登录认证**

```
!codex login
```

用 ChatGPT 账号或 API Key 完成认证。

---

## 可用命令

插件提供六个斜线命令：

|命令|说明|
|---|---|
|`/codex:review`|标准只读代码审查，与在 Codex 中直接运行 `/review` 效果相同|
|`/codex:adversarial-review`|对抗性审查，专门质疑实现决策、权衡和失败模式|
|`/codex:rescue`|把任务完全交给 Codex，让它调查 bug 或尝试修复|
|`/codex:status`|查看后台任务进度|
|`/codex:result`|查看后台任务结果|
|`/codex:cancel`|取消进行中的后台任务|

---

## 可选：Review Gate（谨慎使用）

```
/codex:setup --enable-review-gate
```

这会启用一个"stop hook"，Codex 自动审查 Claude Code 的输出，如果检测到问题则阻止输出直到修复完成。但 README 有明确警告：Claude Code 和 Codex 之间的循环可能迅速消耗使用额度，建议仅在人工监督下启用。

---

## 典型使用场景

实践中的推荐工作流：用 Claude Code 的 Opus 4.6 做架构规划和复杂功能开发，然后把针对性的 bug 修复交给 Codex 的 gpt-5.4-mini 处理（速度优先于推理深度）；或者让 Claude Code 写实现，在每次合并前让 Codex 做对抗性审查。